"""Few-shot multi-turn emotion classification with chain-of-thought prompting.

Loads DailyDialog, builds 4 turn-by-turn few-shot prompts adapted to the target
model's chat template, generates completions, extracts the predicted emotion
from each completion, and reports per-turn / overall metrics.
"""

import json
import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    T5ForConditionalGeneration,
)

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS: List[str] = [
    "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral",
]
EMOTION_OPTIONS: List[str] = [
    "[A] Anger", "[B] Disgust", "[C] Fear", "[D] Happiness",
    "[E] Sadness", "[F] Surprise", "[G] Neutral",
]
EMOTION_LABEL_MAP = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise",
}

OPTIONS = (
    "[A] Anger, [B] Disgust, [C] Fear, [D] Happiness, "
    "[E] Sadness, [F] Surprise or [G] Neutral."
)
SYSTEM_INTRO = (
    "You are an emotional classifier. Follow the logic of the following example. "
    "Be aware of the emotional context of the dialogue."
)
SYSTEM_TRIGGER = (
    "Apply the logic learned in the example. "
    "The answer must end with one of the different possible options."
)

EXAMPLES: List[Tuple[str, str]] = [
    (
        "Are things still going badly with your houseguest?",
        "Let's think step by step. Step 1: Assess the information about the situation and the given context. A personal question is asked to find out the status of another person. Step 2: So the most likely option is [G] Neutral.",
    ),
    (
        "Getting worse. Now he's eating me out of house and home. I've tried talking to him but it all goes in one ear and out the other. He makes himself at home, which is fine. But what really gets me is that yesterday he walked into the living room in the raw and I had company over! That was the last straw.",
        "Let's think step by step. Step 1: Appraise information about the given situation and context. The person feels frustrated and helpless for not being able to live together. Step 2: So the most likely option is [A] Anger.",
    ),
    (
        "Leo, I really think you're beating around the bush with this guy. I know he used to be your best friend in college, but I really think it's time to lay down the law.",
        "Let's think step by step. Step 1: Evaluate information about the given situation and context. The person seems to care about Leo's well-being and wants him to act to resolve the situation, giving him a possible solution. Step 2: So the most likely option is [G] Neutral.",
    ),
]

DIALOGUE_TURNS = 4


# ---------------------------------------------------------------------------
# Configuration & special tokens
# ---------------------------------------------------------------------------


_FAMILY_KEYS = (
    "automodel", "automodelforcasuallm", "llamaforcasuallm",
    "automodelforseq2seqlm", "t5forconditionalgeneration",
    "trustedremotecode",
    "repeatedprompts", "repeatedpromptsllama3", "repeatedpromptsgemma",
    "repeatedpromptsmistral", "repeatedpromptschatglm3", "repeatedpromptsglm4",
    "repeatedpromptsbloomz", "repeatedpromptsphi3", "repeatedpromptsairboros",
    "repeatedpromptszephyralpha", "repeatedpromptszephyrbeta",
    "repeatedpromptszephyrgemma", "repeatedpromptsinternlm",
    "inputtemplate1", "inputtemplate2", "inputtemplate3",
    "inputtemplatellama2", "inputtemplatellama3", "inputtemplategemma",
    "inputtemplatephi3", "inputtemplatemistral", "inputtemplatechatglm3",
    "inputtemplateglm4", "inputtemplateinternlm",
)


@dataclass
class Config:
    """Runtime configuration loaded from a JSON file."""

    dataset: str
    model_name: str
    gpu: int
    debug: bool
    families: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path="config.json") -> "Config":
        with open(path, encoding="utf-8") as fp:
            cfg = json.load(fp)
        args = cfg["args"]
        return cls(
            dataset=args["dataset"],
            model_name=args["model"],
            gpu=args["gpu"],
            debug=args["dev"],
            families={k: cfg[k]["models"] for k in _FAMILY_KEYS},
        )

    @property
    def model_short_name(self) -> str:
        return self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name

    @property
    def data_dir(self) -> Path:
        return Path("data") / self.dataset

    @property
    def model_dir(self) -> Path:
        return self.data_dir / self.model_short_name

    def in_family(self, family: str) -> bool:
        return self.model_short_name in self.families.get(family, ())

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PromptTokens:
    """Special tokens used by chat-template prompt builders."""

    bos: str = ""
    system_start: str = ""
    system_end: str = ""
    system: str = ""
    user: str = ""
    assistant: str = ""
    eos: str = ""

    @classmethod
    def for_config(cls, cfg: Config) -> "PromptTokens":
        if cfg.in_family("inputtemplate1"):
            return cls(user="\nUSER:", assistant="\nASSISTANT:\n")
        if cfg.in_family("inputtemplate2"):
            return cls(user="\n### Instruction:", assistant="\n### Response:\n")
        if cfg.in_family("inputtemplate3"):
            return cls(system="<|system|>", user="<|user|>",
                       assistant="<|assistant|>", eos="</s>")
        if cfg.in_family("inputtemplatellama2"):
            return cls(bos="<s>", system_start="<<SYS>>", system_end="<</SYS>>",
                       user="[INST]", assistant="[/INST]", eos="</s>")
        if cfg.in_family("inputtemplatellama3"):
            return cls(bos="<|begin_of_text|>", user="<|start_header_id|>",
                       assistant="<|end_header_id|>", eos="<|eot_id|>")
        if cfg.in_family("inputtemplategemma"):
            return cls(user="<start_of_turn>user", assistant="<start_of_turn>model",
                       eos="<end_of_turn>")
        if cfg.in_family("inputtemplatephi3"):
            return cls(user="<|user|>", assistant="<|assistant|>", eos="<|end|>")
        if cfg.in_family("inputtemplatemistral"):
            return cls(bos="<s>", user="[INST] ", assistant="[/INST]", eos="</s>")
        if cfg.in_family("inputtemplatechatglm3"):
            return cls(system="<|system|>", user="<|user|>", assistant="<|assistant|>")
        if cfg.in_family("inputtemplateglm4"):
            return cls(system="<|system|>", user="<|user|>", assistant="<|assistant|>",
                       eos="<|endoftext|>")
        if cfg.in_family("inputtemplateinternlm"):
            return cls(system="<|im_start|>system", user="<|im_start|>user",
                       assistant="<|im_start|>assistant", eos="<|im_end|>")
        return cls()


# ---------------------------------------------------------------------------
# Model loading & generation
# ---------------------------------------------------------------------------


class ModelManager:
    """Loads model + tokenizer and exposes a generate() method."""

    GENERATION_DEFAULTS = dict(
        do_sample=False, max_new_tokens=200,
        top_k=0, top_p=1.0, temperature=0.7,
    )

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device: Optional[torch.device] = None

    def load(self) -> None:
        cfg = self.cfg
        logging.info("LOADING MODEL: %s", cfg.model_name)
        trust = cfg.in_family("trustedremotecode")

        kwargs = dict(
            load_in_4bit=False, load_in_8bit=False,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=trust,
        )

        if cfg.in_family("automodel"):
            self.model = AutoModel.from_pretrained(cfg.model_name, **kwargs)
        elif cfg.in_family("automodelforcasuallm"):
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        elif cfg.in_family("llamaforcasuallm"):
            self.model = LlamaForCausalLM.from_pretrained(cfg.model_name, **kwargs)
        elif cfg.in_family("automodelforseq2seqlm"):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, **kwargs)
        elif cfg.in_family("t5forconditionalgeneration"):
            kwargs["load_in_8bit"] = True
            self.model = T5ForConditionalGeneration.from_pretrained(cfg.model_name, **kwargs)
        else:
            raise ValueError(f"Model type not configured for '{cfg.model_short_name}'")

        self.device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        tok_cls = LlamaTokenizer if cfg.in_family("llamaforcasuallm") else AutoTokenizer
        self.tokenizer = tok_cls.from_pretrained(
            cfg.model_name, use_fast=True, trust_remote_code=trust,
        )

    def generate(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        gen_config = GenerationConfig(**self.GENERATION_DEFAULTS)
        output = self.model.generate(input_ids=input_ids, generation_config=gen_config)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Prompt building (one method per chat-template family)
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Builds family-specific few-shot prompts.

    Every builder method takes:
        dialogue:    list of N dialogue segments (the turns to ask about).
        completions: list of N-1 prior completions for turns 0..N-2.
    """

    def __init__(self, cfg: Config, tokens: PromptTokens):
        self.cfg = cfg
        self.t = tokens

    def build(self, dialogue: List[str], completions: List[str]) -> str:
        for family, builder in self._dispatch():
            if self.cfg.in_family(family):
                return builder(dialogue, completions)
        return self._default(dialogue, completions)

    def _dispatch(self):
        return (
            ("inputtemplatellama3", self._llama3),
            ("inputtemplategemma", self._gemma_phi3),
            ("inputtemplatephi3", self._gemma_phi3),
            ("inputtemplatemistral", self._mistral),
            ("inputtemplatechatglm3", self._chatglm_glm4),
            ("inputtemplateglm4", self._chatglm_glm4),
            ("inputtemplatellama2", self._llama2),
            ("inputtemplateinternlm", self._internlm),
            ("inputtemplate3", self._template3),
            ("inputtemplate1", self._template_1or2),
            ("inputtemplate2", self._template_1or2),
        )

    # -- Llama-3 chat (instruct & base): role-tagged blocks ------------------

    def _llama3(self, dialogue, completions):
        t = self.t
        n = len(dialogue)
        out = [f"{t.bos}{t.user}system{t.assistant}\n        \n{SYSTEM_INTRO}{t.eos}"]
        for q, a in EXAMPLES:
            out.append(
                f"{t.user}user{t.assistant}\n\nQ: {q} {OPTIONS}{t.eos}"
                f"{t.user}assistant{t.assistant}\n\nA: {a}{t.eos}"
            )
        if n == 1:
            out.append(
                f"{t.user}system{t.assistant}\n\n{SYSTEM_TRIGGER}{t.eos}"
                f"{t.user}user{t.assistant}\n\n"
            )
        else:
            out.append(
                f"\n\n{t.bos}{t.user}system{t.assistant}\n\n{SYSTEM_TRIGGER}{t.eos}"
                f"{t.user}user{t.assistant}\n\n"
            )
        for i, dia in enumerate(dialogue):
            sep = " " if i == 0 else ""
            out.append(f"Q:{sep}{dia}{OPTIONS}{t.eos}{t.user}assistant{t.assistant}\n\n")
            if i < len(completions):
                out.append(f"A: {completions[i]}{t.eos}{t.user}user{t.assistant}\n\n")
            else:
                out.append("A:")
        return "".join(out)

    # -- Gemma / Phi-3: <user>/<assistant> turns -----------------------------

    def _gemma_phi3(self, dialogue, completions):
        t = self.t
        n = len(dialogue)
        out = [f"{t.user}\n{SYSTEM_INTRO}\n"]
        for i, (q, a) in enumerate(EXAMPLES):
            if i > 0:
                out.append(f"{t.user}\n")
            out.append(f"Q: {q} {OPTIONS}{t.eos}\n{t.assistant}\nA: {a}{t.eos}\n")
        out.append(f"{t.user}\n{SYSTEM_TRIGGER}\n")
        for i, dia in enumerate(dialogue):
            q_text = dia if i == 0 else dia.strip()
            if i > 0:
                out.append(f"{t.user}\n")
            out.append(f"Q: {q_text}{OPTIONS}{t.eos}\n{t.assistant}\n")
            if i < len(completions):
                out.append(f"A: {completions[i]}{t.eos}\n")
            else:
                out.append("A: " if n == 1 else "A:")
        return "".join(out)

    # -- Mistral instruct ----------------------------------------------------

    def _mistral(self, dialogue, completions):
        t = self.t
        n = len(dialogue)
        out = [f"{t.bos}{t.user}{SYSTEM_INTRO}\n"]
        for i, (q, a) in enumerate(EXAMPLES):
            out.append(f"Q: {q} {OPTIONS}\n")
            if i == len(EXAMPLES) - 1:
                out.append(f"{t.assistant}\n")
            out.append(f"A: {a}\n")
        out.append(f"{t.eos}\n{t.user}\n{SYSTEM_TRIGGER}\n")
        for i, dia in enumerate(dialogue):
            out.append(f"Q: {dia.strip()}{OPTIONS}\n")
            if i == n - 1:
                out.append(f"{t.assistant}\n")
            if i < len(completions):
                out.append(f"A: {completions[i]}\n")
            else:
                out.append("A: ")
        return "".join(out)

    # -- ChatGLM3 / GLM-4: identical template strings ------------------------

    def _chatglm_glm4(self, dialogue, completions):
        t = self.t
        n = len(dialogue)
        out = [f"{t.system}\n{SYSTEM_INTRO}\n"]
        for q, a in EXAMPLES:
            out.append(f"{t.user}\nQ: {q} {OPTIONS}\n{t.assistant}\nA: {a}\n")
        out.append(f"{t.system}\n{SYSTEM_TRIGGER}\n")
        for i, dia in enumerate(dialogue):
            q_text = dia.strip() if n > 1 else dia
            out.append(f"{t.user}\nQ: {q_text}{OPTIONS}\n{t.assistant}")
            if i < len(completions):
                out.append(f"\nA: {completions[i]}\n")
        return "".join(out)

    # -- Llama-2 chat: <<SYS>>/[INST] blocks ---------------------------------

    def _llama2(self, dialogue, completions):
        t = self.t
        out = [f"{t.bos}{t.user} {t.system_start}\n{SYSTEM_INTRO}\n{t.system_end}\n\n"]
        for i, (q, a) in enumerate(EXAMPLES):
            prefix = "" if i == 0 else f"{t.bos}{t.user} "
            out.append(f"{prefix}Q: {q} {OPTIONS} {t.assistant} A: {a} {t.eos}\n")
        out.append(f"\n{t.bos}{t.user} {t.system_start}\n{SYSTEM_TRIGGER}\n{t.system_end}\n\n")
        for i, dia in enumerate(dialogue):
            prefix = "" if i == 0 else f"{t.bos}{t.user} "
            out.append(f"{prefix}Q: {dia}{OPTIONS} {t.assistant} A: ")
            if i < len(completions):
                out.append(f"{completions[i]} {t.eos}\n")
        return "".join(out)

    # -- InternLM (im_start/im_end) ------------------------------------------

    def _internlm(self, dialogue, completions):
        return self._system_block(dialogue, completions, eos_after_trigger=False)

    # -- Generic <|system|>/<|user|>/<|assistant|> with </s> -----------------

    def _template3(self, dialogue, completions):
        return self._system_block(dialogue, completions, eos_after_trigger=True)

    def _system_block(self, dialogue, completions, *, eos_after_trigger: bool):
        t = self.t
        out = [f"{t.system}\n{SYSTEM_INTRO}{t.eos}\n"]
        for q, a in EXAMPLES:
            out.append(f"{t.user}\nQ: {q} {OPTIONS}{t.eos}\n{t.assistant}\nA: {a}{t.eos}\n")
        trigger_eos = t.eos if eos_after_trigger else ""
        out.append(f"\n{t.system}\n{SYSTEM_TRIGGER}{trigger_eos}\n")
        for i, dia in enumerate(dialogue):
            if i == 0:
                q_text, sep = dia, ""
            else:
                q_text, sep = dia.strip(), " "
            out.append(f"{t.user}\nQ: {q_text}{sep}{OPTIONS}{t.eos}\n{t.assistant}\n")
            if i < len(completions):
                out.append(f"A: {completions[i]}{t.eos}\n")
            else:
                out.append("A: ")
        return "".join(out)

    # -- Vicuna / Alpaca / Wizard etc. (USER:/ASSISTANT: or ### Instruction) -

    def _template_1or2(self, dialogue, completions):
        t = self.t
        out = [f"{t.bos}{t.user}{t.system_start}{SYSTEM_INTRO}{t.system_end}\n\n"]
        for i, (q, a) in enumerate(EXAMPLES):
            prefix = "" if i == 0 else f"{t.bos} {t.user} "
            out.append(f"{prefix}Q: {q} {OPTIONS} {t.assistant} A: {a} {t.eos}\n")
        out.append(f"\n{t.bos}{t.user}{t.system_start}{SYSTEM_TRIGGER}{t.system_end}\n\n")
        for i, dia in enumerate(dialogue):
            prefix = "" if i == 0 else f"{t.bos} {t.user} "
            out.append(f"{prefix}Q: {dia}{OPTIONS} {t.assistant} A: ")
            if i < len(completions):
                out.append(f"{completions[i]} {t.eos}\n")
        return "".join(out)

    # -- Plain-text fallback -------------------------------------------------

    def _default(self, dialogue, completions):
        out = [f"{SYSTEM_INTRO}\n"]
        for q, a in EXAMPLES:
            out.append(f"Q: {q} {OPTIONS}\nA: {a}\n")
        out.append(f"\n{SYSTEM_TRIGGER}\n")
        for i, dia in enumerate(dialogue):
            q_text = dia if i == 0 else dia.strip()
            out.append(f"Q: {q_text}{OPTIONS}\n")
            if i < len(completions):
                out.append(f"A: {completions[i]}\n")
            else:
                out.append("A: ")
        return "".join(out)


# ---------------------------------------------------------------------------
# Completion post-processing
# ---------------------------------------------------------------------------


class CompletionPostProcessor:
    """Strips the prompt echo / formatting noise from raw model outputs."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def process(self, completion: str, prompt: str) -> str:
        c = self.cfg
        if c.in_family("repeatedprompts"):
            return completion[completion.rfind("A:") + 2:]
        if c.in_family("repeatedpromptsllama3"):
            return self._llama3(completion, prompt)
        if c.in_family("repeatedpromptsgemma"):
            return self._gemma(completion)
        if c.in_family("repeatedpromptsmistral"):
            return self._mistral(completion, prompt)
        if c.in_family("repeatedpromptschatglm3") or c.in_family("repeatedpromptsglm4"):
            return self._chatglm_glm4(completion, prompt)
        if c.in_family("repeatedpromptsbloomz"):
            return self._bloomz(completion)
        if c.in_family("repeatedpromptsphi3"):
            return self._phi3(completion, prompt)
        if c.in_family("repeatedpromptsairboros"):
            return self._airboros(completion)
        if c.in_family("repeatedpromptszephyralpha"):
            return self._zephyr_alpha(completion, prompt)
        if c.in_family("repeatedpromptszephyrbeta"):
            return self._zephyr_beta(completion, prompt)
        if c.in_family("repeatedpromptszephyrgemma"):
            return self._zephyr_gemma(completion, prompt)
        if c.in_family("repeatedpromptsinternlm"):
            return self._internlm(completion, prompt)
        return completion

    @staticmethod
    def _truncate_at_option(text: str) -> str:
        for opt in EMOTION_OPTIONS:
            pos = text.find(opt)
            if pos != -1:
                return text[:pos + len(opt)] + "."
        return text

    @classmethod
    def _llama3(cls, completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-3][:-55].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").lower():
                comp_lines = comp_lines[i:]
                break
        processed = ""
        for s in comp_lines:
            if "A: " in s:
                processed = s
                break
        processed = (
            processed
            .replace(".assistant", "")
            .replace(".user", "")
            .replace("A:", "")
            + "."
        )
        return cls._truncate_at_option(processed)

    @classmethod
    def _gemma(cls, completion):
        pos = completion.rfind("A:")
        return cls._truncate_at_option(completion[pos + 2:])

    @staticmethod
    def _mistral(completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-3].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        processed = ""
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").lower():
                processed = comp_lines[i + 2]
                break
        processed = processed.replace("A:", "").strip() + "."
        return processed.replace("..", ".")

    @staticmethod
    def _chatglm_glm4(completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-2].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        processed = ""
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").lower():
                try:
                    processed = comp_lines[i + 2]
                    break
                except IndexError:
                    for s2 in comp_lines[i:]:
                        if "A: " in s2:
                            processed = s2
                            break
                    break
        return processed.replace("A:", "").strip()

    @staticmethod
    def _bloomz(completion):
        pos = completion.rfind("A:")
        char = completion[pos + 4:pos + 5]
        for opt in EMOTION_OPTIONS:
            if opt[1:2] == char:
                return opt
        return char

    @staticmethod
    def _phi3(completion, prompt):
        marker = prompt.splitlines()[-3][:-7]
        pos = completion.find(marker)
        sub = completion[pos + len(marker):]
        end = sub.find("Q:")
        return (sub[:end] + ".").replace("A:", "")

    @staticmethod
    def _airboros(completion):
        pos = completion.rfind("A:")
        sub = completion[pos:]
        processed = ""
        for opt in EMOTION_OPTIONS:
            end = sub.find(opt)
            if end != -1:
                processed = sub[:end + len(opt)] + "."
                break
        return processed.replace("A:", "")

    @staticmethod
    def _zephyr_alpha(completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-3][:-4].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").lower():
                comp_lines = comp_lines[i + 1:]
                break
        joined = " ".join(comp_lines)
        end = joined.find("<|user|>")
        processed = joined[:end]  # original behaviour: end == -1 trims last char
        processed = (
            processed
            .replace("<|assistant|>", "")
            .replace("A:", "")
            .strip()
            + "."
        )
        return processed.replace("..", ".")

    @staticmethod
    def _zephyr_beta(completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-3][:-4].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").lower():
                comp_lines = comp_lines[i:]
                break
        processed = ""
        for s in comp_lines:
            if "A: " in s:
                processed = s
                break
        processed = processed.replace("A:", "").strip() + "."
        return processed.replace("..", ".")

    @staticmethod
    def _zephyr_gemma(completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-3][:-4].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").replace("</s>", "").lower():
                comp_lines = comp_lines[i + 2:]
                break
        processed = ""
        for s in comp_lines:
            if "the answer is" in s.lower():
                processed = s
                break
        return processed.replace("A:", "").replace("</s>", "").strip()

    @staticmethod
    def _internlm(completion, prompt):
        prompt_lines = prompt.splitlines()
        question = prompt_lines[-3][:-10].replace(" ", "").lower()
        comp_lines = completion.splitlines()
        for i, s in enumerate(comp_lines):
            if question in s.replace(" ", "").lower():
                comp_lines = comp_lines[i:]
                break
        processed = ""
        for s in comp_lines:
            if "A: " in s:
                processed = s
                break
        processed = (
            processed
            .replace("A:", "")
            .replace("<|im_end|>", "")
            .strip()
            + "."
        )
        return processed.replace("...", ".").replace("..", ".")


# ---------------------------------------------------------------------------
# Dataset & dialogue loading
# ---------------------------------------------------------------------------


def download_dailydialog(dataset_name: str, dataset_dir: Path) -> pd.DataFrame:
    """Download DailyDialog and persist a flat one-row-per-turn CSV."""
    daily = load_dataset("daily_dialog")

    rows: List[Tuple[str, str, str, str, int]] = []
    dial_count = 0
    for split in ("train", "validation", "test"):
        for dial, emotions in zip(daily[split]["dialog"], daily[split]["emotion"]):
            for turn_idx, (text, emo) in enumerate(zip(dial, emotions)):
                uid = f"DAILYD-{dial_count:06d}-{turn_idx:04d}"
                did = uid[:-5]
                sid = "A" if turn_idx % 2 == 0 else "B"
                rows.append((uid, did, sid, text, emo))
            dial_count += 1

    df = pd.DataFrame(rows, columns=["UID", "DID", "SID", "SEG", "EMOTION"])
    df["EMOTION"] = df["EMOTION"].map(EMOTION_LABEL_MAP)
    df.to_csv(dataset_dir / f"{dataset_name}_main.csv", index=False)
    return df


def load_dialogues(cfg: Config) -> pd.DataFrame:
    """Return only dialogues with exactly DIALOGUE_TURNS turns, building cache as needed."""
    dialogues_path = cfg.data_dir / f"{cfg.dataset}_dialogues.csv"
    if dialogues_path.exists():
        logging.info("DIALOGUES FOUND")
        return pd.read_csv(dialogues_path)

    main_path = cfg.data_dir / f"{cfg.dataset}_main.csv"
    if main_path.exists():
        logging.info("PROCESSED DAILYDIALOG FOUND")
        df_main = pd.read_csv(main_path)
    else:
        df_main = download_dailydialog(cfg.dataset, cfg.data_dir)

    counts = df_main["DID"].value_counts()
    keep_dids = counts[counts == DIALOGUE_TURNS].index
    df = df_main[df_main["DID"].isin(keep_dids)].reset_index(drop=True)
    df.to_csv(dialogues_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Emotion extraction & metrics
# ---------------------------------------------------------------------------


_PUNCT_RE = re.compile(r"['.\[\]!,*)@#%(&$_?.^]")


def extract_emotion(completion: str) -> str:
    """Map a model completion string to one of EMOTIONS or 'none'."""
    for opt in EMOTION_OPTIONS:
        if opt in completion:
            return opt.split()[-1].lower()
    normalized = _PUNCT_RE.sub(" ", completion).lower()
    for emo in EMOTIONS:
        if emo in normalized:
            return emo
    return "none"


def report_metrics(y_true, y_pred, label: str) -> None:
    matches = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    total = len(y_true)
    acc = matches / total if total else 0.0
    logging.info(
        "\n%s\nhits: %d\ntotal: %d\nAccuracy: %.2f %%\n"
        "F1: %.4f\nPrecision: %.4f\nRecall: %.4f\n",
        label, matches, total, acc * 100,
        f1_score(y_true, y_pred, average="macro"),
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro"),
    )


def evaluate_predictions(df: pd.DataFrame) -> None:
    """Per-turn and overall accuracy / F1 / precision / recall."""
    per_turn_true: List[list] = [[] for _ in range(DIALOGUE_TURNS)]
    per_turn_pred: List[list] = [[] for _ in range(DIALOGUE_TURNS)]

    for _, sub in df.groupby("DID", sort=False):
        sub = sub.reset_index(drop=True)
        for t in range(DIALOGUE_TURNS):
            per_turn_true[t].append(sub["EMOTION"][t])
            per_turn_pred[t].append(sub["EMOTION_COMPLETION"][t])

    for t in range(DIALOGUE_TURNS):
        report_metrics(per_turn_true[t], per_turn_pred[t], f"#turns: {t}")
    report_metrics(
        df["EMOTION"].tolist(),
        df["EMOTION_COMPLETION"].tolist(),
        "All dataset",
    )

    nones = (df["EMOTION_COMPLETION"] == "none").sum()
    logging.info("#nones: %d", nones)
    counts = df["EMOTION_COMPLETION"].value_counts().sort_index()
    logging.info("count_values:\n%s", counts.to_string())


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


class Phase1Pipeline:
    """End-to-end orchestration: prompts → completions → metrics."""

    DEBUG_DIALOGS = 5

    def __init__(
        self,
        cfg: Config,
        model: ModelManager,
        builder: PromptBuilder,
        post: CompletionPostProcessor,
    ):
        self.cfg = cfg
        self.model = model
        self.builder = builder
        self.post = post

    def run(self) -> pd.DataFrame:
        logging.info("LOADING DIALOGUES")
        df_dialogues = load_dialogues(self.cfg)
        if self.cfg.debug:
            df_dialogues = df_dialogues.head(DIALOGUE_TURNS * self.DEBUG_DIALOGS)

        completions_per_turn: List[pd.DataFrame] = []
        for turn in range(1, DIALOGUE_TURNS + 1):
            df = self._run_turn(df_dialogues, completions_per_turn, turn)
            completions_per_turn.append(df)

        logging.info("PROCESSING EMOTIONS FROM COMPLETIONS")
        df_results = self._merge_and_extract(df_dialogues, completions_per_turn)

        logging.info("CHECKING ACCURACY")
        evaluate_predictions(df_results)
        return df_results

    def _run_turn(
        self,
        df_dialogues: pd.DataFrame,
        prior: List[pd.DataFrame],
        turn: int,
    ) -> pd.DataFrame:
        logging.info("CREATING PROMPTS %d", turn)
        unique_dids = df_dialogues["DID"].unique()
        prompts: List[str] = []
        for idx, did in enumerate(tqdm(unique_dids)):
            dialogue = df_dialogues.loc[df_dialogues["DID"] == did, "SEG"].tolist()
            previous = [prior[k]["COMPLETION"][idx] for k in range(turn - 1)]
            prompts.append(self.builder.build(dialogue[:turn], previous))
        df_prompts = pd.DataFrame(prompts, columns=["PROMPT"])

        logging.info("GENERATING COMPLETIONS %d", turn)
        return self._generate_for(df_prompts, turn)

    def _generate_for(self, df_prompts: pd.DataFrame, turn: int) -> pd.DataFrame:
        outputs: List[str] = []
        first = True
        for prompt in tqdm(df_prompts["PROMPT"].tolist()):
            raw = self.model.generate(prompt)
            processed = self.post.process(raw, prompt)
            processed = "".join(processed.splitlines()).strip()
            outputs.append(processed)
            if first:
                logging.info("\n\nFIRST PROMPT:\n%s", prompt)
                logging.info("\n\nCOMPLETION:\n%s", raw)
                logging.info("\n\nPROCESSED COMPLETION:\n%s", processed)
                first = self.cfg.debug

        df = df_prompts.copy()
        df["COMPLETION"] = outputs

        out_path = (
            self.cfg.model_dir
            / f"{self.cfg.dataset}_{self.cfg.model_short_name}_completions_{turn}.csv"
        )
        logging.info("SAVING COMPLETIONS %d", turn)
        df.to_csv(out_path, index=False)
        return df

    def _merge_and_extract(
        self,
        df_dialogues: pd.DataFrame,
        completions_per_turn: List[pd.DataFrame],
    ) -> pd.DataFrame:
        prompts = [d["PROMPT"] for d in completions_per_turn]
        comps = [d["COMPLETION"] for d in completions_per_turn]
        n_dialogues = len(prompts[0])

        interleaved_prompts = [
            prompts[t][i] for i in range(n_dialogues) for t in range(DIALOGUE_TURNS)
        ]
        interleaved_completions = [
            comps[t][i] for i in range(n_dialogues) for t in range(DIALOGUE_TURNS)
        ]

        df = df_dialogues.copy().reset_index(drop=True)
        df.insert(4, "PROMPT", interleaved_prompts)
        df.insert(5, "COMPLETION", interleaved_completions)
        df["EMOTION_COMPLETION"] = df["COMPLETION"].map(extract_emotion)

        out_path = (
            self.cfg.model_dir
            / f"{self.cfg.dataset}_{self.cfg.model_short_name}_emotion_completion_1.csv"
        )
        df.to_csv(out_path, index=False)
        return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path: str = "config.json") -> None:
    cfg = Config.load(config_path)
    cfg.ensure_directories()

    model = ModelManager(cfg)
    model.load()

    tokens = PromptTokens.for_config(cfg)
    builder = PromptBuilder(cfg, tokens)
    post = CompletionPostProcessor(cfg)

    Phase1Pipeline(cfg, model, builder, post).run()


if __name__ == "__main__":
    main()
