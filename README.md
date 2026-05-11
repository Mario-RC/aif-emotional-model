# RLAIF for Emotional Dialogue Alignment

This repository contains the research workflow used to build, analyze and improve emotionally-aware conversational agents with reinforcement learning from AI feedback (RLAIF). The project combines dialogue generation, supervised fine-tuning, reward modeling, preference data creation and human evaluation around multi-turn conversations where both the user and the chatbot express controlled emotions.

The codebase is organized as an experimental workspace rather than as a single Python package. It mixes custom data-generation and evaluation scripts with a single shared copy of [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) at [llama-factory/](./llama-factory), installed as an editable package. The per-stage `llama-factory-*` folders are thin **workspaces**: they keep only project-specific `data/`, `examples/`, `saves/`, `logs/`, shell launchers and small helper scripts, while all launchers use the same shared `llamafactory-cli`.

## What the project does

- Builds emotionally-structured dialogue datasets.
- Evaluates base foundation models on dialogue quality and emotional control.
- Generates prompt, comparison and preference data with external LLM APIs.
- Trains supervised, reward and RL-aligned dialogue models.
- Compares model outputs with automatic metrics and human annotations.

## Repository structure

```text
rlaif/
├── llama-factory/
├── phase1-foundation-eval/
│   └── 1-foundation_eval.py
├── phase2-sft-alignment/
│   └── sft-model/
│       ├── sft-demonstration-dataset/
│       ├── sft-llama-factory-training/
│       └── sft-eval/
│           └── human-eval/
└── phase3-rlaif-alignment/
    ├── reward-model/
    │   ├── rm-prompt-dataset/
    │   ├── rm-comparison-dataset/
    │   │   ├── llama-factory-predict/
    │   │   └── candidate-curation/
    │   ├── rm-preference-dataset/
    │   ├── rm-llama-factory-training/
    │   └── rm-eval/
    │       └── human-eval/
    └── rlaif-model/
        ├── ppo-unlabeled-prompts-dataset/
        ├── dpo-comparison-dataset/
        │   ├── llama-factory-predict/
        │   └── candidate-curation/
        ├── dpo-preference-dataset/
        ├── rlaif-llama-factory-training/
        └── rlaif-eval/
            └── human-eval/
```

### LLaMA-Factory: canonical install + thin workspaces

The repository keeps a single shared copy of the LLaMA-Factory framework at [llama-factory/](./llama-factory). It is installed once into the project venv as an editable package:

```bash
source ./.vrlaif/bin/activate
pip install -e ./llama-factory
```

After this, the `llamafactory-cli` binary is on the venv's `PATH` and resolves `import llamafactory` to the canonical source tree. Every per-stage folder named `llama-factory-*/` (`sft-llama-factory-training`, `rm-llama-factory-training`, `llama-factory-predict`, `rlaif-llama-factory-training`) is a thin **workspace** that contains only project content:

- `data/` — workspace-specific `dataset_info.json` plus the project datasets used in that stage. Upstream LLaMA-Factory default datasets (`alpaca_*`, `belle_multiturn`, `c4_demo`, `dpo_*_demo`, `glaive_toolcall_*`, `hh_rlhf_en`, `identity`, `kto_*`, `mllm_*`, `ultra_chat`, `wiki_demo`) are **not duplicated**; they live only at [llama-factory/data/](./llama-factory/data).
- `examples/` — workspace-specific YAML configs for training / inference, all under [`train_lora/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/train_lora). The other LF default subfolders (`accelerate/`, `deepspeed/`, `extras/`, `merge_lora/`, `train_full/`, `train_qlora/`) live only at [llama-factory/examples/](./llama-factory/examples).
- `saves/` — checkpoints and predictions for that stage's experiments.
- `logs/` — log output.
- `*.sh` — launchers that call `llamafactory-cli` on relative paths inside the workspace.
- `emotional_results.py` (some workspaces) — local helper for emotional-control metrics.

All workspaces use the same folder naming under `examples/`: `train_lora/{gemma2,glm4,llama3,mistral,phi3}/...yaml` (the older `lora_single_gpu/`, `qlora_single_gpu/`, `gemma-2/`, `Llama-3/`, `Mistral/` etc. names from earlier LLaMA-Factory snapshots have been normalized so launchers work uniformly across all phases).

Launchers run with the workspace as cwd, e.g.:

```bash
cd phase3-rlaif-alignment/reward-model/rm-llama-factory-training
bash <launcher>.sh   # internally: llamafactory-cli train examples/.../*.yaml
```

This keeps every experiment self-contained on the data/config side while the framework code lives in exactly one place.

## Pipeline overview

### Phase 1: Foundation model evaluation

The first phase evaluates open foundation models on the target emotional dialogue task. The main entrypoint is [phase1-foundation-eval/1-foundation_eval.py](./phase1-foundation-eval/1-foundation_eval.py), which:

- Downloads and preprocesses the `daily_dialog` dataset.
- Filters dialogues to the structure required by the project.
- Loads Hugging Face models with `transformers` and `torch`.
- Generates and stores model outputs for later analysis.

This phase is mainly useful for building a baseline before fine-tuning and RL alignment.

### Phase 2: Supervised fine-tuning and human evaluation

`phase2-sft-alignment/` groups the work that turns the foundation-model baseline from Phase 1 into a supervised, emotionally-aware dialogue model, and collects the human judgments used to validate it. Everything lives under [sft-model/](./phase2-sft-alignment/sft-model): the demonstration-dataset generator, the LLaMA-Factory training workspace, and the human-evaluation folder under `sft-eval/`.

#### [phase2-sft-alignment/sft-model/sft-eval/human-eval](./phase2-sft-alignment/sft-model/sft-eval/human-eval)

Contains the full human-eval protocol used to rate SFT model outputs on the emotional-dialogue task. Each dialogue has a `PROMPT` (user turn with a controlled emotion) and a three-sentence `RESPONSE` from the chatbot (empathetic sentence, emotion-bearing sentence, and a follow-up question). Annotators judge these responses across four tasks:

- **Task 1 — Empathy:** is the chatbot's response empathetic with respect to the user prompt (binary Yes/No).
- **Task 2 — Emotion labeling:** identify the emotion expressed by the user and by the chatbot among 7 categories (Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral).
- **Task 3 — Follow-up question quality:** is the chatbot's follow-up question adequate given the previous user turn (binary Yes/No).
- **Task 4 — Overall quality:** 1–5 rating of the whole chatbot response given the user turn.

The split between inter-annotator-agreement items (about 30%, shared across annotators) and variability items (about 70%, disjoint per annotator) is described in [human_annotations.txt](./phase2-sft-alignment/sft-model/sft-eval/human-eval/human_annotations.txt). Generated annotation spreadsheets live under `data/`; the anonymized filled-in annotator responses live under [results/](./phase2-sft-alignment/sft-model/sft-eval/human-eval/results). The current analysis supports five annotators (`anno1`...`anno5`); `anno5` only contributes non-empty annotations for Task 1, Task 3 and Task 4, so empty Task 2 answers are ignored by the metrics.

The pipeline has been refactored into a reusable package under [src/](./phase2-sft-alignment/sft-model/sft-eval/human-eval/src) (modules for configuration, dialogue parsing, task building, annotator sampling, metrics and plots) with two numbered entrypoints that reflect the execution order:

- [1-generate_annotations.py](./phase2-sft-alignment/sft-model/sft-eval/human-eval/1-generate_annotations.py) — builds the annotation spreadsheets from SFT model predictions, producing `data/sft_demonstration_dataset_test_predictions.csv`, the auxiliary files, task files and annotator-facing Excel files under `data/`.
- [2-analyze_results.py](./phase2-sft-alignment/sft-model/sft-eval/human-eval/2-analyze_results.py) — aggregates annotator responses: Rank@K / mean-std / Task-2 hit tables, Krippendorff alpha, and the histograms saved under `hist/`.

#### [phase2-sft-alignment/sft-model/](./phase2-sft-alignment/sft-model)

Contains the supervised fine-tuning pipeline: demonstration-data generation and the LLaMA-Factory workspace used to train and evaluate the SFT models.

- [phase2-sft-alignment/sft-model/sft-demonstration-dataset/](./phase2-sft-alignment/sft-model/sft-demonstration-dataset) — generates the SFT dataset of structured emotional dialogues and prepares it for `LLaMA-Factory`. The three numbered scripts run in order:
  - [1-generate_sft_demonstration_dataset.py](./phase2-sft-alignment/sft-model/sft-demonstration-dataset/1-generate_sft_demonstration_dataset.py): generates multi-turn emotional dialogues via Azure OpenAI (ChatGPT) over a fixed set of topics and emotion combinations, with periodic CSV checkpoints.
  - [2-formatted_sft_demonstration_dataset.py](./phase2-sft-alignment/sft-model/sft-demonstration-dataset/2-formatted_sft_demonstration_dataset.py): parses the raw generation CSV into structured per-speaker / per-turn CSVs (`sft_demonstration_dataset_formatted.csv`, `sft_demonstration_dataset_turns.csv`).
  - [3-train_test_sft_demonstration_dataset_json.py](./phase2-sft-alignment/sft-model/sft-demonstration-dataset/3-train_test_sft_demonstration_dataset_json.py): builds the `LLaMA-Factory`-compatible train/test JSON with an emotion-balanced test split via stratified sampling over `(EMOTION_RESPONSE_1, EMOTION_RESPONSE_2)` pairs.
- [phase2-sft-alignment/sft-model/sft-llama-factory-training/](./phase2-sft-alignment/sft-model/sft-llama-factory-training) — LLaMA-Factory workspace used for SFT training and evaluation runs, with launchers such as [sft_train.sh](./phase2-sft-alignment/sft-model/sft-llama-factory-training/sft_train.sh).

### Phase 3: Reward modeling and RLAIF

`phase3-rlaif-alignment/` implements the full AI-feedback pipeline on top of the SFT model from Phase 2: generate candidate responses, collect preferences from LLM judges, train a reward model, and then use it (or the judges directly) to produce DPO-style preference data for RL alignment. It is split into two sibling trees: [phase3-rlaif-alignment/reward-model](./phase3-rlaif-alignment/reward-model) and [phase3-rlaif-alignment/rlaif-model](./phase3-rlaif-alignment/rlaif-model).

#### [phase3-rlaif-alignment/reward-model/](./phase3-rlaif-alignment/reward-model)

Produces the preference dataset used to train the reward model, trains it, and evaluates it.

- [rm-prompt-dataset/](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset) — builds the prompt pool that will later be fed to candidate models. The four numbered scripts run in order:
  - [1-generate_rm_prompt_dataset.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/1-generate_rm_prompt_dataset.py): generates emotion-controlled user prompts via Azure OpenAI (GPT-4 / GPT-4-Turbo), configured through [config.json](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/config.json).
  - [2-formatted_rm_prompt_dataset.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/2-formatted_rm_prompt_dataset.py) and [3-train_test_rm_prompt_dataset_json.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/3-train_test_rm_prompt_dataset_json.py): normalize and split the prompt pool into train/test JSON.
  - [4-merge_demonstration_prompt_datasets.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/4-merge_demonstration_prompt_datasets.py): merges these new prompts with the Phase 2 demonstration data so the same pool can be reused by the candidate-prediction step.
  - Shared utilities live in [_lib.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/_lib.py) (topics, emotion catalogue, dialogue template, JSON helpers).
- [rm-comparison-dataset/](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset) — runs multiple SFT candidate models on the prompt pool to produce the responses that will be compared. It wraps a LLaMA-Factory inference workspace plus a numbered candidate-curation pipeline:
  - [llama-factory-predict/](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/llama-factory-predict): inference workspace used to generate candidate responses for each prompt; results are aggregated by [emotional_results.py](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/llama-factory-predict/emotional_results.py).
  - [candidate-curation/](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation): six scripts total, three train scripts plus three `_test` variants. `1-embeddings_semantic_similarity_*` computes embeddings + Distinct-N + scoring, `2-delete_respones_*` joins the per-model predictions and drops the weakest response per row, and `3-add_negative_samples_*` injects negative-quality samples. Common code lives in [_lib.py](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation/_lib.py).
- [rm-preference-dataset/](./phase3-rlaif-alignment/reward-model/rm-preference-dataset) — turns candidate responses into preference pairs by asking LLM judges to rate them. The judges (GPT-4, GPT-4-Turbo, GPT-4O, Claude-3-Opus, Claude-3.5-Sonnet, Gemini-1.0-Pro, Gemini-1.5-Pro, Gemini-1.5-Flash and Llama-3.1-405B) are configured via [config.json](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/config.json). The API-generated rating CSVs needed to run the later steps without secrets are stored under [data/](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/data). All steps are `.py` scripts (with `_test` counterparts) backed by a shared [_lib.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/_lib.py) holding the emotion catalogue, expression-level table, prompt builder and unified `RatingClient`:
  - [1-preprocess_rm_preference_dataset.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/1-preprocess_rm_preference_dataset.py): prepares the response pairs for judging.
  - [2-generate_rating_data.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/2-generate_rating_data.py): calls the LLM judges (OpenAI, Anthropic, Gemini, Llama) to rate each response.
  - [3-rates_to_ranks.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/3-rates_to_ranks.py): converts numeric ratings into pairwise rankings (chosen / rejected).
  - [4-postprocess_rm_preference_dataset.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/4-postprocess_rm_preference_dataset.py) and [5-format_rm_preference_dataset.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/5-format_rm_preference_dataset.py): clean and export the `LLaMA-Factory`-compatible preference dataset.
- [rm-llama-factory-training/](./phase3-rlaif-alignment/reward-model/rm-llama-factory-training) — LLaMA-Factory workspace where the reward model is trained and evaluated. Reward-model LoRA checkpoints are stored under `saves/<model>/lora/rm_*`; prediction metrics for the normal test split and the human-annotation split are stored under `saves/<model>/predict/rm_*` and `saves/<model>/predict/ha_*`. The reward-model training YAMLs point back to the Phase 2 `sft_3ep` adapters by relative path instead of duplicating SFT checkpoints in Phase 3.
- [rm-eval/](./phase3-rlaif-alignment/reward-model/rm-eval) — evaluates reward-model predictions against held-out preference data and human annotations. The current executable pipeline is under [human-eval/](./phase3-rlaif-alignment/reward-model/rm-eval/human-eval):
  - [1-generate_human_eval_sft_predictions_csv.py](./phase3-rlaif-alignment/reward-model/rm-eval/human-eval/1-generate_human_eval_sft_predictions_csv.py): rebuilds `data/rm_human_eval_sft_predictions_test.csv` when the raw SFT prediction JSON/JSONL files for GLM4, Gemma, Llama 3, Mistral and Phi 3 are available.
  - [2-human_annotations_reward_model.py](./phase3-rlaif-alignment/reward-model/rm-eval/human-eval/2-human_annotations_reward_model.py): reads the Phase 2 anonymized annotation XLSX files directly, combines them with `data/rm_human_eval_sft_predictions_test.csv` and the Phase 3 prompt metadata, and writes `results/human_annotations_reward_model_test.json` plus intermediate CSV/JSON files.
  - [3-analyze_reward_model_test_results.py](./phase3-rlaif-alignment/reward-model/rm-eval/human-eval/3-analyze_reward_model_test_results.py): reads `rm-llama-factory-training/saves/<model>/predict/{rm_*,ha_*}/all_results.json` and prints the reward-model and human-annotation evaluation tables.

#### [phase3-rlaif-alignment/rlaif-model/](./phase3-rlaif-alignment/rlaif-model)

Uses the same prompt → candidate → rating → preference flow as `reward-model/`, but targeted at producing DPO training data and the final RL-aligned dialogue model.

- [ppo-unlabeled-prompts-dataset/](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset) — generates an additional pool of emotional prompts dedicated to the RLAIF round (separate from the reward-model prompt pool) and merges it with the Phase 2 demonstration data. Four numbered scripts plus a shared [_lib.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/_lib.py) (1000 topics + emotion catalogue + dialogue template) mirror the `rm-prompt-dataset/` pipeline:
  - [1-generate_ppo_unlabeled_prompts_dataset.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/1-generate_ppo_unlabeled_prompts_dataset.py), [2-formatted_ppo_unlabeled_prompts_dataset.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/2-formatted_ppo_unlabeled_prompts_dataset.py), [3-train_test_ppo_unlabeled_prompts_dataset_json.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/3-train_test_ppo_unlabeled_prompts_dataset_json.py), [4-merge_demonstration_prompt_ppo_unlabeled_prompts_datasets.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/4-merge_demonstration_prompt_ppo_unlabeled_prompts_datasets.py).
- [dpo-comparison-dataset/](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset) — inference + curation of candidate responses for the DPO round. Contains its own [llama-factory-predict/](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset/llama-factory-predict) workspace plus a [candidate-curation/](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset/candidate-curation) pipeline of three numbered scripts (`1-embeddings_*`, `2-delete_respones_*`, `3-add_negative_samples_*` with `_test` variants) backed by [_lib.py](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset/candidate-curation/_lib.py).
- [dpo-preference-dataset/](./phase3-rlaif-alignment/rlaif-model/dpo-preference-dataset) — same 5-step rating pipeline as `rm-preference-dataset/` but consumes the curated DPO candidates and outputs DPO pairs (`1-preprocess_*` → `5-format_*`, all with `_test` variants and shared [_lib.py](./phase3-rlaif-alignment/rlaif-model/dpo-preference-dataset/_lib.py)). The cross-cutting plot step [6-rates_to_ranks_dpr.py](./phase3-rlaif-alignment/rlaif-model/dpo-preference-dataset/6-rates_to_ranks_dpr.py) renders combined train+test plots over reward-model + RLAIF rates.
- [rlaif-llama-factory-training/](./phase3-rlaif-alignment/rlaif-model/rlaif-llama-factory-training) — current LLaMA-Factory workspace used for the DPO / RLAIF training runs on top of the SFT checkpoint.
- [rlaif-eval/](./phase3-rlaif-alignment/rlaif-model/rlaif-eval) — evaluation of the RL-aligned model: [1-analyze_rl_test_results.py](./phase3-rlaif-alignment/rlaif-model/rlaif-eval/1-analyze_rl_test_results.py) aggregates automatic metrics across DPO/PPO/SFT-DPR run families, and [human-eval/](./phase3-rlaif-alignment/rlaif-model/rlaif-eval/human-eval) mirrors the Phase 2 human-eval protocol applied to the final RLAIF outputs ([1-human_annotations_generation.py](./phase3-rlaif-alignment/rlaif-model/rlaif-eval/human-eval/1-human_annotations_generation.py) builds the annotator XLSX files; [2-human_annotations_results.py](./phase3-rlaif-alignment/rlaif-model/rlaif-eval/human-eval/2-human_annotations_results.py) aggregates the filled responses; the annotator-facing guide is in [README.md](./phase3-rlaif-alignment/rlaif-model/rlaif-eval/human-eval/README.md)).

## Main technologies

- Python 3.12
- PyTorch
- Hugging Face `transformers`, `datasets`, `accelerate`, `peft`, `trl` and `sentence-transformers`
- `LLaMA-Factory`
- OpenAI, Anthropic, Gemini and Azure AI inference APIs
- Jupyter, pandas and plotting libraries for analysis

## Environment

Use Python `3.12.x` for the project environment. The development machine may already have a local `./.vrlaif` virtual environment, but it is treated as a local execution artifact rather than a source file.

The dependency manifest in [requirements.txt](./requirements.txt) was prepared to cover:

- The custom scripts under `phase1-foundation-eval` and `phase3-rlaif-alignment`.
- The analysis notebooks and annotation tooling.
- The shared `LLaMA-Factory` workflows used throughout the project.

## Installation

```bash
source ./.vrlaif/bin/activate
pip install -r requirements.txt
pip install -e ./llama-factory
```

The last line installs the canonical LLaMA-Factory copy as an editable package so that every workspace can call `llamafactory-cli` against the same source tree (see [LLaMA-Factory: canonical install + thin workspaces](#llama-factory-canonical-install--thin-workspaces)). `requirements.txt` pins `transformers` to a version compatible with the project LLaMA-Factory workflows.

## GPU and training notes

This repository includes large-model training and inference workflows. In practice, full experiments usually require:

- Access to an NVIDIA GPU.
- A CUDA-compatible PyTorch installation.
- Enough VRAM for the selected base model and training method.

Some packages are environment-specific and may need to be installed separately depending on the target machine:

- `deepspeed`
- `bitsandbytes`
- `vllm`

For that reason, the base installation favors a reproducible default environment and leaves host-specific acceleration choices to the execution server.

## Configuration

Several scripts expect local JSON configuration files. Naming follows a single convention across phases:

- `config.json` — non-secret model registry / template settings (e.g. [phase1-foundation-eval/config.json](./phase1-foundation-eval/config.json)).
- `config.json` — local per-stage configuration. In generation stages it stores Azure OpenAI deployments keyed with normalized uppercase names such as `CHATGPT`, `GPT-4`, `GPT-4-TURBO` and `GPT-4O`; in preference-rating stages it stores the multi-provider LLM judges (Azure OpenAI + Anthropic + Gemini + Llama).

The per-stage `config.json` files share the same nested-by-deployment schema (`{"<DEPLOYMENT>": {"MODEL": ..., "AZURE_OPENAI_ENDPOINT": ..., ...}}`) and uppercase top-level keys so the same naming convention works in every phase.

Recommended practice:

- Keep secrets outside versioned source files whenever possible.
- Use environment variables for real credentials; leave placeholders in `config.json`.
- Supported secret environment variables are `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_INFERENCE_ENDPOINT` / `ENDPOINT` and `AZURE_INFERENCE_CREDENTIAL`.
- Avoid committing API keys, endpoints or other sensitive data into notebooks and scripts.

## Typical workflows

### Evaluate foundation models

```bash
cd phase1-foundation-eval
python 1-foundation_eval.py
```

### Generate RM Prompt Dataset

```bash
cd phase3-rlaif-alignment/reward-model/rm-prompt-dataset
python 1-generate_rm_prompt_dataset.py
```

### Generate RM Preference Dataset

```bash
cd phase3-rlaif-alignment/reward-model/rm-preference-dataset
python 2-generate_rating_data.py
```

### Generate PPO Unlabeled Prompts Dataset

```bash
cd phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset
python 1-generate_ppo_unlabeled_prompts_dataset.py
```

### Run LLaMA-Factory workspace jobs

The repository includes several stage-specific LLaMA-Factory workspaces under `phase2-sft-alignment` and `phase3-rlaif-alignment`. Those are used for:

- Supervised fine-tuning
- Reward-model training
- Prediction/evaluation pipelines
- DPO or RLAIF-style optimization

In practice, these runs are launched from inside the corresponding workspace with project-specific YAML files and shell scripts. Reward-model and RLAIF training YAMLs should reference the Phase 2 `sft_3ep` adapters by relative path instead of copying SFT checkpoints between phases.

### Build RM human-eval data

```bash
cd phase3-rlaif-alignment/reward-model/rm-eval/human-eval
python 2-human_annotations_reward_model.py
python 3-analyze_reward_model_test_results.py
```

`1-generate_human_eval_sft_predictions_csv.py` is only needed when regenerating `data/rm_human_eval_sft_predictions_test.csv` from raw SFT prediction JSON/JSONL files.

## Data and outputs

This repository stores source code plus a small number of experiment artifacts that are needed to reproduce downstream steps without private credentials or missing raw model generations. The general rule is:

- Keep reusable source code, non-secret configs and small canonical inputs.
- Keep anonymized human annotation spreadsheets because they cannot be regenerated.
- Keep API-generated judge rating CSVs that are needed to run later preference steps without API keys.
- Keep `phase3-rlaif-alignment/reward-model/rm-eval/human-eval/data/rm_human_eval_sft_predictions_test.csv`, because it is required by `2-human_annotations_reward_model.py` and cannot be regenerated exactly unless the five raw SFT prediction files are also available.
- Do not version generated `results/`, `logs/`, `__pycache__/`, temporary notebooks, model checkpoints or large `saves/` outputs unless a specific artifact is deliberately needed for reproducibility.

Important generated paths:

- `phase2-sft-alignment/sft-model/sft-eval/human-eval/data/` is generated by `1-generate_annotations.py`.
- `phase3-rlaif-alignment/reward-model/rm-eval/human-eval/results/` is generated by `2-human_annotations_reward_model.py`.
- `phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation/data/` is generated by the candidate-curation scripts.
- `phase3-rlaif-alignment/reward-model/rm-llama-factory-training/saves/` is generated by LLaMA-Factory training and prediction runs.

## Reproducibility recommendations

- Use Python `3.12.x`.
- Install dependencies from `requirements.txt` before reproducing old experiments.
- Record the exact CUDA, driver and PyTorch build used on each machine.
- Keep track of the exact LLaMA-Factory workspace, YAML and launcher used for each run.
- Version control configs carefully, but keep credentials out of the repository.
- Save logs independently for foundation, SFT, reward-model and RL/RLAIF runs.

## Known caveats

- This is not a single packaged application; it is a multi-phase research workspace.
- Some scripts require local credentials, private checkpoints or machine-specific GPU availability.
- Large parts of the project are experiment-driven, so not every folder has the same level of cleanup or standardization.

## License and provenance

This repository contains custom project code together with the shared upstream [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) source under [llama-factory/](./llama-factory). The licensing and attribution of upstream components should be preserved in their respective subdirectories.
