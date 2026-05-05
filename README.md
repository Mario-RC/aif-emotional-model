# RLAIF for Emotional Dialogue Alignment

The codebase is organized as an experimental workspace rather than as a single Python package. It mixes custom data-generation and evaluation scripts with a single shared copy of [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) at [llama-factory/](./llama-factory) (installed as an editable package), and a number of thin per-stage **workspaces** named `llama-factory-*/` that hold only the project-specific content (`data/`, `examples/`, `saves/`, `logs/`, `*.sh` launchers and an optional `emotional_results.py`) — every workspace runs its launchers against the same shared CLI.

## What the project does

- Builds emotionally-structured dialogue datasets.
- Evaluates base foundation models on dialogue quality and emotional control.
- Generates prompt, comparison and preference data with external LLM APIs.
- Trains supervised, reward and RL-aligned dialogue models.

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
└── phase3-rlaif-alignment/
    ├── reward-model/
    │   ├── rm-prompt-dataset/
    │   ├── rm-comparison-dataset/
    │   │   ├── llama-factory-predict/
    │   │   └── candidate-curation/
    │   ├── rm-preference-dataset/
    │   ├── rm-llama-factory-training/
    └── rlaif-model/
        ├── ppo-unlabeled-prompts-dataset/
        ├── dpo-comparison-dataset/
        │   ├── llama-factory-predict/
        │   └── candidate-curation/
        ├── dpo-preference-dataset/
        ├── rlaif-llama-factory-training/
```

### LLaMA-Factory: canonical install + thin workspaces

The repository keeps a single shared copy of the LLaMA-Factory framework at [llama-factory/](./llama-factory). It is installed once into the project venv as an editable package:

```bash
source ./.vrlaif/bin/activate
pip install -e ./llama-factory
```

After this, the `llamafactory-cli` binary is on the venv's `PATH` and resolves `import llamafactory` to the canonical source tree. Every per-stage folder named `llama-factory-*/` (`sft-llama-factory-legacy`, `sft-llama-factory-training`, `rm-llama-factory-training`, `llama-factory-predict`, `rlaif-llama-factory-legacy`, `rlaif-llama-factory-training`) is a thin **workspace** that contains only project content:

- `data/` — workspace-specific `dataset_info.json` plus the project datasets used in that stage. Upstream LLaMA-Factory default datasets (`alpaca_*`, `belle_multiturn`, `c4_demo`, `dpo_*_demo`, `glaive_toolcall_*`, `hh_rlhf_en`, `identity`, `kto_*`, `mllm_*`, `ultra_chat`, `wiki_demo`) are **not duplicated**; they live only at [llama-factory/data/](./llama-factory/data).
- `examples/` — workspace-specific YAML configs for training / inference, all under [`train_lora/`](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples/train_lora) (and `inference/` in `rlaif-llama-factory-legacy`). The other LF default subfolders (`accelerate/`, `deepspeed/`, `extras/`, `merge_lora/`, `train_full/`, `train_qlora/`) live only at [llama-factory/examples/](./llama-factory/examples).
- `saves/` — checkpoints and predictions for that stage's experiments.
- `logs/` — log output.
- `*.sh` — launchers that call `llamafactory-cli` on relative paths inside the workspace.
- `emotional_results.py` (some workspaces) — local helper for emotional-control metrics.

All workspaces use the same folder naming under `examples/`: `train_lora/{gemma2,glm4,llama3,mistral,phi3}/...yaml` (the older `lora_single_gpu/`, `qlora_single_gpu/`, `gemma-2/`, `Llama-3/`, `Mistral/` etc. names from earlier LLaMA-Factory snapshots have been normalized so launchers work uniformly across all phases).

Launchers run with the workspace as cwd, e.g.:

```bash
cd phase3-rlaif-alignment/reward-model/rm-llama-factory-training
bash rm_train_eval.sh   # internally: llamafactory-cli train examples/.../*.yaml
```

This keeps every experiment self-contained on the data side while the framework code lives in exactly one place.

## Pipeline overview

### Phase 1: Foundation model evaluation

The first phase evaluates open foundation models on the target emotional dialogue task. The main entrypoint is [phase1-foundation-eval/1-foundation_eval.py](./phase1-foundation-eval/1-foundation_eval.py), which:

- Downloads and preprocesses the `daily_dialog` dataset.
- Filters dialogues to the structure required by the project.
- Loads Hugging Face models with `transformers` and `torch`.
- Generates and stores model outputs for later analysis.

This phase is mainly useful for building a baseline before fine-tuning and RL alignment.

### Phase 2: Supervised fine-tuning

#### [phase2-sft-alignment/sft-model/](./phase2-sft-alignment/sft-model)

Contains the supervised fine-tuning pipeline: demonstration-data generation and two LLaMA-Factory workspaces used to train and evaluate the SFT models.

- [phase2-sft-alignment/sft-model/sft-demonstration-dataset/](./phase2-sft-alignment/sft-model/sft-demonstration-dataset) — generates the SFT dataset of structured emotional dialogues and prepares it for `LLaMA-Factory`. The three numbered scripts run in order:
  - [1-generate_sft_demonstration_dataset.py](./phase2-sft-alignment/sft-model/sft-demonstration-dataset/1-generate_sft_demonstration_dataset.py): generates multi-turn emotional dialogues via Azure OpenAI (ChatGPT) over a fixed set of topics and emotion combinations, with periodic CSV checkpoints.
  - [2-formatted_sft_demonstration_dataset.py](./phase2-sft-alignment/sft-model/sft-demonstration-dataset/2-formatted_sft_demonstration_dataset.py): parses the raw generation CSV into structured per-speaker / per-turn CSVs (`sft_demonstration_dataset_formatted.csv`, `sft_demonstration_dataset_turns.csv`).
  - [3-train_test_sft_demonstration_dataset_json.py](./phase2-sft-alignment/sft-model/sft-demonstration-dataset/3-train_test_sft_demonstration_dataset_json.py): builds the `LLaMA-Factory`-compatible train/test JSON with an emotion-balanced test split via stratified sampling over `(EMOTION_RESPONSE_1, EMOTION_RESPONSE_2)` pairs.
- [phase2-sft-alignment/sft-model/sft-llama-factory-legacy/](./phase2-sft-alignment/sft-model/sft-llama-factory-legacy) — LLaMA-Factory workspace that consumes the emotion-balanced dataset produced above. It holds project-specific launchers ([sft_train_eval.sh](./phase2-sft-alignment/sft-model/sft-llama-factory-legacy/sft_train_eval.sh) for SFT + evaluation, [foundation_eval.sh](./phase2-sft-alignment/sft-model/sft-llama-factory-legacy/foundation_eval.sh) for foundation-model evaluation under the same pipeline) and an [emotional_results.py](./phase2-sft-alignment/sft-model/sft-llama-factory-legacy/emotional_results.py) helper to summarize emotional-control metrics over the predictions in [saves/](./phase2-sft-alignment/sft-model/sft-llama-factory-legacy/saves).
- [phase2-sft-alignment/sft-model/sft-llama-factory-training/](./phase2-sft-alignment/sft-model/sft-llama-factory-training) — a second LLaMA-Factory workspace used for additional SFT training and evaluation runs (see [sft_train_eval_2.sh](./phase2-sft-alignment/sft-model/sft-llama-factory-training/sft_train_eval_2.sh) and [sft_train_eval_61.sh](./phase2-sft-alignment/sft-model/sft-llama-factory-training/sft_train_eval_61.sh)).

### Phase 3: Reward modeling and RLAIF

`phase3-rlaif-alignment/` implements the full AI-feedback pipeline on top of the SFT model from Phase 2: generate candidate responses, collect preferences from LLM judges, train a reward model, and then use it (or the judges directly) to produce DPO-style preference data for RL alignment. It is split into two sibling trees: [phase3-rlaif-alignment/reward-model](./phase3-rlaif-alignment/reward-model) and [phase3-rlaif-alignment/rlaif-model](./phase3-rlaif-alignment/rlaif-model).

#### [phase3-rlaif-alignment/reward-model/](./phase3-rlaif-alignment/reward-model)

Produces the preference dataset used to train the reward model, trains it, and evaluates it.

- [rm-prompt-dataset/](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset) — builds the prompt pool that will later be fed to candidate models. The four numbered scripts run in order, with the original notebooks preserved under `bck/`:
  - [1-generate_rm_prompt_dataset.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/1-generate_rm_prompt_dataset.py): generates emotion-controlled user prompts via Azure OpenAI (GPT-4-Turbo / GPT-4US), configured through [config_gpt.json](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/config_gpt.json).
  - [2-formatted_rm_prompt_dataset.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/2-formatted_rm_prompt_dataset.py) and [3-train_test_rm_prompt_dataset_json.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/3-train_test_rm_prompt_dataset_json.py): normalize and split the prompt pool into train/test JSON.
  - [4-merge_demonstration_prompt_datasets.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/4-merge_demonstration_prompt_datasets.py): merges these new prompts with the Phase 2 demonstration data so the same pool can be reused by the candidate-prediction step.
  - Shared utilities live in [_lib.py](./phase3-rlaif-alignment/reward-model/rm-prompt-dataset/_lib.py) (topics, emotion catalogue, dialogue template, JSON helpers).
- [rm-comparison-dataset/](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset) — runs multiple candidate models on the prompt pool to produce the responses that will be compared. It wraps a LLaMA-Factory workspace plus a numbered five-step candidate-curation pipeline:
  - [llama-factory-predict/](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/llama-factory-predict): inference workspace used to generate candidate responses for each prompt; results are aggregated by [emotional_results.py](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/llama-factory-predict/emotional_results.py).
  - [candidate-curation/](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation): five numbered scripts (with `_test` variants for the evaluation split): `1-format_chosen_samples_*` and `2-generate_chosen_samples_*` prepare the LLaMA-Factory inference inputs, `3-embeddings_semantic_similarity_*` computes embeddings + Distinct-N + scoring, `4-delete_respones_*` joins the per-model chosen predictions and drops the worst per row, `5-add_negative_samples_*` injects negative-quality samples. Common code lives in [_lib.py](./phase3-rlaif-alignment/reward-model/rm-comparison-dataset/candidate-curation/_lib.py).
- [rm-preference-dataset/](./phase3-rlaif-alignment/reward-model/rm-preference-dataset) — turns candidate responses into preference pairs by asking LLM judges to rate them. The judges (GPT-4, GPT-4-Turbo, GPT-4O, Claude-3-Opus, Claude-3.5-Sonnet, Gemini-1.0-Pro, Gemini-1.5-Pro, Gemini-1.5-Flash and Llama-3.1-405B) are configured via [config_llm.json](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/config_llm.json) and their outputs are stored under [data/](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/data). All steps are now `.py` (with `_test` counterparts) backed by a shared [_lib.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/_lib.py) holding the emotion catalogue, expression-level table, prompt builder and unified `RatingClient`:
  - [1-preprocess_rm_preference_dataset.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/1-preprocess_rm_preference_dataset.py): prepares the response pairs for judging.
  - [2-generate_rating_data.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/2-generate_rating_data.py): calls the LLM judges (OpenAI, Anthropic, Gemini, Llama) to rate each response.
  - [3-rates_to_ranks.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/3-rates_to_ranks.py): converts numeric ratings into pairwise rankings (chosen / rejected).
  - [4-postprocess_rm_preference_dataset.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/4-postprocess_rm_preference_dataset.py) and [5-format_rm_preference_dataset.py](./phase3-rlaif-alignment/reward-model/rm-preference-dataset/5-format_rm_preference_dataset.py): clean and export the `LLaMA-Factory`-compatible preference dataset.
- [rm-llama-factory-training/](./phase3-rlaif-alignment/reward-model/rm-llama-factory-training) — LLaMA-Factory workspace where the reward model is actually trained on the preference dataset produced above. Checkpoints for the different base models (gemma-2-9b-it, glm-4-9b-chat-1m, Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.3, Phi-3-small-8k-instruct) live under [saves/](./phase3-rlaif-alignment/reward-model/rm-llama-factory-training/saves).

#### [phase3-rlaif-alignment/rlaif-model/](./phase3-rlaif-alignment/rlaif-model)

Uses the same prompt → candidate → rating → preference flow as `reward-model/`, but targeted at producing DPO training data and the final RL-aligned dialogue model.

- [ppo-unlabeled-prompts-dataset/](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset) — generates an additional pool of emotional prompts dedicated to the RLAIF round (separate from the reward-model prompt pool) and merges it with the Phase 2 demonstration data. Four numbered scripts plus a shared [_lib.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/_lib.py) (1000 topics + emotion catalogue + dialogue template) mirror the `rm-prompt-dataset/` pipeline:
  - [1-generate_ppo_unlabeled_prompts_dataset.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/1-generate_ppo_unlabeled_prompts_dataset.py), [2-formatted_ppo_unlabeled_prompts_dataset.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/2-formatted_ppo_unlabeled_prompts_dataset.py), [3-train_test_ppo_unlabeled_prompts_dataset_json.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/3-train_test_ppo_unlabeled_prompts_dataset_json.py), [4-merge_demonstration_prompt_ppo_unlabeled_prompts_datasets.py](./phase3-rlaif-alignment/rlaif-model/ppo-unlabeled-prompts-dataset/4-merge_demonstration_prompt_ppo_unlabeled_prompts_datasets.py).
- [dpo-comparison-dataset/](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset) — inference + curation of candidate responses for the DPO round. Contains its own [llama-factory-predict/](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset/llama-factory-predict) workspace plus a [candidate-curation/](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset/candidate-curation) pipeline of three numbered scripts (`1-embeddings_*`, `2-delete_respones_*`, `3-add_negative_samples_*` with `_test` variants) backed by [_lib.py](./phase3-rlaif-alignment/rlaif-model/dpo-comparison-dataset/candidate-curation/_lib.py).
- [dpo-preference-dataset/](./phase3-rlaif-alignment/rlaif-model/dpo-preference-dataset) — same 5-step rating pipeline as `rm-preference-dataset/` but consumes the curated DPO candidates and outputs DPO pairs (`1-preprocess_*` → `5-format_*`, all with `_test` variants and shared [_lib.py](./phase3-rlaif-alignment/rlaif-model/dpo-preference-dataset/_lib.py)). The cross-cutting plot step [6-rates_to_ranks_dpr.py](./phase3-rlaif-alignment/rlaif-model/dpo-preference-dataset/6-rates_to_ranks_dpr.py) renders combined train+test plots over reward-model + RLAIF rates.
- [rlaif-llama-factory-training/](./phase3-rlaif-alignment/rlaif-model/rlaif-llama-factory-training) — current LLaMA-Factory workspace used for the DPO / RLAIF training runs on top of the SFT checkpoint.
- [rlaif-llama-factory-legacy/](./phase3-rlaif-alignment/rlaif-model/rlaif-llama-factory-legacy) — earlier version of the same workspace, kept for reference and to reproduce previous experiments.

## Main technologies

- Python 3.10
- PyTorch
- Hugging Face `transformers`, `datasets`, `accelerate`, `peft` and `trl`
- `LLaMA-Factory`
- OpenAI, Anthropic, Gemini and Azure AI inference APIs
- Jupyter, pandas and plotting libraries for analysis

## Environment

The repository already contains a local virtual environment at `./.vrlaif`. The inspected interpreter there is Python `3.10.12`, which matches the project target well.

The dependency manifest in [requirements.txt](./requirements.txt) was prepared to cover:

- The custom scripts under `phase1-foundation-eval` and `phase3-rlaif-alignment`.
- The analysis notebooks and annotation tooling.
- The embedded `LLaMA-Factory` workflows used throughout the project.

## Installation

```bash
source ./.vrlaif/bin/activate
pip install -r requirements.txt
pip install -e ./llama-factory
pip install "transformers==4.48.3"
```

The second line installs the canonical LLaMA-Factory copy as an editable package so that every workspace can call `llamafactory-cli` against the same source tree (see [LLaMA-Factory: canonical install + thin workspaces](#llama-factory-canonical-install--thin-workspaces)). The third line pins `transformers` to a version inside the range LLaMA-Factory 0.9.2 supports (`>=4.41.2,<=4.48.3`, excluding 4.46.x / 4.47.x / 4.48.0); newer transformers versions break the canonical CLI.

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
- `config_gpt.json` — Azure OpenAI deployments. Used by the Phase 2 demonstration generator and the Phase 3 prompt / PPO-prompt scripts.
- `config_llm.json` — multi-provider LLM judges (Azure OpenAI + Anthropic + Gemini + Llama). Used by the Phase 3 preference-rating scripts.

All `config_gpt.json` and `config_llm.json` files share the same nested-by-deployment schema (`{"<DEPLOYMENT>": {"MODEL": ..., "AZURE_OPENAI_ENDPOINT": ..., ...}}`) so the same loader code works in every phase.

Recommended practice:

- Keep secrets outside versioned source files whenever possible.
- Use environment variables or ignored local config files for real credentials.
- Avoid committing API keys, endpoints or other sensitive data into notebooks and scripts.

Recommended practice:

- Keep secrets outside versioned source files whenever possible.
- Use environment variables or ignored local config files for real credentials.
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

### Run training with bundled LLaMA-Factory copies

The repository includes several local `LLaMA-Factory` instances under `phase2-sft-alignment` and `phase3-rlaif-alignment`. Those are used for:

- Supervised fine-tuning
- Reward-model training
- Prediction/evaluation pipelines
- DPO or RLAIF-style optimization

In practice, these runs are usually launched from inside the corresponding subdirectory with project-specific YAML files and shell scripts.

## Data and outputs

This repository stores both source code and experiment artifacts, including:

- CSV and JSON dialogue datasets
- Jupyter notebooks for analysis
- Experiment logs
- Model predictions and ratings
- Training and evaluation outputs under the bundled workspaces

Because the project is research-oriented, it is useful to separate conceptually:

- Reusable source code
- Generated datasets
- Training checkpoints
- Evaluation outputs
- Temporary logs

## Reproducibility recommendations

- Use Python `3.10.x`.
- Install dependencies from `requirements.txt` before reproducing old experiments.
- Record the exact CUDA, driver and PyTorch build used on each machine.
- Keep track of the exact `LLaMA-Factory` subdirectory used for each run.
- Version control configs carefully, but keep credentials out of the repository.
- Save logs independently for foundation, SFT, reward-model and RL/RLAIF runs.

## Known caveats

- This is not a single packaged application; it is a multi-phase research workspace.
- Different parts of the repository embed different `LLaMA-Factory` variants.
- Some historical scripts assume local credentials, private checkpoints or machine-specific paths.
- Large parts of the project are experiment-driven, so not every folder has the same level of cleanup or standardization.

## Suggested next improvements

- Split dependencies into `base`, `gpu` and `dev` requirement files.
- Move API secrets fully to environment variables.
- Add smoke tests for each phase.

## License and provenance

This repository contains custom project code together with local copies of upstream [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) code under `phase2-sft-alignment` and `phase3-rlaif-alignment`. The licensing and attribution of those upstream components should be preserved in their respective subdirectories.
