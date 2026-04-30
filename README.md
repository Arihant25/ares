# ARES — Misconception Detection and Socratic Question Generation

This repository contains the full experimental pipeline for the ARES research paper. The study evaluates four language models — two large (Grok 4.1 Fast, Qwen3 235B) and two small (Gemma 3n E2B, LFM2-VL-3B) — on two educational NLP tasks: detecting student misconceptions and generating Socratic questions to guide learning.

---

## Research Overview

Two tasks are studied:

1. **Misconception Detection**: Given a student's statement, identify the specific incorrect belief underlying it.
2. **Socratic Question Generation**: Given a misconception, produce one Socratic question that guides the student to discover the correct understanding without being told the answer directly.

Three experimental conditions are compared:

| Condition | Description |
|-----------|-------------|
| Baseline | Single combined prompt: model detects misconception and generates a question in one shot |
| Task-focused | Two sequential prompts: first detect misconception, then generate question with the detected misconception as additional context |
| Finetuned | Two specialist models (one per task), fine-tuned on `data/finetuning.json` using LoRA |

The Gemma and LFM2 models are each fine-tuned into a MisconceptionDetector and a SocraticGenerator, yielding two finetuned pipelines.

---

## Repository Structure

```
.
├── data/
│   ├── finetuning.json          Training data: nested JSON with student statements,
│   │                            misconceptions, Socratic sequences, and resolution insights
│   ├── finalTestSet.jsonc       Test set used for all evaluation runs
│   ├── socraticGenerations.json Supplementary Socratic generation data
│   └── WhyTestSetIsGood.md      Rationale for the test set design
│
├── finetune.py                  Fine-tunes Gemma or LFM2 on the misconception or Socratic task
├── test_baseline.py             Runs baseline evaluation (single prompt) for one model
├── test_taskfocused.py          Runs task-focused evaluation (two prompts) for one model
├── test_finetuned.py            Runs the finetuned model pipeline for Gemma or LFM2
├── app.py                       FastAPI web UI for blind human evaluation
├── statistical_analysis.py      Python script to generate LaTeX tables for the paper
├── run_all.sh                   Orchestrates all runs in the correct order
│
├── outputs/                     All output JSON files (one per run, created at runtime)
│   ├── baseline_grok.json
│   ├── baseline_qwen.json
│   ├── baseline_gemma.json
│   ├── baseline_lfm2.json
│   ├── taskfocused_grok.json
│   ├── taskfocused_qwen.json
│   ├── taskfocused_gemma.json
│   ├── taskfocused_lfm2.json
│   ├── finetuned_gemma.json
│   ├── finetuned_lfm2.json
│   ├── kappa_evaluation.json    Test evaluations specifically for Cohen's Kappa analysis
│   └── evaluation.json          Human evaluation scores from the web UI
│
├── logs/                        Per-run stdout/stderr logs (created at runtime)
├── models/                      Saved LoRA adapters (created at runtime, gitignored)
│
├── pyproject.toml               Python dependencies (managed with uv)
├── .env.example                 Template for environment variables
└── .gitignore
```

---

## System Specifications

All experiments reported in the paper were conducted on the following hardware:

| Component | Specification |
|-----------|---------------|
| Machine | Lenovo ThinkStation P5 |
| CPU | Intel Xeon w3-2435 (8-core / 16-thread, 3.1 GHz base) |
| RAM | 32 GB DDR5 ECC |
| GPU | NVIDIA RTX A2000 (16 GB GDDR6 VRAM) |
| Disk | 1 TB NVMe SSD |
| OS | Ubuntu 24.10 LTS |

> **Note:** A GPU with less VRAM may require reducing batch size or sequence length in `finetune.py`.

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — fast Python package manager
- [ollama](https://ollama.com/) — for local Gemma and LFM2 inference (unfinetuned runs)
- A [Weights & Biases](https://wandb.ai/) account for finetuning tracking
- An [OpenRouter](https://openrouter.ai/) API key for Grok and Qwen inference

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd ares
git checkout final

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install packages
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r pyproject.toml

# Install unsloth with the correct CUDA/PyTorch wheel for your system.
# For CUDA 12.4 + PyTorch 2.5.0:
pip install "unsloth[cu124-torch250]"
# Or for a safe default:
pip install unsloth
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your keys:
#   OPENROUTER_API_KEY=sk-or-v1-...
#   WANDB_API_KEY=...
```

### 3. Pull ollama models (for unfinetuned SLM runs)

```bash
ollama pull hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q8_0
ollama pull hf.co/LiquidAI/LFM2-VL-3B-GGUF:Q8_0
```

---

## Running the Full Pipeline

The easiest way to run everything is through `run_all.sh`, which handles sequencing and parallelisation:

```bash
bash run_all.sh
```

This runs in three phases:
1. **Finetuning** (sequential, ~2-4 hours per variant)
2. **Evaluation** (LLM runs in parallel; SLM/finetuned runs sequential)
3. **Web UI** (starts after tests complete)

### Flags

| Flag | Effect |
|------|--------|
| `--skip-finetune` | Skip finetuning, use existing adapter weights in `models/` |
| `--skip-tests` | Skip evaluation runs, go straight to web UI |
| `--only-ui` | Start only the web UI (assumes outputs already exist) |

---

## Running Components Individually

### Finetuning

Fine-tune a model on a specific task:

```bash
python finetune.py --model gemma --task misconception
python finetune.py --model gemma --task socratic
python finetune.py --model lfm2  --task misconception
python finetune.py --model lfm2  --task socratic
```

Adapters are saved to `models/{model}-{task}/adapter/`. Training config is saved to `models/{model}-{task}/config.json`. Runs are tracked in Weights & Biases under project `ares-research`.

### Evaluation runs

All test scripts accept `--model [grok|qwen|gemma|lfm2]`. Output is written to `outputs/` and saved after each item, so runs can be resumed at any time.

```bash
# Baseline (one combined prompt)
python test_baseline.py --model grok
python test_baseline.py --model qwen
python test_baseline.py --model gemma
python test_baseline.py --model lfm2

# Task-focused (two sequential prompts)
python test_taskfocused.py --model grok
python test_taskfocused.py --model qwen
python test_taskfocused.py --model gemma
python test_taskfocused.py --model lfm2

# Finetuned model pipeline (requires adapters in models/)
python test_finetuned.py --model gemma
python test_finetuned.py --model lfm2
```

### Web evaluation UI

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in a browser.

---

## Web Evaluation Interface

The web UI supports blind human evaluation of model outputs.

### Pages

| URL | Description |
|-----|-------------|
| `/` | Landing page — enter your name to begin |
| `/evaluate` | Evaluation interface — pick a task type, rate outputs blindly |
| `/kappa` | Kappa evaluation interface — fixed subsample of prompts for Cohen's Kappa |
| `/analysis` | Inter-rater agreement stats, per-model average scores |
| `/analysis/disagreements` | Items where evaluators disagreed (model info revealed here) |

### Scoring parameters

**Socratic question outputs** (baseline, task-focused step 2, finetuned step 2):
1. Relevance — directly addresses the specific misconception
2. Socratic Quality — guides discovery rather than giving the answer
3. Clarity — clear and well-phrased
4. Cognitive Challenge — promotes meaningful reflection
5. Naturalness — sounds like something a real teacher would ask

**Misconception detection outputs** (task-focused step 1):
1. Accuracy — correctly identifies the actual misconception
2. Precision — specific and targeted, not vague
3. Clarity — clearly and concisely expressed
4. Depth — captures the underlying cognitive error
5. Actionability — would help a teacher address this gap

### Blind evaluation

Evaluators see only the input and output. The model and approach (e.g., `baseline_grok`) are hidden during voting. They are only revealed on the `/analysis/disagreements` page for post-hoc analysis.

### Scores storage

All scores are appended to `outputs/evaluation.json`.

---

## Output Format

Each output file is a JSON array. Every entry has:

```json
{
  "Input": "student statement text",
  "GroundTruth_Misconception": "...",
  "GroundTruth_Question": "...",
  "Output": "model output",
  "Model-Approach": "grok-baseline",
  "Latency": 1.23
}
```

Task-focused and finetuned entries additionally include:

```json
{
  "Detected_Misconception": "step 1 output",
  "Step1_Latency": 0.45
}
```

---

## Models

| Key | Model | Interface |
|-----|-------|-----------|
| `grok` | x-ai/grok-4.1-fast | OpenRouter |
| `qwen` | qwen/qwen3-235b-a22b-2507 | OpenRouter |
| `gemma` | unsloth/gemma-3n-E2B-it (base via ollama; fine-tuned via HF) | ollama / transformers |
| `lfm2` | LiquidAI/LFM2-VL-3B (base via ollama; fine-tuned via HF) | ollama / transformers |

---

## Reproducing Results

The steps below replicate the full experimental pipeline as run on the [ThinkStation P5 hardware](#system-specifications) described above. Latency figures will vary with different hardware; all other results (scores, rankings, statistics) should be reproducible regardless of GPU model, provided VRAM is sufficient.

### Step 1 — Environment setup

```bash
git clone <repo-url>
cd ares
git checkout final

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

uv pip install -r pyproject.toml

# Install Unsloth for your CUDA version (CUDA 12.x + PyTorch 2.5):
pip install "unsloth[cu124-torch250]"
# Fallback:
pip install unsloth
```

### Step 2 — Configure API keys

```bash
cp .env.example .env
# Fill in OPENROUTER_API_KEY and WANDB_API_KEY
```

### Step 3 — Pull local models via ollama

```bash
ollama pull hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q8_0
ollama pull hf.co/LiquidAI/LFM2-VL-3B-GGUF:Q8_0
```

### Step 4 — Run the full pipeline

```bash
bash run_all.sh
```

Expected runtimes on the ThinkStation P5:

| Phase | Approx. duration |
|-------|------------------|
| Finetuning (4 adapters total) | ~2–4 hours per adapter |
| Baseline + task-focused evaluation (all models) | ~1–2 hours |
| Finetuned evaluation (Gemma + LFM2) | ~30–60 minutes |

### Step 5 — Human evaluation

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Have **at least two independent evaluators** complete both the `/evaluate` and `/kappa` interfaces before proceeding.

### Step 6 — Statistical analysis

```bash
uv run --with scipy --with matplotlib python statistical_analysis.py
```

This reads from `outputs/kappa_evaluation.json`, `outputs/cross_model_evaluation.json`, `outputs/evaluation.json`, and all ten per-run output files, then produces:

- `author_ext_stats.tex` — Wilcoxon + Spearman generalisability table (external raters vs. authors)
- `llm_judge_bias.tex` — LLM-as-judge inflation and rank-correlation table
- `llm_judge_stats.json` — numerical snapshot of self-bias and composite inflation
- `latency_stats.tex` — mean, std, and median inference latency per configuration, with per-phase breakdown for two-step approaches
- `latency_plot.pdf` — grouped bar chart of median latency across all models and configurations

### Resuming interrupted runs

All test scripts checkpoint after every item — if a run is interrupted, re-running the same command will pick up where it left off. Finetuning must be restarted from scratch if interrupted, but saved adapters in `models/` will be reused if `--skip-finetune` is passed to `run_all.sh`.
