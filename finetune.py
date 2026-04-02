"""
finetune.py — Fine-tune SLMs for two education tasks.

Tasks:
  misconception  — detect the incorrect belief behind a student statement
  socratic       — generate a Socratic question for a given misconception

Models:
  lfm2     — LiquidAI/LFM2-VL-3B       (standard PEFT + bitsandbytes 8-bit)
  smollm3  — HuggingFaceTB/SmolLM3-3B  (standard PEFT + bitsandbytes 8-bit)

Usage:
  python finetune.py --model lfm2     --task misconception
  python finetune.py --model smollm3  --task socratic
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants / Hyperparameters
# ---------------------------------------------------------------------------

LFM2_MODEL_ID     = "LiquidAI/LFM2-VL-3B"       # base fp16, loaded in 8-bit via bnb
SMOLLM3_MODEL_ID  = "HuggingFaceTB/SmolLM3-3B"  # base fp16, loaded in 8-bit via bnb

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_BIAS = "none"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# Fallback target modules for LFM2 if the full set fails
LORA_TARGET_MODULES_FALLBACK = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
NUM_EPOCHS = 3
WARMUP_STEPS = 10
LR_SCHEDULER = "linear"
SEED = 3407
MAX_SEQ_LENGTH = 1024

WANDB_PROJECT = "ares-research"

FINETUNING_DATA_PATH = Path("data/finetuning.json")
TEST_SET_PATH = Path("data/finalTestSet.jsonc")

MISCONCEPTION_PROMPT = (
    "You are an expert educator analyzing student responses. "
    "Your task is to identify the specific misconception underlying the student's statement.\n\n"
    "Student Statement: {student_statement}\n\n"
    "Misconception:\n"
)

MISCONCEPTION_NEGATIVE_PROMPT = (
    "You are an expert educator analyzing student responses. "
    "Your task is to identify the specific misconception underlying the student's statement.\n\n"
    "Student Statement: {resolution_insight}\n\n"
    "Misconception:\n"
)

SOCRATIC_PROMPT = (
    "You are a Socratic educator. Given a student's misconception, generate one precise Socratic "
    "question that guides the student to discover the correct understanding themselves, without "
    "directly giving them the answer.\n\n"
    "Misconception: {incorrect_belief}\n\n"
    "Socratic Question:\n"
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SLMs on education tasks (misconception detection / Socratic generation)"
    )
    parser.add_argument(
        "--model",
        choices=["lfm2", "smollm3"],
        required=True,
        help="Which model to fine-tune: lfm2 or smollm3",
    )
    parser.add_argument(
        "--task",
        choices=["misconception", "socratic"],
        required=True,
        help="Which task to train on: misconception or socratic",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_finetuning_json(path: Path) -> list[dict]:
    """Load and flatten the nested finetuning.json into a list of misconception dicts.

    Each returned dict contains:
        student_statement, incorrect_belief, socratic_sequence (list),
        resolution_insight, bloom_level, topic, concept
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for level in raw:
        for chapter in level.get("chapters", []):
            topic = chapter.get("topic", "Unknown")
            for concept_block in chapter.get("concepts", []):
                concept = concept_block.get("concept", "Unknown")
                for mc in concept_block.get("misconceptions", []):
                    records.append(
                        {
                            "student_statement": mc.get("student_statement", "").strip(),
                            "incorrect_belief": mc.get("incorrect_belief", "").strip(),
                            "socratic_sequence": mc.get("socratic_sequence", []),
                            "resolution_insight": mc.get("resolution_insight", "").strip(),
                            "bloom_level": mc.get("bloom_level", "").strip(),
                            "topic": topic,
                            "concept": concept,
                        }
                    )

    print(f"  Loaded {len(records)} misconception records from {path}")
    return records


def load_test_set_jsonc(path: Path) -> list[dict]:
    """Load finalTestSet.jsonc, stripping JS-style // comments before parsing."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    content = re.sub(r"//.*", "", content)
    data = json.loads(content)
    items = data.get("TrainPerturbed", [])
    print(f"  Loaded {len(items)} test-set items from {path}")
    return items


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def build_misconception_dataset(records: list[dict], test_items: list[dict]) -> list[dict]:
    """Build (text, topic) tuples for the MisconceptionDetector task.

    Positive examples: student_statement → incorrect_belief
    Negative examples: resolution_insight → "" (int(0.2 * len(test_items)) samples)
    """
    examples = []

    # Positive examples from finetuning data
    for rec in records:
        if not rec["student_statement"] or not rec["incorrect_belief"]:
            continue
        prompt = MISCONCEPTION_PROMPT.format(student_statement=rec["student_statement"])
        text = prompt + rec["incorrect_belief"]
        examples.append({"text": text, "topic": rec["topic"]})

    # Negative examples — use resolution_insight with empty target
    n_neg = int(0.2 * len(test_items))
    neg_pool = [r for r in records if r["resolution_insight"]]
    random.seed(SEED)
    neg_sample = random.sample(neg_pool, min(n_neg, len(neg_pool)))
    for rec in neg_sample:
        prompt = MISCONCEPTION_NEGATIVE_PROMPT.format(
            resolution_insight=rec["resolution_insight"]
        )
        text = prompt  # empty string output — model learns to produce nothing
        examples.append({"text": text, "topic": rec["topic"]})

    random.shuffle(examples)
    print(f"  MisconceptionDetector dataset: {len(examples)} examples "
          f"({len(examples) - len(neg_sample)} positive, {len(neg_sample)} negative)")
    return examples


def build_socratic_dataset(records: list[dict]) -> list[dict]:
    """Build (text, topic) tuples for the SocraticGenerator task.

    Input:  incorrect_belief
    Output: first item of socratic_sequence
    """
    examples = []
    for rec in records:
        if not rec["incorrect_belief"] or not rec["socratic_sequence"]:
            continue
        first_question = rec["socratic_sequence"][0].strip()
        prompt = SOCRATIC_PROMPT.format(incorrect_belief=rec["incorrect_belief"])
        text = prompt + first_question
        examples.append({"text": text, "topic": rec["topic"]})

    print(f"  SocraticGenerator dataset: {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_dataset(
    examples: list[dict], test_size: float = 0.1
) -> tuple[list[dict], list[dict]]:
    """90/10 split, stratified by topic when possible."""
    from sklearn.model_selection import train_test_split

    topics = [e["topic"] for e in examples]

    # Stratified split requires each class to have >= 2 samples
    topic_counts: dict[str, int] = {}
    for t in topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    valid_stratify = all(c >= 2 for c in topic_counts.values())

    if valid_stratify and len(examples) >= 10:
        train, val = train_test_split(
            examples, test_size=test_size, random_state=SEED, stratify=topics
        )
    else:
        train, val = train_test_split(examples, test_size=test_size, random_state=SEED)

    print(f"  Split → {len(train)} train / {len(val)} val")
    return train, val


# ---------------------------------------------------------------------------
# LFM2 model setup (standard PEFT + bitsandbytes 8-bit)
# ---------------------------------------------------------------------------

def load_lfm2(task: str):
    """Load LFM2-VL-3B with bitsandbytes 8-bit and apply LoRA via PEFT."""
    print("\n  Loading LFM2-VL-3B with bitsandbytes 8-bit …")
    import torch
    from transformers.models.lfm2_vl import Lfm2VlForConditionalGeneration
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        LFM2_MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # LFM2-VL-3B is a vision-language model; load with its dedicated class
    model = Lfm2VlForConditionalGeneration.from_pretrained(
        LFM2_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False

    # Freeze vision tower — only fine-tune the language model
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False

    # LoRA targets within the language model sub-module
    # prefix with "model.language_model." to reach the right layers
    def _make_peft(target_modules):
        lora_cfg = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias=LORA_BIAS,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_cfg)

    try:
        peft_model = _make_peft(LORA_TARGET_MODULES)
        print("  Applied LoRA with full target_modules set.")
    except ValueError as exc:
        print(f"  Full target_modules failed ({exc}); falling back to attention-only.")
        peft_model = _make_peft(LORA_TARGET_MODULES_FALLBACK)

    peft_model.print_trainable_parameters()

    # Enable gradient checkpointing
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()

    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# SmolLM3 model setup (standard PEFT + bitsandbytes 8-bit)
# ---------------------------------------------------------------------------

def load_smollm3(task: str):
    """Load SmolLM3-3B with bitsandbytes 8-bit and apply LoRA via PEFT."""
    print("\n  Loading SmolLM3-3B with bitsandbytes 8-bit …")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        SMOLLM3_MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        SMOLLM3_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable()

    return peft_model, tokenizer


# ---------------------------------------------------------------------------
# Formatting function for SFTTrainer
# ---------------------------------------------------------------------------

def make_formatting_func(tokenizer, examples: list[dict]):
    """Return a formatting_func closure for TRL SFTTrainer.

    The function receives a batch dict and returns a list of strings.
    """
    eos = tokenizer.eos_token or ""

    def formatting_func(batch):
        texts = batch["text"]
        return [t + eos for t in texts]

    return formatting_func


# ---------------------------------------------------------------------------
# WandB initialisation
# ---------------------------------------------------------------------------

def init_wandb(model_key: str, task: str, hyperparams: dict):
    """Initialise a WandB run, using WANDB_API_KEY or any existing wandb login."""
    try:
        import wandb  # type: ignore

        run_name = f"{model_key}-{task}-finetune"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config=hyperparams,
        )
        os.environ.pop("WANDB_DISABLED", None)
        print(f"  WandB run initialised: project={WANDB_PROJECT}, name={run_name}")
    except Exception as exc:
        print(f"  WandB initialisation failed ({exc}) — disabling WandB logging.")
        os.environ["WANDB_DISABLED"] = "true"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model,
    tokenizer,
    train_examples: list[dict],
    val_examples: list[dict],
    output_dir: Path,
    model_key: str,
    task: str,
):
    """Run SFTTrainer and save the LoRA adapter + config."""
    from datasets import Dataset as HFDataset  # type: ignore
    from trl import SFTTrainer, SFTConfig  # type: ignore

    train_ds = HFDataset.from_list(train_examples)
    val_ds = HFDataset.from_list(val_examples)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    optimizer_name = "adamw_8bit" if model_key == "gemma" else "adamw_torch"
    report_to = "none" if os.environ.get("WANDB_DISABLED") else "wandb"

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        optim=optimizer_name,
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        dataset_text_field="text",
        # Disable packing to keep individual samples intact
        packing=False,
        # Remove columns not needed by the model
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_examples) > 0 else None,
    )

    print("\n" + "=" * 60)
    print(f"  Starting training: model={model_key}, task={task}")
    print(f"  Train samples : {len(train_examples)}")
    print(f"  Val   samples : {len(val_examples)}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}  (grad accum {GRAD_ACCUM})")
    print(f"  LR            : {LEARNING_RATE}")
    print(f"  Optimizer     : {optimizer_name}")
    print(f"  Output dir    : {output_dir}")
    print("=" * 60 + "\n")

    trainer.train()

    # Save only the adapter (not merged) to save disk space
    print(f"\n  Saving LoRA adapter to {adapter_dir} …")
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Save training config
    hyperparams = {
        "model_id": {"lfm2": LFM2_MODEL_ID, "smollm3": SMOLLM3_MODEL_ID}[model_key],
        "model_key": model_key,
        "task": task,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_bias": LORA_BIAS,
        "lora_target_modules": LORA_TARGET_MODULES,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_epochs": NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "lr_scheduler": LR_SCHEDULER,
        "optimizer": optimizer_name,
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"  Training config saved to {config_path}")

    # Final evaluation
    if len(val_examples) > 0:
        print("\n  Running final evaluation …")
        eval_results = trainer.evaluate()
        print(f"  Eval results: {eval_results}")
        eval_path = output_dir / "eval_results.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print(f"  Eval results saved to {eval_path}")

    return hyperparams


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    model_key: str = args.model
    task: str = args.task

    print("\n" + "=" * 60)
    print(f"  ARES Fine-tuning Pipeline")
    print(f"  Model : {model_key}")
    print(f"  Task  : {task}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[ 1/5 ] Loading data …")

    finetuning_path = Path("data/finetuning.json")
    test_set_path = Path("data/finalTestSet.jsonc")

    if not finetuning_path.exists():
        # Try backup location
        finetuning_path = Path("data/ares_data_backup/finetuning.json")
    if not test_set_path.exists():
        test_set_path = Path("data/ares_data_backup/finalTestSet.jsonc")

    records = load_finetuning_json(finetuning_path)

    test_items: list[dict] = []
    if test_set_path.exists():
        test_items = load_test_set_jsonc(test_set_path)
    else:
        print(f"  Warning: {test_set_path} not found — negative examples will use records only.")

    # ------------------------------------------------------------------
    # 2. Build task-specific dataset
    # ------------------------------------------------------------------
    print("\n[ 2/5 ] Building dataset …")

    if task == "misconception":
        examples = build_misconception_dataset(records, test_items)
    else:
        examples = build_socratic_dataset(records)

    train_examples, val_examples = split_dataset(examples)

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    print("\n[ 3/5 ] Loading model …")

    if model_key == "lfm2":
        model, tokenizer = load_lfm2(task)
    else:
        model, tokenizer = load_smollm3(task)

    # ------------------------------------------------------------------
    # 4. WandB
    # ------------------------------------------------------------------
    print("\n[ 4/5 ] Initialising WandB …")

    hyperparams_preview = {
        "model_key": model_key,
        "task": task,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_epochs": NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "lr_scheduler": LR_SCHEDULER,
        "seed": SEED,
        "max_seq_length": MAX_SEQ_LENGTH,
        "train_samples": len(train_examples),
        "val_samples": len(val_examples),
    }
    init_wandb(model_key, task, hyperparams_preview)

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    print("\n[ 5/5 ] Training …")

    output_dir = Path(f"models/{model_key}-{task}")
    train(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        val_examples=val_examples,
        output_dir=output_dir,
        model_key=model_key,
        task=task,
    )

    # Finish WandB run if active
    if not os.environ.get("WANDB_DISABLED"):
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Adapter saved to: models/{model_key}-{task}/adapter/")
    print(f"  Config  saved to: models/{model_key}-{task}/config.json")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
