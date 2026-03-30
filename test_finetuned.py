"""
test_finetuned.py

Two-step pipeline evaluation with finetuned SLMs:
    Step 1 — MisconceptionDetector  (finetuned model)
    Step 2 — SocraticGenerator      (finetuned model)

Usage:
    python test_finetuned.py --model gemma
    python test_finetuned.py --model lfm2
"""

import argparse
import json
import os
import re
import time

import unsloth  # must be first to apply optimizations before transformers/peft
import torch
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_PATH = "data/finalTestSet.jsonc"
OUTPUT_DIR = "outputs"

GEMMA_BASE = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"
LFM2_BASE = "LiquidAI/LFM2-VL-3B"

ADAPTER_PATHS = {
    "gemma": {
        "detector": "models/gemma-misconception/adapter",
        "generator": "models/gemma-socratic/adapter",
    },
    "lfm2": {
        "detector": "models/lfm2-misconception/adapter",
        "generator": "models/lfm2-socratic/adapter",
    },
}

# ---------------------------------------------------------------------------
# Prompt templates — must match finetune.py exactly
# ---------------------------------------------------------------------------

DETECTOR_PROMPT = (
    "You are an expert educator analyzing student responses. "
    "Your task is to identify the specific misconception underlying the student's statement.\n\n"
    "Student Statement: {student_statement}\n\n"
    "Misconception:"
)

GENERATOR_PROMPT = (
    "You are a Socratic educator. Given a student's misconception, generate one precise "
    "Socratic question that guides the student to discover the correct understanding "
    "themselves, without directly giving them the answer.\n\n"
    "Misconception: {detected_misconception}\n\n"
    "Socratic Question:"
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    """Load JSONC, return all items from all categories with 'category' field attached."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    raw = re.sub(r"//[^\n]*", "", raw)
    data = json.loads(raw)
    items = []
    for category, category_items in data.items():
        for item in category_items:
            items.append({**item, "category": category})
    return items


def load_existing(output_path: str) -> list[dict]:
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as fh:
            try:
                return json.load(fh)
            except json.JSONDecodeError:
                return []
    return []


def save_results(output_path: str, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gemma_models():
    """Load Gemma base model then apply separate PEFT adapters for each step."""
    from peft import PeftModel
    from unsloth import FastVisionModel  # noqa: F401 (unsloth already imported at top)

    print("Loading Gemma base model ...")
    base_model, processor = FastVisionModel.from_pretrained(
        GEMMA_BASE,
        load_in_4bit=False,
        load_in_8bit=True,
        device_map={"": 0},
    )
    # Use the underlying text tokenizer directly to avoid vision processor overhead
    # during text-only finetuned inference.
    tokenizer = processor.tokenizer

    print("Applying Gemma misconception-detector adapter ...")
    detector_model = PeftModel.from_pretrained(
        base_model, ADAPTER_PATHS["gemma"]["detector"]
    )

    print("Applying Gemma socratic-generator adapter ...")
    generator_model = PeftModel.from_pretrained(
        base_model, ADAPTER_PATHS["gemma"]["generator"]
    )

    return detector_model, generator_model, tokenizer


def load_lfm2_models():
    """Load LFM2 base model (8-bit) then apply separate PEFT adapters."""
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from transformers.models.lfm2_vl import Lfm2VlForConditionalGeneration

    bnb = BitsAndBytesConfig(load_in_8bit=True)

    print("Loading LFM2 base model ...")
    base = Lfm2VlForConditionalGeneration.from_pretrained(
        LFM2_BASE,
        quantization_config=bnb,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(LFM2_BASE, trust_remote_code=True)

    print("Applying LFM2 misconception-detector adapter ...")
    detector_model = PeftModel.from_pretrained(
        base, ADAPTER_PATHS["lfm2"]["detector"]
    )

    print("Applying LFM2 socratic-generator adapter ...")
    generator_model = PeftModel.from_pretrained(
        base, ADAPTER_PATHS["lfm2"]["generator"]
    )

    return detector_model, generator_model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 150, use_chat_template: bool = False) -> tuple[str, float]:
    """Tokenize prompt, run greedy generation, return only the new tokens."""
    device = next(model.parameters()).device
    if use_chat_template:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    # Move input tensors to the model device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    latency = time.time() - start

    # Extract only newly generated tokens
    new_ids = output_ids[0][prompt_len:]
    generated = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return generated, latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Finetuned two-step pipeline evaluation")
    parser.add_argument(
        "--model",
        required=True,
        choices=["gemma", "lfm2"],
        help="Which finetuned model pair to use",
    )
    args = parser.parse_args()
    model_name = args.model
    approach_tag = f"{model_name}-finetuned"

    output_path = os.path.join(OUTPUT_DIR, f"finetuned_{model_name}.json")

    # Load data
    items = load_data(DATA_PATH)
    results = load_existing(output_path)
    already_done = {r["Input"] for r in results}

    total = len(items)
    print(f"Loaded {total} items. {len(already_done)} already processed.")

    # Load models
    if model_name == "gemma":
        detector_model, generator_model, tokenizer = load_gemma_models()
    else:
        detector_model, generator_model, tokenizer = load_lfm2_models()

    use_chat_template = False  # both models finetuned on plain completion, not chat format
    print("Models ready. Starting evaluation ...\n")

    for i, item in enumerate(items):
        student_statement = item["student_statement"]
        incorrect_belief = item["incorrect_belief"]
        socratic_question = item["socratic_question"]
        category = item["category"]

        print(f"[{i + 1}/{total}] [{category}] {student_statement[:60]}...")

        if student_statement in already_done:
            print("  Skipping (already in output).")
            continue

        try:
            # --- Step 1: Misconception detection ---
            det_prompt = DETECTOR_PROMPT.format(student_statement=student_statement)
            detected_misconception, step1_latency = generate_text(
                detector_model, tokenizer, det_prompt, use_chat_template=use_chat_template
            )
            print(f"  Step 1 ({step1_latency:.2f}s): {detected_misconception[:80]}")

            # --- Step 2: Socratic question generation ---
            gen_prompt = GENERATOR_PROMPT.format(
                detected_misconception=detected_misconception
            )
            socratic_output, step2_latency = generate_text(
                generator_model, tokenizer, gen_prompt, use_chat_template=use_chat_template
            )
            total_latency = step1_latency + step2_latency
            print(f"  Step 2 ({step2_latency:.2f}s): {socratic_output[:80]}")

            result = {
                "Category": category,
                "Input": student_statement,
                "GroundTruth_Misconception": incorrect_belief,
                "GroundTruth_Question": socratic_question,
                "Detected_Misconception": detected_misconception,
                "Step1_Latency": round(step1_latency, 4),
                "Output": socratic_output,
                "Model-Approach": approach_tag,
                "Latency": round(total_latency, 4),
            }
            results.append(result)
            save_results(output_path, results)
            print(f"  Saved. Total latency: {total_latency:.2f}s")

        except Exception as exc:
            print(f"  Error on item {i + 1}: {exc}")
            continue

    print(f"\nFinished. {len(results)} results saved to {output_path}")


if __name__ == "__main__":
    main()
