"""
test_baseline.py

Single-prompt baseline evaluation: detects misconception AND generates a
Socratic question in one call.

Usage:
    python test_baseline.py --model [grok|qwen|gemma|lfm2]
"""

import argparse
import json
import os
import re
import time

import ollama
import openai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

OPENROUTER_MODELS = {
    "grok": "x-ai/grok-4.1-fast",
    "qwen": "qwen/qwen3-235b-a22b-2507",
}

OLLAMA_MODELS = {
    "gemma": "hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q8_0",
    "lfm2": "hf.co/LiquidAI/LFM2-VL-3B-GGUF:Q8_0",
}

DATA_PATH = "data/finalTestSet.jsonc"
OUTPUT_DIR = "outputs"

SYSTEM_PROMPT = (
    "You are an expert AI tutor. You analyze student statements to identify "
    "misconceptions and generate Socratic questions to guide learning."
)


def user_prompt(student_statement: str) -> str:
    return (
        f'A student made the following statement:\n\n"{student_statement}"\n\n'
        "Please:\n"
        "1. Identify the specific misconception underlying this statement "
        "(be concise and precise).\n"
        "2. Generate one Socratic question that guides the student to discover "
        "the correct understanding themselves, without directly giving them the answer.\n\n"
        "Format your response as:\n"
        "Misconception: <the identified misconception>\n"
        "Socratic Question: <your question>"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    """Load JSONC by stripping // and /* */ comments before parsing."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    # Remove block comments
    raw = re.sub(r"/\*.*?\*/", "", raw, flags=re.DOTALL)
    # Remove line comments
    raw = re.sub(r"//[^\n]*", "", raw)
    data = json.loads(raw)
    return data["TrainPerturbed"]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

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
# Inference helpers
# ---------------------------------------------------------------------------

def call_openrouter(model_id: str, messages: list[dict]) -> tuple[str, float]:
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    start = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.3,
    )
    latency = time.time() - start
    return resp.choices[0].message.content, latency


def call_ollama(model_id: str, messages: list[dict]) -> tuple[str, float]:
    start = time.time()
    response = ollama.chat(
        model=model_id,
        messages=messages,
        options={"temperature": 0.3},
    )
    latency = time.time() - start
    return response["message"]["content"], latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation script")
    parser.add_argument(
        "--model",
        required=True,
        choices=["grok", "qwen", "gemma", "lfm2"],
        help="Model to use for evaluation",
    )
    args = parser.parse_args()
    model_name = args.model

    output_path = os.path.join(OUTPUT_DIR, f"baseline_{model_name}.json")
    items = load_data(DATA_PATH)
    results = load_existing(output_path)

    already_done = {r["Input"] for r in results}
    total = len(items)

    is_openrouter = model_name in OPENROUTER_MODELS
    model_id = (
        OPENROUTER_MODELS[model_name]
        if is_openrouter
        else OLLAMA_MODELS[model_name]
    )
    approach_tag = f"{model_name}-baseline"

    for i, item in enumerate(items):
        student_statement = item["student_statement"]
        incorrect_belief = item["incorrect_belief"]
        socratic_question = item["socratic_question"]

        print(f"Processing item {i + 1}/{total}: {student_statement[:60]}...")

        if student_statement in already_done:
            print(f"  Skipping (already in output).")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(student_statement)},
        ]

        try:
            if is_openrouter:
                output_text, latency = call_openrouter(model_id, messages)
            else:
                output_text, latency = call_ollama(model_id, messages)
        except Exception as exc:
            print(f"  Error processing item {i + 1}: {exc}")
            continue

        result = {
            "Input": student_statement,
            "GroundTruth_Misconception": incorrect_belief,
            "GroundTruth_Question": socratic_question,
            "Output": output_text,
            "Model-Approach": approach_tag,
            "Latency": round(latency, 4),
        }
        results.append(result)
        save_results(output_path, results)
        print(f"  Done. Latency: {latency:.2f}s")

    print(f"\nFinished. Results saved to {output_path}")


if __name__ == "__main__":
    main()
