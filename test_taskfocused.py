"""
test_taskfocused.py

Two-step task-focused evaluation:
  Step 1 — Detect the misconception in the student statement.
  Step 2 — Generate a Socratic question based on the detected misconception.

Usage:
    python test_taskfocused.py --model [grok|qwen|gemma|lfm2]
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
    "smollm3": "hf.co/unsloth/SmolLM3-3B-GGUF:Q8_0",
}

# LFM2-VL is unsupported in Ollama (PR #14069 closed as stale); use llama-cpp-python directly.
LFM2_GGUF = "/usr/share/ollama/.ollama/models/blobs/sha256-2b1c0ecb28b802cc1c8a8afd42a4746ac9e563e33fe2c87c5948864bda23fe39"

_lfm2_model = None


def _get_lfm2():
    global _lfm2_model
    if _lfm2_model is None:
        from llama_cpp import Llama
        print("Loading LFM2-VL via llama-cpp-python...")
        _lfm2_model = Llama(
            model_path=LFM2_GGUF,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        print("LFM2-VL loaded.")
    return _lfm2_model

DATA_PATH = "data/finalTestSet.jsonc"
OUTPUT_DIR = "outputs"

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

STEP1_SYSTEM = (
    "You are an expert educator who identifies misconceptions in student statements."
)


def step1_user(student_statement: str) -> str:
    return (
        f'A student made the following statement:\n\n"{student_statement}"\n\n'
        "Identify the specific misconception underlying this statement. "
        "Be concise and precise — state only the core incorrect belief, "
        "in one or two sentences.\n\n"
        "Misconception:"
    )


STEP2_SYSTEM = (
    "You are a Socratic educator who guides students to discover correct "
    "understanding through questioning."
)


def step2_user(detected_misconception: str, student_statement: str) -> str:
    return (
        f'A student holds the following misconception:\n\n"{detected_misconception}"\n\n'
        f'Their original statement was: "{student_statement}"\n\n'
        "Generate exactly one Socratic question that guides this student to discover "
        "the correct understanding themselves. The question should not give away the "
        "answer — it should provoke reflection.\n\n"
        "Socratic Question:"
    )


# ---------------------------------------------------------------------------
# Data loading
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


def call_llama_cpp(messages: list[dict]) -> tuple[str, float]:
    llm = _get_lfm2()
    start = time.time()
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.3,
        max_tokens=512,
    )
    latency = time.time() - start
    return out["choices"][0]["message"]["content"], latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Task-focused evaluation script")
    parser.add_argument(
        "--model",
        required=True,
        choices=["grok", "qwen", "lfm2", "smollm3"],
        help="Model to use for evaluation",
    )
    args = parser.parse_args()
    model_name = args.model

    output_path = os.path.join(OUTPUT_DIR, f"taskfocused_{model_name}.json")
    items = load_data(DATA_PATH)
    results = load_existing(output_path)

    already_done = {r["Input"] for r in results}
    total = len(items)

    is_openrouter = model_name in OPENROUTER_MODELS
    is_llama_cpp = model_name == "lfm2"
    model_id = OPENROUTER_MODELS.get(model_name) or OLLAMA_MODELS.get(model_name)
    approach_tag = f"{model_name}-taskfocused"

    if is_llama_cpp:
        _get_lfm2()  # load model once upfront

    for i, item in enumerate(items):
        student_statement = item["student_statement"]
        incorrect_belief = item["incorrect_belief"]
        socratic_question = item["socratic_question"]
        category = item["category"]

        print(f"Processing item {i + 1}/{total} [{category}]: {student_statement[:50]}...")

        if student_statement in already_done:
            print(f"  Skipping (already in output).")
            continue

        # -- Step 1: detect misconception ----------------------------------
        step1_messages = [
            {"role": "system", "content": STEP1_SYSTEM},
            {"role": "user", "content": step1_user(student_statement)},
        ]

        try:
            if is_openrouter:
                detected_misconception, step1_latency = call_openrouter(
                    model_id, step1_messages
                )
            elif is_llama_cpp:
                detected_misconception, step1_latency = call_llama_cpp(step1_messages)
            else:
                detected_misconception, step1_latency = call_ollama(
                    model_id, step1_messages
                )
        except Exception as exc:
            print(f"  Error in Step 1 for item {i + 1}: {exc}")
            continue

        # Strip a leading "Misconception:" label if the model echoed it back
        detected_misconception = re.sub(
            r"^\s*Misconception:\s*", "", detected_misconception, flags=re.IGNORECASE
        ).strip()

        # -- Step 2: generate Socratic question ----------------------------
        step2_messages = [
            {"role": "system", "content": STEP2_SYSTEM},
            {
                "role": "user",
                "content": step2_user(detected_misconception, student_statement),
            },
        ]

        try:
            if is_openrouter:
                socratic_output, step2_latency = call_openrouter(
                    model_id, step2_messages
                )
            elif is_llama_cpp:
                socratic_output, step2_latency = call_llama_cpp(step2_messages)
            else:
                socratic_output, step2_latency = call_ollama(
                    model_id, step2_messages
                )
        except Exception as exc:
            print(f"  Error in Step 2 for item {i + 1}: {exc}")
            continue

        # Strip a leading "Socratic Question:" label if echoed back
        socratic_output = re.sub(
            r"^\s*Socratic Question:\s*", "", socratic_output, flags=re.IGNORECASE
        ).strip()

        total_latency = step1_latency + step2_latency

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
        print(
            f"  Done. Step1: {step1_latency:.2f}s  Step2: {step2_latency:.2f}s  "
            f"Total: {total_latency:.2f}s"
        )

    print(f"\nFinished. Results saved to {output_path}")


if __name__ == "__main__":
    main()
