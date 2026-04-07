"""
Cross-model peer evaluation for ARES outputs.

This script lets every model evaluate all models' outputs (including self)
across baseline, task-focused, and finetuned runs.

Usage examples:
    uv run evaluate_peer_models.py --estimate-only
    uv run evaluate_peer_models.py
    uv run evaluate_peer_models.py --evaluators grok qwen --max-items 50
"""

import argparse
import concurrent.futures
import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path

import ollama
import openai
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_MODELS = {
    "grok": "x-ai/grok-4.1-fast",
    "qwen": "qwen/qwen3-235b-a22b-2507",
}

# OpenRouter headline pricing as of 2026-04-07 (USD per 1M tokens).
OPENROUTER_PRICING = {
    "grok": {"input": 0.20, "output": 0.50},
    "qwen": {"input": 0.071, "output": 0.10},
}

OLLAMA_MODELS = {
    "smollm3": "hf.co/unsloth/SmolLM3-3B-GGUF:Q8_0",
}

# LFM2-VL is unsupported in Ollama; mirror existing test scripts.
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


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def clean_json_text(text: str) -> str:
    text = strip_think_tags(text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict:
    cleaned = clean_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                repaired = re.sub(r"([\}\]\"\d])\s*\n\s*(\")", r"\1,\n  \2", repaired)
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    # Fallback: salvage rubric fields from malformed JSON-like text.
    score_keys = ["relevance", "socratic_quality", "clarity", "cognitive_challenge", "naturalness"]
    scores = {}
    for key in score_keys:
        m = re.search(rf'"{key}"\s*:\s*(-?\d+)', cleaned)
        if m:
            scores[key] = int(m.group(1))

    rationale = ""
    rmatch = re.search(r'"rationale"\s*:\s*"([^\"]*)"', cleaned, flags=re.DOTALL)
    if rmatch:
        rationale = rmatch.group(1).strip()

    if not scores and not rationale:
        raise json.JSONDecodeError("Unable to parse evaluator JSON", cleaned, 0)

    return {"scores": scores, "rationale": rationale}


def clamp_score(value) -> int:
    try:
        score = int(value)
    except Exception:
        return 3
    return max(1, min(5, score))


def validate_eval(payload: dict) -> tuple[dict, str]:
    scores = payload.get("scores", {})
    normalized = {
        "relevance": clamp_score(scores.get("relevance", 3)),
        "socratic_quality": clamp_score(scores.get("socratic_quality", 3)),
        "clarity": clamp_score(scores.get("clarity", 3)),
        "cognitive_challenge": clamp_score(scores.get("cognitive_challenge", 3)),
        "naturalness": clamp_score(scores.get("naturalness", 3)),
    }
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        rationale = "No rationale provided."
    return normalized, rationale


def infer_base_model(run_id: str, model_approach: str | None) -> str:
    if model_approach and "-" in model_approach:
        return model_approach.split("-", 1)[0].lower()
    if "_" in run_id:
        return run_id.split("_", 1)[1].lower()
    return run_id.lower()


def load_output_records(outputs_dir: Path) -> list[dict]:
    records = []
    patterns = ["baseline_*.json", "taskfocused_*.json", "finetuned_*.json"]
    for pattern in patterns:
        for path in sorted(outputs_dir.glob(pattern)):
            run_id = path.stem
            with open(path, "r", encoding="utf-8") as fh:
                try:
                    items = json.load(fh)
                except json.JSONDecodeError:
                    print(f"Warning: could not parse {path}, skipping")
                    continue

            for idx, item in enumerate(items):
                records.append(
                    {
                        "run_id": run_id,
                        "item_index": idx,
                        "category": item.get("Category", ""),
                        "input": item.get("Input", ""),
                        "ground_truth_misconception": item.get("GroundTruth_Misconception", ""),
                        "output": item.get("Output", ""),
                        "model_approach": item.get("Model-Approach", ""),
                        "target_base": infer_base_model(run_id, item.get("Model-Approach")),
                    }
                )
    return records


def build_messages(record: dict) -> list[dict]:
    system = (
        "You are an expert evaluator of Socratic tutoring questions. "
        "Score only the candidate question. Be strict, fair, and rubric-grounded. "
        "Return ONLY JSON."
    )

    user = (
        "Evaluate the candidate Socratic question below on a 1-5 scale for each rubric criterion.\n"
        "Rubric:\n"
        "- relevance: directly addresses the provided misconception\n"
        "- socratic_quality: guides discovery without giving away the answer\n"
        "- clarity: clear and well-phrased\n"
        "- cognitive_challenge: prompts meaningful reflection\n"
        "- naturalness: sounds like a real teacher\n\n"
        f"Student statement:\n{record['input']}\n\n"
        f"Target misconception (ground truth):\n{record['ground_truth_misconception']}\n\n"
        f"Candidate question:\n{record['output']}\n\n"
        "Important: Score relevance and Socratic quality with respect to the target misconception above.\n\n"
        "Respond with EXACTLY this JSON schema:\n"
        "{\n"
        "  \"scores\": {\n"
        "    \"relevance\": <1-5 int>,\n"
        "    \"socratic_quality\": <1-5 int>,\n"
        "    \"clarity\": <1-5 int>,\n"
        "    \"cognitive_challenge\": <1-5 int>,\n"
        "    \"naturalness\": <1-5 int>\n"
        "  },\n"
        "  \"rationale\": \"<= 35 words\"\n"
        "}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_openrouter(judge: str, messages: list[dict]) -> tuple[str, float, dict | None]:
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    start = time.time()
    resp = client.chat.completions.create(
        model=OPENROUTER_MODELS[judge],
        messages=messages,
        temperature=0.0,
    )
    latency = time.time() - start
    usage = None
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }
    return resp.choices[0].message.content, latency, usage


def call_ollama(judge: str, messages: list[dict]) -> tuple[str, float]:
    start = time.time()
    response = ollama.chat(
        model=OLLAMA_MODELS[judge],
        messages=messages,
        options={"temperature": 0.0},
    )
    latency = time.time() - start
    return response["message"]["content"], latency


def call_llama_cpp(messages: list[dict]) -> tuple[str, float]:
    llm = _get_lfm2()
    start = time.time()
    out = llm.create_chat_completion(
        messages=messages,
        temperature=0.0,
        max_tokens=256,
    )
    latency = time.time() - start
    return out["choices"][0]["message"]["content"], latency


def expected_calls(records: list[dict], evaluators: list[str]) -> dict[str, int]:
    calls = {}
    for judge in evaluators:
        calls[judge] = len(records)
    return calls


def estimate_cost(
    calls: dict[str, int],
    est_input_tokens_per_call: int,
    est_output_tokens_per_call: int,
) -> dict:
    per_judge = {}
    total = 0.0
    for judge, n_calls in calls.items():
        if judge not in OPENROUTER_PRICING:
            per_judge[judge] = {
                "calls": n_calls,
                "est_input_tokens": n_calls * est_input_tokens_per_call,
                "est_output_tokens": n_calls * est_output_tokens_per_call,
                "est_cost_usd": 0.0,
            }
            continue

        in_tokens = n_calls * est_input_tokens_per_call
        out_tokens = n_calls * est_output_tokens_per_call
        price_in = OPENROUTER_PRICING[judge]["input"]
        price_out = OPENROUTER_PRICING[judge]["output"]
        cost = (in_tokens / 1_000_000.0) * price_in + (out_tokens / 1_000_000.0) * price_out
        per_judge[judge] = {
            "calls": n_calls,
            "est_input_tokens": in_tokens,
            "est_output_tokens": out_tokens,
            "est_cost_usd": round(cost, 6),
        }
        total += cost

    return {
        "pricing_per_1m_tokens_usd": OPENROUTER_PRICING,
        "estimate_input_tokens_per_call": est_input_tokens_per_call,
        "estimate_output_tokens_per_call": est_output_tokens_per_call,
        "per_judge": per_judge,
        "total_est_cost_usd": round(total, 6),
    }


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []


def save_results(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


def run_evaluation(
    records: list[dict],
    evaluators: list[str],
    output_path: Path,
    overwrite: bool,
    max_items: int | None,
    parallel: bool,
) -> None:
    existing = [] if overwrite else load_existing(output_path)
    done = {
        (row.get("evaluator"), row.get("run_id"), row.get("item_index"))
        for row in existing
    }
    results = existing[:]
    io_lock = threading.Lock()

    stats = defaultdict(
        lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
    )

    def evaluate_judge(judge: str) -> None:
        pending = records[:]
        if max_items is not None:
            pending = pending[:max_items]

        print(f"\nEvaluator {judge}: {len(pending)} candidate items")

        for idx, record in enumerate(pending, start=1):
            key = (judge, record["run_id"], record["item_index"])
            with io_lock:
                if key in done:
                    continue
                done.add(key)

            messages = build_messages(record)
            try:
                if judge in OPENROUTER_MODELS:
                    raw_text, latency, usage = call_openrouter(judge, messages)
                elif judge == "lfm2":
                    raw_text, latency = call_llama_cpp(messages)
                    usage = None
                else:
                    raw_text, latency = call_ollama(judge, messages)
                    usage = None

                payload = parse_json_object(raw_text)
                scores, rationale = validate_eval(payload)

                row = {
                    "evaluator": judge,
                    "target_base": record["target_base"],
                    "run_id": record["run_id"],
                    "item_index": record["item_index"],
                    "category": record["category"],
                    "input": record["input"],
                    "candidate_output": record["output"],
                    "scores": scores,
                    "rationale": rationale,
                    "latency": round(latency, 4),
                    "usage": usage,
                    "estimated_call_cost_usd": 0.0,
                }

                if judge in OPENROUTER_PRICING and usage:
                    pt = usage.get("prompt_tokens") or 0
                    ct = usage.get("completion_tokens") or 0
                    price = OPENROUTER_PRICING[judge]
                    call_cost = (pt / 1_000_000.0) * price["input"] + (ct / 1_000_000.0) * price["output"]
                    row["estimated_call_cost_usd"] = round(call_cost, 8)
                with io_lock:
                    if judge in OPENROUTER_PRICING and usage:
                        stats[judge]["prompt_tokens"] += pt
                        stats[judge]["completion_tokens"] += ct
                        stats[judge]["cost_usd"] += call_cost
                    stats[judge]["calls"] += 1
                    results.append(row)
                    save_results(output_path, results)

                if idx % 25 == 0:
                    print(f"  {judge}: processed {idx}/{len(pending)}")

            except Exception as exc:
                with io_lock:
                    done.discard(key)
                print(
                    f"  Error: evaluator={judge} run={record['run_id']} "
                    f"item={record['item_index']} err={exc}"
                )
                continue

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(evaluators)) as ex:
            futures = [ex.submit(evaluate_judge, judge) for judge in evaluators]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        for judge in evaluators:
            evaluate_judge(judge)

    print("\nRun complete.")
    print(f"Saved: {output_path}")

    if stats:
        print("\nOpenRouter usage summary:")
        for judge in evaluators:
            s = stats[judge]
            if s["calls"] == 0:
                continue
            print(
                f"- {judge}: calls={s['calls']}, prompt_tokens={s['prompt_tokens']}, "
                f"completion_tokens={s['completion_tokens']}, est_cost=${s['cost_usd']:.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model peer evaluator for ARES outputs")
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory containing baseline_*.json, taskfocused_*.json, finetuned_*.json",
    )
    parser.add_argument(
        "--output-file",
        default="outputs/cross_model_evaluation.json",
        help="Path to save peer evaluation results",
    )
    parser.add_argument(
        "--evaluators",
        nargs="+",
        default=["grok", "qwen", "smollm3", "lfm2"],
        choices=["grok", "qwen", "smollm3", "lfm2"],
        help="Judge models to run",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only print expected number of calls and estimated cost",
    )
    parser.add_argument(
        "--estimate-input-tokens-per-call",
        type=int,
        default=320,
        help="Token estimate per call for prompt/input side",
    )
    parser.add_argument(
        "--estimate-output-tokens-per-call",
        type=int,
        default=90,
        help="Token estimate per call for completion/output side",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap per evaluator for quick pilot runs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file instead of resuming",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run evaluators sequentially instead of in parallel",
    )

    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    output_path = Path(args.output_file)
    records = load_output_records(outputs_dir)

    if not records:
        raise RuntimeError(f"No output records found in {outputs_dir}")

    calls = expected_calls(records, args.evaluators)
    estimate = estimate_cost(
        calls,
        est_input_tokens_per_call=args.estimate_input_tokens_per_call,
        est_output_tokens_per_call=args.estimate_output_tokens_per_call,
    )

    print("Expected run size:")
    print(json.dumps({"records": len(records), "calls": calls}, indent=2))
    print("\nExpected cost estimate (USD):")
    print(json.dumps(estimate, indent=2))

    if args.estimate_only:
        return

    run_evaluation(
        records=records,
        evaluators=args.evaluators,
        output_path=output_path,
        overwrite=args.overwrite,
        max_items=args.max_items,
        parallel=not args.sequential,
    )


if __name__ == "__main__":
    main()
