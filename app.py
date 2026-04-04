"""
app.py

FastAPI blind-evaluation web application for ARES research.

Run with:
    uvicorn app:app --reload --port 8000
"""

import json
import os
import random
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, BaseLoader

load_dotenv()

app = FastAPI(title="ARES Evaluation")

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path("outputs")
EVAL_FILE = OUTPUTS_DIR / "evaluation.json"

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

run_data: dict[str, list[dict]] = {}   # run_id -> list of items
_eval_lock = threading.Lock()
evaluations: list[dict] = []

# ---------------------------------------------------------------------------
# Single unified scoring rubric (all approaches generate Socratic questions)
# ---------------------------------------------------------------------------

PARAMETERS = [
    ("relevance",           "Relevance",           "Directly addresses the student's specific error"),
    ("socratic_quality",    "Socratic Quality",    "Guides discovery rather than giving away the answer"),
    ("clarity",             "Clarity",             "Clear and well-phrased"),
    ("cognitive_challenge", "Cognitive Challenge", "Promotes meaningful reflection"),
    ("naturalness",         "Naturalness",         "Sounds like something a real teacher would ask"),
]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_run_file(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError:
            return []


def load_all_runs() -> None:
    """Scan outputs/ for baseline_*, taskfocused_*, finetuned_* JSON files."""
    run_data.clear()
    patterns = ["baseline_*.json", "taskfocused_*.json", "finetuned_*.json"]
    for pattern in patterns:
        for path in sorted(OUTPUTS_DIR.glob(pattern)):
            run_id = path.stem
            items = load_run_file(path)
            if items:
                run_data[run_id] = items


def load_evaluations() -> None:
    global evaluations
    if EVAL_FILE.exists():
        with open(EVAL_FILE, "r", encoding="utf-8") as fh:
            try:
                evaluations = json.load(fh)
            except json.JSONDecodeError:
                evaluations = []
    else:
        evaluations = []


def save_evaluations() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVAL_FILE, "w", encoding="utf-8") as fh:
        json.dump(evaluations, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

prompt_index: dict[str, list[tuple]] = {}  # input_text -> [(run_id, idx, item), ...]


def build_prompt_index() -> None:
    """Group all outputs by their Input text."""
    prompt_index.clear()
    for run_id, items in run_data.items():
        for idx, item in enumerate(items):
            key = item.get("Input", "")
            if key:
                prompt_index.setdefault(key, []).append((run_id, idx, item))


@app.on_event("startup")
async def startup_event():
    load_all_runs()
    load_evaluations()
    build_prompt_index()


# ---------------------------------------------------------------------------
# Prompt-level lookup (all outputs for the same input)
# ---------------------------------------------------------------------------

def find_unrated_prompt(evaluator_name: str) -> dict | None:
    """Return all outputs for a random prompt not yet rated by this evaluator."""
    rated_inputs = {
        ev.get("input", "")
        for ev in evaluations
        if ev["evaluator"] == evaluator_name
    }

    candidates = [inp for inp in prompt_index if inp not in rated_inputs]
    if not candidates:
        return None

    input_text = random.choice(candidates)
    outputs = prompt_index[input_text]
    shuffled = outputs[:]
    random.shuffle(shuffled)

    ground_truth = shuffled[0][2].get("GroundTruth_Question", "") if shuffled else ""

    return {
        "input": input_text,
        "ground_truth_question": ground_truth,
        "outputs": [
            {
                "run_id": run_id,
                "item_index": idx,
                "output": item.get("Output", ""),
            }
            for run_id, idx, item in shuffled
        ],
    }


def count_rated_prompts(evaluator_name: str) -> int:
    rated_inputs = {
        ev.get("input", "")
        for ev in evaluations
        if ev["evaluator"] == evaluator_name
    }
    return len(rated_inputs)


def total_prompts() -> int:
    return len(prompt_index)


def get_all_progress() -> list[dict]:
    """Return rated prompt count for every known evaluator, sorted by count desc."""
    counts: dict[str, int] = {}
    for ev in evaluations:
        inp = ev.get("input", "")
        name = ev["evaluator"]
        counts.setdefault(name, set()).add(inp)
    total = total_prompts()
    return [
        {"evaluator": name, "rated": len(inps), "total": total}
        for name, inps in sorted(counts.items(), key=lambda x: -len(x[1]))
    ]



# ---------------------------------------------------------------------------
# Inter-rater agreement helpers
# ---------------------------------------------------------------------------

def compute_agreement_stats() -> dict[str, Any]:
    groups: dict[tuple, list[dict]] = {}
    for ev in evaluations:
        key = (ev["run_id"], ev["item_index"])
        groups.setdefault(key, []).append(ev)

    multi = {k: v for k, v in groups.items() if len(v) >= 2}

    total_pairs = 0
    total_agreement = 0.0
    disagreements = []

    for (run_id, item_index), evals in multi.items():
        param_keys = list(evals[0]["scores"].keys())
        max_diff_any = 0

        for a_idx in range(len(evals)):
            for b_idx in range(a_idx + 1, len(evals)):
                a_scores = evals[a_idx]["scores"]
                b_scores = evals[b_idx]["scores"]
                for key in param_keys:
                    diff = abs(a_scores.get(key, 0) - b_scores.get(key, 0))
                    if diff > max_diff_any:
                        max_diff_any = diff
                    agreement = 1.0 - diff / 4.0
                    total_agreement += agreement
                    total_pairs += 1

        if max_diff_any >= 2:
            eval_breakdown = [
                {"evaluator": ev["evaluator"], "scores": ev["scores"]}
                for ev in evals
            ]
            disagreements.append({
                "run_id": run_id,
                "item_index": item_index,
                "input": evals[0].get("input", ""),
                "output": evals[0].get("output", ""),
                "max_diff": max_diff_any,
                "evaluators": eval_breakdown,
            })

    overall_agreement = (total_agreement / total_pairs) if total_pairs > 0 else 1.0
    return {
        "agreement_score": round(overall_agreement, 4),
        "disagreement_count": len(disagreements),
        "disagreements": sorted(disagreements, key=lambda x: -x["max_diff"]),
    }


def compute_run_stats() -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for ev in evaluations:
        rid = ev["run_id"]
        if rid not in stats:
            stats[rid] = {"num_ratings": 0, "score_sums": {}, "evaluators": set()}
        stats[rid]["num_ratings"] += 1
        stats[rid]["evaluators"].add(ev["evaluator"])
        for param, val in ev["scores"].items():
            stats[rid]["score_sums"].setdefault(param, []).append(val)

    result = {}
    for rid, s in stats.items():
        avg_scores = {
            p: round(sum(vals) / len(vals), 2)
            for p, vals in s["score_sums"].items()
        }
        result[rid] = {
            "num_ratings": s["num_ratings"],
            "avg_scores": avg_scores,
            "evaluators": sorted(s["evaluators"]),
        }
    return result


# ---------------------------------------------------------------------------
# Jinja2 setup
# ---------------------------------------------------------------------------

jinja_env = Environment(loader=BaseLoader())

# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

LANDING_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ARES Research Evaluation</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .card { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 2.5rem 3rem; max-width: 420px; width: 100%; }
  h1 { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.4rem; }
  .subtitle { color: #6c757d; font-size: 0.9rem; margin-bottom: 2rem; }
  label { display: block; font-size: 0.88rem; font-weight: 600; margin-bottom: 0.4rem; }
  input[type=text] { width: 100%; padding: 0.55rem 0.75rem; border: 1px solid #ced4da; border-radius: 4px; font-size: 1rem; }
  input[type=text]:focus { outline: 2px solid #333; outline-offset: 1px; }
  button { margin-top: 1.2rem; width: 100%; padding: 0.65rem; background: #333; color: #fff; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; font-weight: 600; }
  button:hover { background: #222; }
</style>
</head>
<body>
<div class="card">
  <h1>ARES Research Evaluation</h1>
  <p class="subtitle">Blind evaluation of AI tutoring model outputs</p>
  <form method="POST" action="/login">
    <label for="name">Your name</label>
    <input type="text" id="name" name="name" placeholder="Enter your name" required autocomplete="off">
    <button type="submit">Start Evaluating</button>
  </form>
</div>
</body>
</html>
"""

EVALUATE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Evaluate - ARES</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; padding: 1.5rem; }
  .top-bar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 0.5rem; }
  .top-bar h1 { font-size: 1.1rem; font-weight: 700; }
  .top-bar .meta { font-size: 0.85rem; color: #6c757d; }
  .top-bar a { font-size: 0.85rem; color: #333; }
  .action-row { margin-bottom: 1.5rem; }
  .action-row button { padding: 0.6rem 1.6rem; background: #333; color: #fff; border: none; border-radius: 4px; font-size: 0.95rem; cursor: pointer; font-weight: 600; }
  .action-row button:hover { background: #222; }
  #prompt-area { display: none; }
  .card { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 1.5rem; margin-bottom: 1rem; }
  .field-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #6c757d; margin-bottom: 0.4rem; }
  .content-text { font-size: 0.97rem; line-height: 1.55; white-space: pre-wrap; }
  .gt-toggle { display: inline-flex; align-items: center; gap: 0.4rem; background: none; border: 1px solid #ced4da; border-radius: 4px; padding: 0.3rem 0.75rem; font-size: 0.82rem; color: #495057; cursor: pointer; margin-top: 0.75rem; font-weight: 600; }
  .gt-toggle:hover { border-color: #333; color: #333; }
  #gt-area { display: none; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed #dee2e6; }
  .outputs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
  .output-box { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 1.25rem; display: flex; flex-direction: column; gap: 1rem; }
  .output-box-header { display: flex; align-items: center; gap: 0.5rem; }
  .box-label { font-size: 1rem; font-weight: 700; background: #333; color: #fff; border-radius: 4px; width: 28px; height: 28px; display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0; }
  .output-text { font-size: 0.93rem; line-height: 1.55; white-space: pre-wrap; color: #212529; }
  .ratings-section { border-top: 1px solid #f1f3f5; padding-top: 1rem; }
  .param-row { margin-bottom: 0.9rem; }
  .param-label { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.15rem; }
  .param-desc { font-size: 0.75rem; color: #6c757d; margin-bottom: 0.4rem; }
  .scale { display: flex; gap: 0.4rem; align-items: center; }
  .scale input[type=radio] { display: none; }
  .scale label { display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border: 1px solid #ced4da; border-radius: 4px; font-size: 0.88rem; font-weight: 600; cursor: pointer; user-select: none; }
  .scale input[type=radio]:checked + label { background: #333; color: #fff; border-color: #333; }
  .scale label:hover { border-color: #555; }
  .scale-wrap { display: inline-block; }
  .scale-ends { display: flex; justify-content: space-between; font-size: 0.72rem; color: #6c757d; margin-top: 0.2rem; }
  .submit-row { margin-top: 0.5rem; }
  #submit-btn { padding: 0.65rem 2.5rem; background: #333; color: #fff; border: none; border-radius: 4px; font-size: 1rem; cursor: pointer; font-weight: 600; }
  #submit-btn:hover { background: #222; }
  #submit-btn:disabled { background: #aaa; cursor: not-allowed; }
  #error-msg { display: none; color: #c0392b; font-size: 0.88rem; margin-top: 0.6rem; }
  #done-msg { display: none; padding: 1.2rem; background: #fff; border: 1px solid #dee2e6; border-radius: 6px; font-size: 0.97rem; }
  .progress-panel { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 1.25rem; }
  .progress-panel h3 { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #6c757d; margin-bottom: 0.75rem; }
  .progress-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.55rem; }
  .progress-row:last-child { margin-bottom: 0; }
  .progress-name { font-size: 0.85rem; font-weight: 600; min-width: 120px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .progress-name.me { color: #333; }
  .progress-track { flex: 1; background: #e9ecef; border-radius: 999px; height: 8px; overflow: hidden; }
  .progress-fill { height: 100%; background: #333; border-radius: 999px; transition: width 0.3s ease; }
  .progress-fill.other { background: #adb5bd; }
  .progress-label { font-size: 0.78rem; color: #6c757d; min-width: 52px; text-align: right; }
  @media (max-width: 500px) { .outputs-grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="top-bar">
  <h1>ARES Evaluation</h1>
  <span class="meta">Evaluator: <strong>{{ evaluator_name }}</strong></span>
  <a href="/analysis">View Analysis</a>
</div>

<div class="progress-panel">
  <h3>Evaluator Progress</h3>
  <div id="progress-rows"><div style="font-size:0.85rem;color:#6c757d;">Loading...</div></div>
</div>

<div class="action-row">
  <button onclick="getPrompt()">Next Prompt</button>
</div>

<div id="prompt-area">
  <div class="card" style="margin-bottom:1rem;">
    <div class="field-label">Input (Student Misconception)</div>
    <div class="content-text" id="input-text"></div>
    <button class="gt-toggle" onclick="toggleGT()" id="gt-btn">Show Ground Truth</button>
    <div id="gt-area">
      <div class="field-label" style="margin-top:0.75rem;">Ground Truth Question</div>
      <div class="content-text" id="gt-text"></div>
    </div>
  </div>

  <div class="field-label" style="margin-bottom:0.6rem;">Rate each output (1 = Poor, 5 = Excellent)</div>
  <div class="outputs-grid" id="outputs-grid"></div>

  <div class="submit-row">
    <button type="button" id="submit-btn" onclick="submitAll()">Submit All Ratings</button>
    <div id="error-msg"></div>
  </div>
</div>

<div id="done-msg">All prompts have been rated. Thank you!</div>

<script>
let currentPrompt = null;
let gtVisible = false;
const ME = "{{ evaluator_name }}";

const PARAMETERS = [
  ["relevance",           "Relevance",           "Directly addresses the student's specific error"],
  ["socratic_quality",    "Socratic Quality",    "Guides discovery rather than giving away the answer"],
  ["clarity",             "Clarity",             "Clear and well-phrased"],
  ["cognitive_challenge", "Cognitive Challenge", "Promotes meaningful reflection"],
  ["naturalness",         "Naturalness",         "Sounds like something a real teacher would ask"],
];

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function loadProgress() {
  const resp = await fetch('/api/progress');
  if (!resp.ok) return;
  const data = await resp.json();
  const total = data.total;
  const rows = data.evaluators;
  if (!rows.find(r => r.evaluator === ME)) rows.push({evaluator: ME, rated: 0, total});
  let html = '';
  for (const row of rows) {
    const pct = total > 0 ? (row.rated / total * 100).toFixed(1) : 0;
    const isMe = row.evaluator === ME;
    html += `<div class="progress-row">
      <div class="progress-name${isMe ? ' me' : ''}">${esc(row.evaluator)}${isMe ? ' (you)' : ''}</div>
      <div class="progress-track"><div class="progress-fill ${isMe ? '' : 'other'}" style="width:${pct}%"></div></div>
      <div class="progress-label">${row.rated}/${total}</div>
    </div>`;
  }
  document.getElementById('progress-rows').innerHTML = html || '<div style="font-size:0.85rem;color:#6c757d;">No ratings yet.</div>';
}

function buildOutputBoxHtml(output, boxIndex) {
  const letter = LETTERS[boxIndex] || String(boxIndex + 1);
  const paramHtml = PARAMETERS.map(([key, label, desc]) => {
    const radios = [1,2,3,4,5].map(n => `
      <span>
        <input type="radio" name="box${boxIndex}_${key}" id="box${boxIndex}_${key}_${n}" value="${n}">
        <label for="box${boxIndex}_${key}_${n}">${n}</label>
      </span>
    `).join('');
    return `<div class="param-row">
      <div class="param-label">${label}</div>
      <div class="param-desc">${desc}</div>
      <div class="scale-wrap">
        <div class="scale">${radios}</div>
        <div class="scale-ends"><span>Poor</span><span>Excellent</span></div>
      </div>
    </div>`;
  }).join('');

  return `<div class="output-box" data-box="${boxIndex}">
    <div class="output-box-header">
      <span class="box-label">${letter}</span>
    </div>
    <div class="output-text">${esc(output.output)}</div>
    <div class="ratings-section">${paramHtml}</div>
  </div>`;
}

function toggleGT() {
  gtVisible = !gtVisible;
  document.getElementById('gt-area').style.display = gtVisible ? 'block' : 'none';
  document.getElementById('gt-btn').textContent = gtVisible ? 'Hide Ground Truth' : 'Show Ground Truth';
}

async function getPrompt() {
  document.getElementById('done-msg').style.display = 'none';
  document.getElementById('prompt-area').style.display = 'none';
  document.getElementById('error-msg').style.display = 'none';
  gtVisible = false;
  document.getElementById('gt-area').style.display = 'none';
  document.getElementById('gt-btn').textContent = 'Show Ground Truth';

  const resp = await fetch('/evaluate/prompt');
  if (resp.status === 404) {
    document.getElementById('done-msg').style.display = 'block';
    return;
  }
  if (!resp.ok) { alert('Error fetching prompt'); return; }
  currentPrompt = await resp.json();

  document.getElementById('input-text').textContent = currentPrompt.input;
  document.getElementById('gt-text').textContent = currentPrompt.ground_truth_question || '(not available)';

  const grid = document.getElementById('outputs-grid');
  grid.innerHTML = currentPrompt.outputs.map((o, i) => buildOutputBoxHtml(o, i)).join('');

  document.getElementById('submit-btn').disabled = false;
  document.getElementById('prompt-area').style.display = 'block';
  window.scrollTo({top: 0, behavior: 'smooth'});
}

async function submitAll() {
  if (!currentPrompt) return;

  const ratings = [];
  let valid = true;
  let firstMissing = null;

  for (let i = 0; i < currentPrompt.outputs.length; i++) {
    const output = currentPrompt.outputs[i];
    const letter = LETTERS[i] || String(i + 1);
    const scores = {};
    for (const [key] of PARAMETERS) {
      const checked = document.querySelector(`input[name="box${i}_${key}"]:checked`);
      if (!checked) {
        valid = false;
        if (!firstMissing) firstMissing = letter;
        break;
      }
      scores[key] = parseInt(checked.value, 10);
    }
    if (!valid) break;
    ratings.push({ run_id: output.run_id, item_index: output.item_index, output: output.output, scores });
  }

  if (!valid) {
    const errEl = document.getElementById('error-msg');
    errEl.textContent = `Please rate all parameters for all outputs (missing ratings in output ${firstMissing}).`;
    errEl.style.display = 'block';
    return;
  }

  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('submit-btn').disabled = true;

  const resp = await fetch('/evaluate/submit-batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: currentPrompt.input, ratings }),
  });

  if (!resp.ok) { alert('Error submitting ratings'); document.getElementById('submit-btn').disabled = false; return; }

  await loadProgress();
  await getPrompt();
}

loadProgress();
</script>
</body>
</html>
"""

ANALYSIS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analysis - ARES</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; padding: 1.5rem; }
  h1 { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem; }
  .links { margin-bottom: 1.5rem; font-size: 0.88rem; }
  .links a { color: #333; margin-right: 1rem; }
  .summary { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
  .stat-box { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 1rem 1.5rem; min-width: 150px; }
  .stat-box .num { font-size: 1.8rem; font-weight: 700; }
  .stat-box .lbl { font-size: 0.8rem; color: #6c757d; margin-top: 0.1rem; }
  h2 { font-size: 1rem; font-weight: 700; margin-bottom: 0.8rem; margin-top: 1.5rem; }
  table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dee2e6; border-radius: 6px; overflow: hidden; font-size: 0.88rem; }
  th { background: #f1f3f5; text-align: left; padding: 0.6rem 0.8rem; font-weight: 600; border-bottom: 1px solid #dee2e6; }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { background: #e9ecef; }
  th .sort-indicator { color: #6c757d; font-size: 0.8em; margin-left: 0.35rem; }
  td { padding: 0.55rem 0.8rem; border-bottom: 1px solid #f1f3f5; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .evaluators { color: #6c757d; font-size: 0.82rem; }
  .cell-max { font-weight: 700; }
  .no-data { padding: 1.5rem; color: #6c757d; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>Analysis</h1>
<div class="links">
  <a href="/evaluate">Back to Evaluate</a>
  <a href="/analysis/disagreements">View Disagreements</a>
  <a href="/api/stats">Raw JSON Stats</a>
</div>

<div class="summary">
  <div class="stat-box"><div class="num" id="total-ratings">-</div><div class="lbl">Total Ratings</div></div>
  <div class="stat-box"><div class="num" id="evaluator-count">-</div><div class="lbl">Evaluators</div></div>
  <div class="stat-box"><div class="num" id="agreement-score">-</div><div class="lbl">Agreement Score</div></div>
  <div class="stat-box"><div class="num" id="disagreement-count">-</div><div class="lbl">Disagreements</div></div>
</div>

<h2>Runs</h2>
<div id="runs-table-area"><div class="no-data">Loading...</div></div>

<script>
let analysisState = {
  runs: [],
  paramList: [],
  sortKey: 'run_id',
  sortDirection: 'asc',
};

async function loadStats() {
  const resp = await fetch('/api/stats');
  if (!resp.ok) return;
  const stats = await resp.json();

  document.getElementById('total-ratings').textContent = stats.total_ratings;
  document.getElementById('evaluator-count').textContent = stats.evaluators.length;
  document.getElementById('agreement-score').textContent = stats.agreement_score.toFixed(2);
  document.getElementById('disagreement-count').textContent = stats.disagreement_count;

  const runs = stats.runs;
  const runIds = Object.keys(runs);
  if (runIds.length === 0) {
    document.getElementById('runs-table-area').innerHTML = '<div class="no-data">No runs have been rated yet.</div>';
    return;
  }

  const allParams = new Set();
  for (const r of Object.values(runs)) Object.keys(r.avg_scores).forEach(p => allParams.add(p));
  analysisState = {
    runs: runIds.map(rid => ({ run_id: rid, ...runs[rid] })),
    paramList: Array.from(allParams),
    sortKey: analysisState.sortKey || 'run_id',
    sortDirection: analysisState.sortDirection || 'asc',
  };
  renderAnalysisTable();
}

function compareValues(a, b, direction) {
  const dir = direction === 'desc' ? -1 : 1;
  const aMissing = a === null || a === undefined || a === '';
  const bMissing = b === null || b === undefined || b === '';
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  if (typeof a === 'number' && typeof b === 'number') return (a - b) * dir;
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' }) * dir;
}

function getCellValue(row, key) {
  if (key === 'run_id') return row.run_id;
  if (key === 'num_ratings') return row.num_ratings;
  if (key === 'evaluators') return row.evaluators.join(', ');
  return row.avg_scores[key];
}

function isNumericColumn(key) {
  return key === 'num_ratings' || analysisState.paramList.includes(key);
}

function renderAnalysisTable() {
  const { runs, paramList, sortKey, sortDirection } = analysisState;
  const sortedRuns = [...runs].sort((a, b) => compareValues(getCellValue(a, sortKey), getCellValue(b, sortKey), sortDirection));

  const numericColumns = ['num_ratings', ...paramList];
  const maxValues = new Map();
  for (const key of numericColumns) {
    let maxValue = null;
    for (const row of sortedRuns) {
      const value = getCellValue(row, key);
      if (typeof value !== 'number' || Number.isNaN(value)) continue;
      if (maxValue === null || value > maxValue) maxValue = value;
    }
    if (maxValue !== null) maxValues.set(key, maxValue);
  }

  const headers = [
    ['run_id', 'Run ID'],
    ['num_ratings', 'Ratings'],
    ['evaluators', 'Evaluators'],
    ...paramList.map(p => [p, p.replace(/_/g, ' ')]),
  ];

  let html = '<table><thead><tr>';
  for (const [key, label] of headers) {
    const isActive = sortKey === key;
    const indicator = isActive ? (sortDirection === 'asc' ? '▲' : '▼') : '';
    html += `<th class="sortable" data-sort-key="${key}">${label}<span class="sort-indicator">${indicator}</span></th>`;
  }
  html += '</tr></thead><tbody>';

  for (const row of sortedRuns) {
    html += `<tr><td><strong>${row.run_id}</strong></td>`;
    html += `<td${maxValues.get('num_ratings') === row.num_ratings ? ' class="cell-max"' : ''}>${row.num_ratings}</td>`;
    html += `<td class="evaluators">${row.evaluators.join(', ')}</td>`;
    for (const p of paramList) {
      const value = row.avg_scores[p];
      const isMax = typeof value === 'number' && maxValues.get(p) === value;
      html += `<td${isMax ? ' class="cell-max"' : ''}>${value !== undefined ? value.toFixed(2) : '-'}</td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';

  const area = document.getElementById('runs-table-area');
  area.innerHTML = html;
  area.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.sortKey;
      if (analysisState.sortKey === key) {
        analysisState.sortDirection = analysisState.sortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        analysisState.sortKey = key;
        analysisState.sortDirection = isNumericColumn(key) ? 'desc' : 'asc';
      }
      renderAnalysisTable();
    });
  });
}
loadStats();
</script>
</body>
</html>
"""

DISAGREEMENTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Disagreements - ARES</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; padding: 1.5rem; }
  h1 { font-size: 1.3rem; font-weight: 700; margin-bottom: 0.3rem; }
  .links { margin-bottom: 1.5rem; font-size: 0.88rem; }
  .links a { color: #333; margin-right: 1rem; }
  .item-card { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 1.2rem 1.5rem; margin-bottom: 1.2rem; }
  .run-label { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #6c757d; margin-bottom: 0.5rem; }
  .diff-badge { display: inline-block; background: #dee2e6; border-radius: 3px; padding: 0.1rem 0.4rem; font-size: 0.78rem; font-weight: 700; margin-left: 0.5rem; }
  .text-block { font-size: 0.9rem; margin-bottom: 0.8rem; }
  .text-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #6c757d; margin-bottom: 0.2rem; }
  .text-content { white-space: pre-wrap; line-height: 1.5; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.8rem; }
  th { background: #f1f3f5; text-align: left; padding: 0.45rem 0.6rem; font-weight: 600; border-bottom: 1px solid #dee2e6; }
  td { padding: 0.4rem 0.6rem; border-bottom: 1px solid #f1f3f5; }
  tr:last-child td { border-bottom: none; }
  .no-data { color: #6c757d; font-size: 0.9rem; padding: 1rem 0; }
</style>
</head>
<body>
<h1>Disagreements</h1>
<div class="links">
  <a href="/analysis">Back to Analysis</a>
  <a href="/evaluate">Evaluate</a>
</div>
<p style="font-size:0.88rem;color:#6c757d;margin-bottom:1.2rem;">Items where evaluators differ by 2+ on any parameter.</p>
<div id="content"><p class="no-data">Loading...</p></div>

<script>
async function loadDisagreements() {
  const resp = await fetch('/api/stats');
  if (!resp.ok) return;
  const stats = await resp.json();
  const items = stats.disagreements || [];

  if (items.length === 0) {
    document.getElementById('content').innerHTML = '<p class="no-data">No disagreements found yet.</p>';
    return;
  }

  let html = '';
  for (const item of items) {
    html += `<div class="item-card">`;
    html += `<div class="run-label">Run: ${item.run_id} &nbsp;|&nbsp; Item #${item.item_index}<span class="diff-badge">max diff: ${item.max_diff}</span></div>`;
    html += `<div class="text-block"><div class="text-label">Input</div><div class="text-content">${esc(item.input)}</div></div>`;
    html += `<div class="text-block"><div class="text-label">Output</div><div class="text-content">${esc(item.output)}</div></div>`;
    const paramKeys = item.evaluators.length > 0 ? Object.keys(item.evaluators[0].scores) : [];
    if (paramKeys.length > 0) {
      html += '<table><thead><tr><th>Evaluator</th>';
      for (const p of paramKeys) html += `<th>${p.replace(/_/g,' ')}</th>`;
      html += '</tr></thead><tbody>';
      for (const ev of item.evaluators) {
        html += `<tr><td><strong>${esc(ev.evaluator)}</strong></td>`;
        for (const p of paramKeys) html += `<td>${ev.scores[p] !== undefined ? ev.scores[p] : '-'}</td>`;
        html += '</tr>';
      }
      html += '</tbody></table>';
    }
    html += '</div>';
  }
  document.getElementById('content').innerHTML = html;
}

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadDisagreements();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def landing():
    return HTMLResponse(content=LANDING_HTML)


@app.post("/login")
async def login(name: str = Form(...)):
    name = name.strip()
    if not name:
        return RedirectResponse(url="/", status_code=303)
    response = RedirectResponse(url="/evaluate", status_code=303)
    response.set_cookie(key="evaluator_name", value=name, httponly=True)
    return response


@app.get("/evaluate", response_class=HTMLResponse)
async def evaluate_page(request: Request):
    name = request.cookies.get("evaluator_name")
    if not name:
        return RedirectResponse(url="/", status_code=303)
    tmpl = jinja_env.from_string(EVALUATE_HTML)
    html = tmpl.render(evaluator_name=name)
    return HTMLResponse(content=html)


@app.get("/evaluate/prompt")
async def get_prompt(request: Request):
    name = request.cookies.get("evaluator_name")
    if not name:
        raise HTTPException(status_code=401, detail="Not logged in")

    prompt = find_unrated_prompt(name)
    if prompt is None:
        raise HTTPException(status_code=404, detail="No more prompts")
    return JSONResponse(content=prompt)


@app.post("/evaluate/submit-batch")
async def submit_batch(request: Request):
    name = request.cookies.get("evaluator_name")
    if not name:
        raise HTTPException(status_code=401, detail="Not logged in")

    body = await request.json()
    input_text = body.get("input", "")
    ratings = body.get("ratings", [])

    if not ratings:
        raise HTTPException(status_code=400, detail="No ratings provided")

    records = []
    for r in ratings:
        run_id = r.get("run_id")
        item_index = r.get("item_index")
        scores = r.get("scores", {})
        if run_id is None or item_index is None:
            continue
        records.append({
            "id": str(uuid.uuid4()),
            "evaluator": name,
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "item_index": item_index,
            "input": input_text,
            "output": r.get("output", ""),
            "scores": scores,
        })

    with _eval_lock:
        evaluations.extend(records)
        save_evaluations()

    return JSONResponse(content={"status": "ok", "rated_prompts": count_rated_prompts(name)})


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    name = request.cookies.get("evaluator_name")
    if not name:
        return RedirectResponse(url="/", status_code=303)
    return HTMLResponse(content=ANALYSIS_HTML)


@app.get("/analysis/disagreements", response_class=HTMLResponse)
async def disagreements_page(request: Request):
    name = request.cookies.get("evaluator_name")
    if not name:
        return RedirectResponse(url="/", status_code=303)
    return HTMLResponse(content=DISAGREEMENTS_HTML)


@app.get("/api/progress")
async def api_progress():
    return JSONResponse(content={
        "total": total_prompts(),
        "evaluators": get_all_progress(),
    })


@app.get("/api/stats")
async def api_stats():
    run_stats = compute_run_stats()
    agreement = compute_agreement_stats()
    all_evaluators = sorted({ev["evaluator"] for ev in evaluations})
    return JSONResponse(content={
        "total_ratings": len(evaluations),
        "evaluators": all_evaluators,
        "runs": run_stats,
        "agreement_score": agreement["agreement_score"],
        "disagreement_count": agreement["disagreement_count"],
        "disagreements": agreement["disagreements"],
    })
