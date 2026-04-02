#!/usr/bin/env bash
# run_all.sh — Orchestrate all finetuning and evaluation runs for ARES.
#
# Usage:
#   bash run_all.sh [--skip-finetune] [--skip-smollm3-finetune] [--skip-tests] [--only-ui]
#
# Finetuning jobs are run sequentially (single GPU).
# LLM test runs (Grok, Qwen) are parallelised across experiments.
# SLM/finetuned test runs are sequential (GPU memory constraint).
# The web UI is started last.

set -euo pipefail

LOGS_DIR="logs"
mkdir -p "$LOGS_DIR" outputs models

SKIP_FINETUNE=false
SKIP_SMOLLM3_FINETUNE=false
SKIP_TESTS=false
ONLY_UI=false

for arg in "$@"; do
  case $arg in
    --skip-finetune)         SKIP_FINETUNE=true ;;
    --skip-smollm3-finetune) SKIP_SMOLLM3_FINETUNE=true ;;
    --skip-tests)            SKIP_TESTS=true ;;
    --only-ui)               ONLY_UI=true; SKIP_FINETUNE=true; SKIP_TESTS=true ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_bg() {
  # run_bg <log_file_stem> <command...>
  local stem="$1"; shift
  local logfile="$LOGS_DIR/${stem}.log"
  log "Starting background: $* → $logfile"
  "$@" > "$logfile" 2>&1 &
  echo $!
}

run_fg() {
  # run_fg <log_file_stem> <command...>
  local stem="$1"; shift
  local logfile="$LOGS_DIR/${stem}.log"
  log "Starting: $* → $logfile"
  "$@" > "$logfile" 2>&1
  log "Done: $stem"
}

wait_pids() {
  local label="$1"; shift
  log "Waiting for $label to finish…"
  for pid in "$@"; do
    wait "$pid" && log "PID $pid completed OK" || log "WARNING: PID $pid exited with non-zero status"
  done
}

# ─────────────────────────────────────────────
# PHASE 1 — FINETUNING (sequential, single GPU)
# ─────────────────────────────────────────────
if [ "$SKIP_FINETUNE" = false ]; then
  log "=== PHASE 1: Finetuning ==="

  run_fg "finetune_lfm2_misconception"  uv run finetune.py --model lfm2  --task misconception
  run_fg "finetune_lfm2_socratic"       uv run finetune.py --model lfm2  --task socratic

  if [ "$SKIP_SMOLLM3_FINETUNE" = false ]; then
    run_fg "finetune_smollm3_misconception" uv run finetune.py --model smollm3 --task misconception
    run_fg "finetune_smollm3_socratic"      uv run finetune.py --model smollm3 --task socratic
  fi

  log "Finetuning complete."
fi

# ─────────────────────────────────────────────
# PHASE 2 — EVALUATION RUNS
# ─────────────────────────────────────────────
if [ "$SKIP_TESTS" = false ]; then
  log "=== PHASE 2: Evaluation runs ==="

  # --- Baseline tests ---
  log "-- Baseline: LLM runs (parallel) --"
  PID_BL_GROK=$(run_bg "baseline_grok"  uv run test_baseline.py --model grok)
  PID_BL_QWEN=$(run_bg "baseline_qwen"  uv run test_baseline.py --model qwen)

  log "-- Baseline: SLM runs (sequential, ollama) --"
  run_fg "baseline_lfm2"    uv run test_baseline.py --model lfm2
  run_fg "baseline_smollm3" uv run test_baseline.py --model smollm3

  wait_pids "baseline LLM runs" "$PID_BL_GROK" "$PID_BL_QWEN"

  # --- Task-focused tests ---
  log "-- Task-focused: LLM runs (parallel) --"
  PID_TF_GROK=$(run_bg "taskfocused_grok"  uv run test_taskfocused.py --model grok)
  PID_TF_QWEN=$(run_bg "taskfocused_qwen"  uv run test_taskfocused.py --model qwen)

  log "-- Task-focused: SLM runs (sequential, ollama) --"
  run_fg "taskfocused_lfm2"    uv run test_taskfocused.py --model lfm2
  run_fg "taskfocused_smollm3" uv run test_taskfocused.py --model smollm3

  wait_pids "task-focused LLM runs" "$PID_TF_GROK" "$PID_TF_QWEN"

  # --- Finetuned model tests (sequential, GPU) ---
  log "-- Finetuned model tests (sequential) --"
  run_fg "finetuned_lfm2"    uv run test_finetuned.py --model lfm2
  run_fg "finetuned_smollm3" uv run test_finetuned.py --model smollm3

  log "All evaluation runs complete."
fi

# ─────────────────────────────────────────────
# PHASE 3 — WEB UI
# ─────────────────────────────────────────────
log "=== PHASE 3: Starting web UI ==="
log "Evaluation interface available at http://localhost:8000"
log "Press Ctrl+C to stop."
uvicorn app:app --host 0.0.0.0 --port 8000 2>&1 | tee "$LOGS_DIR/webui.log"
