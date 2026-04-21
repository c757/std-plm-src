
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PYTHON=/path/to/python LOG_ROOT=./logs ./scripts/run_main_nohup.sh KEY=VALUE ...
# Keys accepted as KEY=VALUE pairs: TASK,MODEL,LLM_LAYERS,BATCH_SIZE,SAMPLE_LEN,PREDICT_LEN,
# TRUNC_K,EPOCH,VAL_EPOCH,USE_GCN(1/0),NODE_EMBEDDING(1/0),FP16(1/0),
# DROPOUT,LR,WEIGHT_DECAY,PATIENCE,FUSION_MODE,LOG_ROOT,PREDICT_VARS,INPUT_LAYOUT,LOG_DIR

# Defaults (can be overridden by environment variables or KEY=VALUE args)
: ${PYTHON:=python}
: ${TASK:=prediction}
: ${MODEL:=transformer}
: ${LLM_LAYERS:=3}
: ${BATCH_SIZE:=1}
: ${SAMPLE_LEN:=9}
: ${PREDICT_LEN:=3}
: ${TRUNC_K:=64}
: ${EPOCH:=10}
: ${VAL_EPOCH:=1}
: ${USE_GCN:=1}
: ${NODE_EMBEDDING:=1}
: ${FP16:=0}
: ${DROPOUT:=0.2}
: ${LR:=1e-4}
: ${WEIGHT_DECAY:=1e-3}
: ${PATIENCE:=5}
: ${FUSION_MODE:=cosine}
: ${LOG_ROOT:=./logs}
: ${PREDICT_VARS:="flow,wind,wave"}
: ${INPUT_LAYOUT:=node}

# Allow overrides passed as KEY=VALUE arguments
for ARG in "$@"; do
  if [[ "$ARG" == *=* ]]; then
    # shellcheck disable=SC2046
    eval "$ARG"
  fi
done

# Work from repo root (script lives in ./scripts)
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

# Create a timestamped run directory unless LOG_DIR explicitly provided
if [ -z "${LOG_DIR:-}" ]; then
  TIMESTR=$(date +"%Y-%m-%d_%H-%M-%S")
  LOG_DIR="${LOG_ROOT}/${TIMESTR}_run"
fi
mkdir -p "$LOG_DIR"

# Build command and flags
CMD=("$PYTHON" main.py)
CMD+=(--task "$TASK")
CMD+=(--model "$MODEL")
CMD+=(--llm_layers "$LLM_LAYERS")
CMD+=(--batch_size "$BATCH_SIZE")
CMD+=(--sample_len "$SAMPLE_LEN")
CMD+=(--predict_len "$PREDICT_LEN")
CMD+=(--trunc_k "$TRUNC_K")
CMD+=(--epoch "$EPOCH")
CMD+=(--val_epoch "$VAL_EPOCH")
if [ "$USE_GCN" = "1" ] || [ "$USE_GCN" = "true" ]; then CMD+=(--use_gcn); fi
if [ "$NODE_EMBEDDING" = "1" ] || [ "$NODE_EMBEDDING" = "true" ]; then CMD+=(--node_embedding); fi
if [ "$FP16" = "1" ] || [ "$FP16" = "true" ]; then CMD+=(--fp16); fi
CMD+=(--dropout "$DROPOUT")
CMD+=(--lr "$LR")
CMD+=(--weight_decay "$WEIGHT_DECAY")
CMD+=(--patience "$PATIENCE")
CMD+=(--fusion_mode "$FUSION_MODE")
CMD+=(--log_root "$LOG_DIR")
CMD+=(--predict_vars "$PREDICT_VARS")
CMD+=(--input_layout "$INPUT_LAYOUT")

# Start with nohup, redirect stdout/stderr to file, save pid
NOHUP_OUT="$LOG_DIR/nohup.out"
echo "Running: ${CMD[*]}" > "$NOHUP_OUT"
nohup "${CMD[@]}" >> "$NOHUP_OUT" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/pid.txt"

echo "Started: PID=$PID"
echo "Logs: $NOHUP_OUT"
echo "Run dir: $LOG_DIR"

# Optionally tail the log (uncomment if you want automatic tailing)
# tail -n 50 -f "$NOHUP_OUT"
