#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export PYTHONUNBUFFERED=1

MODEL_DIR="${MODEL_DIR:-outputs/experiments/ppo/cpu_baseline}"
DATA_ROOT="${DATA_ROOT:-data}"
SPLITS_PATH="${SPLITS_PATH:-${PROJECT_ROOT}/splits.json}"
TORCH_THREADS="${TORCH_THREADS:-20}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MODEL_PATH="${MODEL_DIR}/final_model.zip"

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "PPO model not found: ${MODEL_PATH}" >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo}"
mkdir -p "${LOG_DIR}"
EVAL_LOG_FILE="${LOG_DIR}/eval_cpu_baseline_$(date +%Y%m%d_%H%M%S).log"

# --evaluate-only with no --eval-suites evaluates every evaluation group in
# splits.json (PSPLIB j30-j120 / RG300 / Patterson), the final-model protocol.
nohup python -m scripts.train_ppo \
    --data-root "${DATA_ROOT}" \
    --evaluate-only \
    --splits "${SPLITS_PATH}" \
    --device cpu \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads 1 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --seed 17 \
    --output-dir "${MODEL_DIR}" \
    "$@" >"${EVAL_LOG_FILE}" 2>&1 &

EVAL_PID=$!
echo "evaluation started in background: PID=${EVAL_PID}"
echo "log: ${EVAL_LOG_FILE}"
