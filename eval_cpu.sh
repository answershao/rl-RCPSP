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

RUNS_ROOT="${RUNS_ROOT:-outputs/experiments/ppo/cpu_runs}"
MODEL_DIR="${MODEL_DIR:-}"
MODEL_FILE="${MODEL_FILE:-checkpoints/best_model.zip}"
DATA_ROOT="${DATA_ROOT:-data}"
SPLITS_PATH="${SPLITS_PATH:-${PROJECT_ROOT}/splits.json}"
TORCH_THREADS="${TORCH_THREADS:-20}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_SUITES="${EVAL_SUITES:-psplib_j30,psplib_j60,psplib_j90,psplib_j120}"
EVAL_REF_RULES="${EVAL_REF_RULES:-outputs/rules_psplib/makespan_summary.csv}"
BKS_PATH="${BKS_PATH:-data/bks/bks_psplib.json}"
if [[ -z "${MODEL_DIR}" ]]; then
    if [[ "${MODEL_FILE}" = /* ]]; then
        MODEL_PATH="${MODEL_FILE}"
        MODEL_DIR="$(dirname "${MODEL_PATH}")"
        if [[ "$(basename "${MODEL_DIR}")" == "checkpoints" ]]; then
            MODEL_DIR="$(dirname "${MODEL_DIR}")"
        fi
    else
        MODEL_DIR="$(python -m scripts.ppo_runs \
            --models-root "${RUNS_ROOT}" \
            --model-file "${MODEL_FILE}" \
            --latest-run-dir)"
        echo "auto-selected latest PPO run: ${MODEL_DIR}"
        MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
    fi
else
    if [[ "${MODEL_FILE}" = /* ]]; then
        MODEL_PATH="${MODEL_FILE}"
    else
        MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
    fi
fi

# train_ppo resolves relative --model-path values against --output-dir.
# Pass an absolute path here so the already-resolved run path is not prefixed
# a second time.
if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "PPO model not found: ${MODEL_PATH}" >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo}"
mkdir -p "${LOG_DIR}"
EVAL_LOG_FILE="${LOG_DIR}/eval_cpu_baseline_$(date +%Y%m%d_%H%M%S).log"

nohup python -m scripts.train_ppo \
    --data-root "${DATA_ROOT}" \
    --evaluate-only \
    --splits "${SPLITS_PATH}" \
    --eval-suites "${EVAL_SUITES}" \
    --device cpu \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads 1 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-ref-rules "${EVAL_REF_RULES}" \
    --bks "${BKS_PATH}" \
    --seed 17 \
    --output-dir "${MODEL_DIR}" \
    --model-path "${MODEL_PATH}" \
    "$@" >"${EVAL_LOG_FILE}" 2>&1 &

EVAL_PID=$!
echo "evaluation started in background: PID=${EVAL_PID}"
echo "log: ${EVAL_LOG_FILE}"
