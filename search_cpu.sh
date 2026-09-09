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

MODELS_ROOT="${MODELS_ROOT:-outputs/experiments/ppo/makespan_only}"
MODEL_FILE="${MODEL_FILE:-checkpoints/best_model.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODELS_ROOT}/inference_search}"
BASELINE_RESULTS="${BASELINE_RESULTS:-outputs/baselines_mplib2_10_50_5/makespan_summary.csv}"
SEEDS="${SEEDS:-17}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
WORKERS="${WORKERS:-48}"
EVALUATION_SEED="${EVALUATION_SEED:-20260908}"
VALIDATION_ONLY="${VALIDATION_ONLY:-1}"
TORCH_THREADS="${TORCH_THREADS:-1}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"

read -r -a SEED_ARGS <<< "${SEEDS}"

if [[ -z "${MODELS_ROOT}" ]]; then
    echo "set MODELS_ROOT" >&2
    exit 1
fi
for seed in "${SEED_ARGS[@]}"; do
    model_path="${MODELS_ROOT}/seed${seed}/${MODEL_FILE}"
    if [[ ! -f "${model_path}" ]]; then
        echo "PPO model not found: ${model_path}" >&2
        exit 1
    fi
done
if [[ ! -f "${BASELINE_RESULTS}" ]]; then
    echo "baseline results not found: ${BASELINE_RESULTS}" >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo/makespan_only/inference_search}"
mkdir -p "${LOG_DIR}"
SEARCH_LOG_FILE="${LOG_DIR}/search_$(date +%Y%m%d_%H%M%S).log"

SEARCH_SPLIT_ARGS=()
if [[ "${VALIDATION_ONLY}" == "1" ]]; then
    SEARCH_SPLIT_ARGS=(--validation-only)
fi

nohup python -m scripts.search_ppo \
    --models-root "${MODELS_ROOT}" \
    --model-file "${MODEL_FILE}" \
    --seeds "${SEED_ARGS[@]}" \
    --baseline-results "${BASELINE_RESULTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cpu \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads "${TORCH_INTEROP_THREADS}" \
    --evaluation-seed "${EVALUATION_SEED}" \
    "${SEARCH_SPLIT_ARGS[@]}" \
    "$@" >"${SEARCH_LOG_FILE}" 2>&1 &

SEARCH_PID=$!
echo "inference search started in background: PID=${SEARCH_PID}"
echo "log: ${SEARCH_LOG_FILE}"
echo "results: ${OUTPUT_DIR}"

if [[ "${WAIT_FOR_SEARCH:-0}" == "1" ]]; then
    wait "${SEARCH_PID}"
fi
