#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# This host has 52 physical cores across two NUMA nodes. Keep NumPy work in
# each environment single-threaded and reserve CPU capacity for PPO updates.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export PYTHONUNBUFFERED=1

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/experiments/ppo/cpu_runs/cpu_${RUN_STAMP}}"
DATA_ROOT="${DATA_ROOT:-data}"
SPLITS_PATH="${SPLITS_PATH:-${PROJECT_ROOT}/splits.json}"
# Reference-rule makespans for the validation split (checkpoint selection).
# Build it first (docs/EXECUTION_FLOW.md S3):
#   python -m scripts.baselines --data-root data --instance-workers 8 \
#     --output-csv outputs/rules_psp_grid/makespan_summary.csv
REF_RULES="${REF_RULES:-outputs/rules_psp_grid/makespan_summary.csv}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo}"
mkdir -p "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/cpu_${RUN_STAMP}.log"

if [[ -f "${RUN_DIR}/final_model.zip" && "${ALLOW_OVERWRITE_BASELINE:-0}" != "1" ]]; then
    echo "CPU baseline already exists: ${RUN_DIR}" >&2
    echo "Set RUN_DIR to a new experiment directory, or explicitly set ALLOW_OVERWRITE_BASELINE=1." >&2
    exit 2
fi

# Total CPU footprint should stay close to the physical cores (52 on the CPU
# host).  These can be overridden for benchmarking, for example N_ENVS=40.
N_ENVS="${N_ENVS:-32}"
N_STEPS="${N_STEPS:-384}"
# Minibatch size buys optimiser steps, not throughput: update cost scales with
# n_epochs * rollout, and measured 512 vs 1024 minibatches differ by <5%.  The
# default rollout here is 32 * 384 = 12288, so 1024 gives 12 minibatches per
# epoch instead of 3.  Run `python -m scripts.bench_ppo --help` on the target
# host to confirm.
BATCH_SIZE="${BATCH_SIZE:-1024}"
# PPO update time is strictly linear in n_epochs (measured 0.244/0.687/1.168 s
# for 1/3/5 epochs on the 52-core host) while train/approx_kl stays around 1e-6
# per epoch, four orders of magnitude below TARGET_KL -- the early stop never
# fires, so the extra epochs are near-zero-movement repeat passes.
N_EPOCHS="${N_EPOCHS:-3}"
TORCH_THREADS="${TORCH_THREADS:-20}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
ENT_COEF="${ENT_COEF:-0.005}"
VF_COEF="${VF_COEF:-0.5}"
TARGET_KL="${TARGET_KL:-0.02}"
# gamma=1: the return is then exactly -makespan/scale. gamma<1 weights the
# terminal increment by gamma^(n-1), which drifts with instance size
# (0.97 j30 -> 0.74 RG300) and makes a mixed-size pool fit several objectives.
GAMMA="${GAMMA:-1.0}"
GAE_LAMBDA="${GAE_LAMBDA:-0.98}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-25}"
VALIDATION_MIN_DELTA="${VALIDATION_MIN_DELTA:-0}"
CRITICAL_PATH_SHAPING="${CRITICAL_PATH_SHAPING:-0.5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
# torch.compile is the one untested throughput lever on this host (the update
# is dozens of small ops, i.e. launch-bound).  scripts/train_ppo probes the
# inductor backend at startup and falls back to eager if the toolchain is
# incomplete, so enabling it here cannot abort a run.
TORCH_COMPILE="${TORCH_COMPILE:-1}"
# "reduce-overhead" targets CUDA graphs; "default" is the right mode on CPU.
COMPILE_MODE="${COMPILE_MODE:-default}"
SEED="${SEED:-17}"
EVAL_ALL="${EVAL_ALL:-1}"
EVAL_SUITES="${EVAL_SUITES:-psplib_j30,psplib_j60,psplib_j90,psplib_j120}"
EVAL_ARGS=()
if [[ "${EVAL_ALL}" == "1" ]]; then
    EVAL_ARGS=(--eval-suites "${EVAL_SUITES}")
fi

COMPILE_ARGS=()
if [[ "${TORCH_COMPILE}" == "1" ]]; then
    COMPILE_ARGS=(--torch-compile --compile-mode "${COMPILE_MODE}")
fi

nohup python -m scripts.train_ppo \
    --data-root "${DATA_ROOT}" \
    --splits "${SPLITS_PATH}" \
    --ref-rules "${REF_RULES}" \
    --max-activities 122 \
    --max-resources 4 \
    --n-envs "${N_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-epochs "${N_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --ent-coef "${ENT_COEF}" \
    --vf-coef "${VF_COEF}" \
    --critical-path-shaping "${CRITICAL_PATH_SHAPING}" \
    --target-kl "${TARGET_KL}" \
    --gamma "${GAMMA}" \
    --gae-lambda "${GAE_LAMBDA}" \
    --gin-layers 2 \
    --device cpu \
    --mixed-precision none \
    --vec-env subproc \
    --start-method spawn \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads 1 \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --validation-interval "${VALIDATION_INTERVAL}" \
    --validation-min-delta "${VALIDATION_MIN_DELTA}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --seed "${SEED}" \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    ${COMPILE_ARGS[@]+"${COMPILE_ARGS[@]}"} \
    --output-dir "${RUN_DIR}" \
    "$@" >"${TRAIN_LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "training started in background: PID=${TRAIN_PID}"
echo "log: ${TRAIN_LOG_FILE}"
echo "run dir: ${RUN_DIR}"

if [[ "${WAIT_FOR_TRAINING:-0}" == "1" ]]; then
    wait "${TRAIN_PID}"
fi
