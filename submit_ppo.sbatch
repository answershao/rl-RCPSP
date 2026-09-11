#!/usr/bin/env bash
# rl-RCPSP 的 Slurm 提交模板（CPU 分区）。
#
# 提交：  sbatch submit_ppo.sbatch
# 多 seed（把下面 --array 打开，并取消注释 SEED 那两行）:
#          sbatch --array=0-2 submit_ppo.sbatch
#
# ── 三个必须知道的点 ────────────────────────────────────────────────
# 1) 必须 `module load gcc/14.1.0`。计算节点自带的 /usr/bin/g++ 是 GCC 8.3.1，
#    torch.compile 的 inductor 后端会构建失败；train_ppo 会**静默**降级成 eager
#    （只打印一行 "g++ is unavailable; disabling torch.compile"），训练照跑但明显更慢。
#    ~/.bashrc 里的 module load 被 `[[ $- == *i* ]]` 门控，批量作业拿不到。
# 2) 必须 `WAIT_FOR_TRAINING=1`。train_cpu.sh 内部是 `nohup python ... &`，
#    不加这个，脚本会立刻返回，Slurm 判定作业结束并杀掉后台训练进程。
# 3) 项目脚本内部调**裸 python**，所以要让 rcmpsp 的 python 排在 PATH 最前。
# ──────────────────────────────────────────────────────────────────
#SBATCH -p cpu64
#SBATCH -J ppo
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH -t 08:00:00
#SBATCH --mem=96G
#SBATCH -o /hpc/home/hpc2023098/rl-RCPSP/logs/slurm/ppo_%A_%a.out
#SBATCH -e /hpc/home/hpc2023098/rl-RCPSP/logs/slurm/ppo_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
cd "$HOME/rl-RCPSP"
mkdir -p logs/slurm

# ── 1. 编译器：torch.compile 的前提 ──────────────────────────────
module load gcc/14.1.0

# ── 2. conda 环境 ───────────────────────────────────────────────
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/rcmpsp"

# ── 3. 自检：任何一项不对就早失败，别浪费几小时 ──────────────────
echo "=== 作业环境 ==="
echo "job       : ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "host      : $(hostname)  cpus=${SLURM_CPUS_PER_TASK}  nproc=$(nproc)"
echo "g++       : $(command -v g++)  $(g++ --version | head -1)"
echo "python    : $(command -v python)  $(python -V 2>&1)"
python - <<'PYEOF'
import torch
torch.compile(lambda t: t.sin().cos(), mode="default", dynamic=True)(torch.zeros(2, 2))
print("torch.compile : OK")
PYEOF

# ── 4. 训练参数（对齐 2026-09-11 的对照实验）──────────────────────
SEEDS=(17 29 41)
export SEED="${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}"
export RUN_DIR="outputs/experiments/ppo/cpu_runs/attn_seed${SEED}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

export N_ENVS=32
export N_STEPS=384
export BATCH_SIZE=1024
export N_EPOCHS=3
export TOTAL_TIMESTEPS=6400000
export TORCH_THREADS=20
export GLOBAL_DIM=16
export TORCH_COMPILE=1
export COMPILE_MODE=default
export WAIT_FOR_TRAINING=1

echo "seed=${SEED}  run_dir=${RUN_DIR}"
echo "n_envs=${N_ENVS} n_steps=${N_STEPS} rollout=$((N_ENVS * N_STEPS)) torch_threads=${TORCH_THREADS}"
echo "CPU 需求估算: n_envs(${N_ENVS}×1) + torch(${TORCH_THREADS}) ≈ $((N_ENVS + TORCH_THREADS))，申请 ${SLURM_CPUS_PER_TASK}"
echo "=================="

./train_cpu.sh
