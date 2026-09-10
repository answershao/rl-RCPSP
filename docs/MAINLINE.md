# 主线执行手册（S0–S6）

> 定位：**可复制粘贴的操作手册**。每一步给出命令、输入依赖、产物路径、通过判据与常见失败。
> 与之分工：**[docs/EXECUTION_FLOW.md](EXECUTION_FLOW.md)** = 协议口径 / 方法定义 / 重构路线图（"为什么这样做"）；
> 本文 = 操作顺序（"怎么跑、跑完看什么"）。根 `README.md` = 仓库门面与产物目录规范。
> 更新日期：2026-09-10（对应当前协议 `splits.json`：生成池 `psp_grid_bal`，seed 20260910）。

---

## 0. 前置条件

```bash
conda activate rl                     # py3.12 + torch2.5 + SB3 2.7 + gymnasium 1.2 + numpy2
cd /path/to/rl-RCPSP                  # 所有命令都在仓库根执行（脚本用相对路径）
```

已入库、**不需要重跑**的一次性产物：

| 产物 | 来源 | 说明 |
|---|---|---|
| `data/bks/bks_psplib.json` | `python -m scripts.extract_bks` | S6 的 BKS 输入（PSPLIB sheet 160 个 UB 类列取 min 合成） |
| `data/psplib/`（2040 `.sm`） | PSPLIB 原始数据 | 主测试集，训练期不可见 |
| `data/oras/RCPSP/{RG30,RG300,Patterson}` | ORAS 原始数据 | RG30=历史训练池；RG300/Patterson=可选泛化参考 |
| `splits.json` | S1 的 manifest 拷贝 | **唯一切分清单**，所有入口只读它取数 |

当前协议（`splits.json`）实况：

```json
{ "mode": "psp-grid", "seed": 20260910, "replicates": {"30":8,"60":4,"90":2,"120":1},
  "validation_fraction": 0.2, "widen": true, "cells": 336,
  "counts": { "train": 1148, "validation": 68,
              "train_by_size": {"n30":624,"n60":304,"n90":144,"n120":76} },
  "evaluation": { "psplib_j30":480, "psplib_j60":480, "psplib_j90":480,
                  "psplib_j120":600, "rg300":480, "patterson":110 } }
```

---

## 1. 依赖图

```
S0 环境自检 (pytest)
        │
S1 生成训练池 + 协议 ──────────────────────┐
        │  splits.json                     │
        ├───────────────┬──────────────────┤
        ▼               ▼                  ▼
S2 协议诊断        S3 训练池参照规则      S4 test 侧基线
  (--params 源)     (--ref-rules 源)     (rules → GA → GPHH)
        │               │                  │
        │               ▼                  │
        │          S5 PPO 训练 ────────────┤
        │               │ ppo_eval_summary.csv
        ▼               ▼                  ▼
              S6 汇总出表（aggregate --params）
```

- **S1 是切分协议的唯一来源**：S2/S3/S4/S5 全部从它取数，禁止自行扫描目录当切分。
- **S3 是 S5 的硬前置**：`train_ppo --ref-rules` 默认读 S3 产物，缺了直接报错。
- **S2 是 S6 的硬前置**（只影响 `summary_by_regime.csv`）：缺 `--params` 就没有 regime 分档表。
- S4 与 S5 相互独立，可并行。

---

## 2. 阶段详解

### S0 —— 环境自检

```bash
python -m pytest -q -W error            # 期望 65 passed（≈40s 快测）
python -m pytest -m slow                # 可选：4430 全语料解析回归（≈4min）
```

| 项 | 内容 |
|---|---|
| 输入 | — |
| 产物 | — |
| 通过判据 | `65 passed`，且 `-W error` 下无警告级失败 |
| 失败时 | 先修测试再往下走；后续所有阶段的产物都不可信 |

### S1 —— 生成训练池 + 协议

```bash
python -m scripts.generate_pool \
  --mode psp-grid \
  --replicates 8,4,2,1 \
  --validation-fraction 0.2 \
  --widen \
  --workers 8 \
  --output data/generated/psp_grid_bal \
  --manifest data/generated/psp_grid_bal.json

cp data/generated/psp_grid_bal.json splits.json    # manifest 即协议
```

| 项 | 内容 |
|---|---|
| 输入 | `src/data/generator.py` + `scripts/psplib_design.py`（读 RCLIB xlsx 的 `All` sheet 名义因子网格）+ **当前 `splits.json`**（`--splits`，其 `evaluation` 组会被原样搬到新 manifest） |
| 产物 | `data/generated/psp_grid_bal/{n30,n60,n90,n120}/*.rcp`（1216 个）+ `psp_grid_bal.json`（manifest）+ `psp_grid_bal.specs.json`（生成参数留档） |
| 通过判据 | `counts.train=1148`、`counts.validation=68`、`cells=336` |
| 何时跳过 | 协议不变就**不要重跑**（seed 20260910 可复现；重跑会得到同一批实例） |
| 常见失败 | ① `--replicates` 写成 4 个值的列表时必须对应 n=30/60/90/120 升序；② `--widen` 关掉会让基准格子落在训练包络边界上而不是内部点 |

参数含义：`--replicates` 单值=各规模同配额，列表=按规模配平（单实例决策状态成本 ≈n²，故 8/4/2/1 让各规模训练预算接近）；
`--widen` 每个轴向外扩一步，使 PSPLIB 的基准格子成为训练包络的**内部点**；
`--mode random` 是另一条路（坐标均匀采样 = domain randomization），不覆盖 PSPLIB 分布，主线上不用。

### S2 —— 协议诊断（regime 参数源）

```bash
python -m scripts.instance_stats --data-root data --workers 8
```

| 项 | 内容 |
|---|---|
| 输入 | `splits.json` + 数据根 |
| 产物 | `outputs/instance_stats/{instances,summary,coverage}.csv` |
| 通过判据 | j30–j120 的 RF×RS 联合覆盖 92–100%；时长特征交叠 0.97–0.99 |
| 为什么在 S6 之前 | `instances.csv` 是 `aggregate_results --params` 的唯一输入——**跳掉这步就没有 `summary_by_regime.csv`** |
| 常见失败 | `instance_parameters` 必须返回 `OS` 字段，否则写 `summary.csv` 时直接 `KeyError` |

`instances.csv` 逐实例给出 n/K/RF/RS/NC/OS/CP；`coverage.csv` 给出各套件相对训练池的
直方图交叠（`overlap_*` 5 列）与 `joint_inside_pct`。**overlap 高是泛化的必要条件而非充分条件**，
故评估必须按 regime 分档（S6）。

### S3 —— 训练池参照规则

```bash
python -m scripts.baselines --data-root data --instance-workers 8 \
  --output-csv outputs/rules_psp_grid/makespan_summary.csv
```

| 项 | 内容 |
|---|---|
| 输入 | `psp_grid` 套件（`scripts/common.py::SUITE_SPECS` 已登记 → `data/generated/psp_grid_bal`）+ `splits.json` |
| 产物 | rules CSV（30 列，25 个 makespan 方法列），覆盖 train+validation 全部 1216 个生成实例 |
| 通过判据 | 文件存在，且 `file` 列能覆盖 `splits.json` 的 **68 个 validation** 实例 |
| 何时跳过 | 生成池或规则实现不变时可复用 |
| 常见失败 | ① 覆盖不全 → S5 抛 `--ref-rules does not cover N validation instances`；② instance_id 口径不一致（必须是数据根相对路径去扩展名） |

**不需要带 `--suites`**：`DEFAULT_SUITES` 就是 `psp_grid`（`baselines`/`run_ga` 不带 `--suites` 时默认跑训练池）。
参照列默认取 `serial_LST`（S5 可用 `--ref-rule` 改）。

### S4 —— test 侧基线

```bash
SUITES=psplib_j30,psplib_j60,psplib_j90,psplib_j120

python -m scripts.baselines --data-root data --suites "$SUITES" --instance-workers 8 \
  --seed 17 --output-csv outputs/rules_psplib/makespan_summary.csv

python -m scripts.run_ga --data-root data --suites "$SUITES" --instance-workers 8 \
  --seed 17 --population 50 --generations 200 --output-csv outputs/ga_psplib/ga.csv

python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 40 \
  --seed 17 --eval-suites "$SUITES" --eval-workers 8 --output-dir outputs/gphh_psplib/v1
```

| 项 | 内容 |
|---|---|
| 输入 | 数据根 + `--suites`（GPHH 另需 S1 的 train 分片做训练） |
| 产物 | `outputs/rules_psplib/makespan_summary.csv`、`outputs/ga_psplib/ga.csv`、`outputs/gphh_psplib/v1/{best_rule.txt,eval_summary.csv,history.csv,run_meta.json}` |
| 通过判据 | 三个 CSV 的 `file` 列覆盖所选套件全部实例 |
| 耗时 | 规则=秒级；GA=小时级（50×200×实例数，**最慢的一步，别误重跑**）；GPHH 演化慢，默认在一键脚本里关闭 |
| 常见失败 | `--max-instances 0`（默认）才是全量；误传小值会静默只跑前 N 个 |

`run_gphh --eval-suites` 默认值是 `EVALUATION_SUITES`（role=`final-evaluation` 的 6 个套件）——
**任何训练池都不会进评测默认值**，避免"在拟合唱片上评测"。

### S5 —— PPO 训练

```bash
# 先在本机实测最优 batch / 线程（不要再套用别的机器的结论）
python -m scripts.bench_ppo --caps 122 302 --threads 8 16 20 --batch-sizes 512 1024 4096

# CPU 机（后台 nohup；WAIT_FOR_TRAINING=1 可前台等待）
bash train_cpu.sh
# GPU 机
bash train_a800.sh
```

| 项 | 内容 |
|---|---|
| 输入 | **S1（`splits.json`）+ S3（`outputs/rules_psp_grid/makespan_summary.csv`）** |
| 产物 | `outputs/experiments/ppo/<run>/{final_model.zip, checkpoints/, tensorboard/, ppo_eval_summary.csv}`；日志 `logs/ppo/*.log` |
| 通过判据 | `final_model.zip` 存在 + `ppo_eval_summary.csv` 有行；训练日志里 validation 择优有实际改善 |
| 常见失败 | ① 缺 S3 → `FileNotFoundError` 或 "does not cover N validation instances"；② `RUN_DIR` 已存在 → 脚本直接 `exit 2`（设 `RUN_DIR=` 新目录或 `ALLOW_OVERWRITE_BASELINE=1`） |

常用环境变量：`RUN_DIR`（默认 `outputs/experiments/ppo/cpu_runs/cpu_<时间戳>`）、`REF_RULES`、
`TOTAL_TIMESTEPS`（默认 1e7）、`SEED`（默认 17）、`EVAL_ALL=1`（训练后顺带评全部 evaluation 组）、
`N_ENVS/N_STEPS/BATCH_SIZE/N_EPOCHS/TORCH_THREADS`。

关键口径：默认 `TRAIN_MAX_ACTIVITIES=auto` —— 训练图取训练池最大活动数（当前池 → 122），
保存前用 `widen_policy` 把权重迁到全局 cap 302。**函数等价 ≠ 训练等价**：动作空间是
`Discrete(max_activities)`，`multinomial` 的 RNG 消耗随类别数变化，随机 rollout 会分叉；
所以 "cap=122 训一次" 与 "cap=302 训一次" 是两次独立训练（只能当 A/B 对照），
而 `widen_policy` 前后用 `evaluate_paths`（本就是 `deterministic=True`）比对应逐实例一致。

### S6 —— 汇总出表

```bash
python -m scripts.aggregate_results \
  --rules outputs/rules_psplib/makespan_summary.csv \
  --ga    outputs/ga_psplib/ga.csv \
  --gphh  outputs/gphh_psplib/v1/eval_summary.csv \
  --bks   data/bks/bks_psplib.json \
  --params outputs/instance_stats/instances.csv \
  --out-dir outputs/aggregate_psplib
# PPO 训完后加 --ppo outputs/experiments/ppo/<run>/ppo_eval_summary.csv 重跑，并入 ppo_makespan 列
```

| 项 | 内容 |
|---|---|
| 输入 | S2（`--params`）+ S4（rules/ga/gphh），有 PPO 时 + S5 |
| 产物 | `merged_detail.csv`（按 `file` 键合并全部方法）+ `summary_by_suite.csv`（gap vs BKS）+ `summary_by_regime.csv`（suite × RF 档 × RS 四分位 × 方法） |
| 通过判据 | 若出现 below-BKS 告警需人工复核（BKS 是下界，负 gap 意味着口径或数据有问题） |
| 常见失败 | 缺 `--params` → 静默不产出 regime 表；`--rules` 缺 BKS 里的实例 → 该实例 gap 为空 |

---

## 3. 一键封装与旁路

**一键 = S2 + S4 + S6**（不含 S1 / S3 / S5）：

```bash
bash run_test_baselines.sh                                    # 全量，前台
nohup bash run_test_baselines.sh > /dev/null 2>&1 &           # 后台
RUN_GPHH=1 bash run_test_baselines.sh                         # 带 GPHH
SMOKE_MAX_INSTANCES=2 GA_POPULATION=20 GA_GENERATIONS=30 bash run_test_baselines.sh   # 冒烟
WITH_PPO=1 PPO_SUMMARY=outputs/experiments/ppo/<run>/ppo_eval_summary.csv bash run_test_baselines.sh  # 并入 PPO 列
ALLOW_OVERWRITE=1 bash run_test_baselines.sh                  # 重算（默认已存在的阶段输出会跳过）
```

分阶段开关：`RUN_STATS` / `RUN_RULES` / `RUN_GA` / `RUN_GPHH`（默认 0）/ `RUN_AGGREGATE`；
路径覆盖：`OUT_STATS` / `OUT_RULES` / `OUT_GA` / `OUT_GPHH_DIR` / `OUT_AGG`；
规模覆盖：`GA_POPULATION` / `GA_GENERATIONS` / `GPHH_TRAIN_INSTANCES` / `RULES_WORKERS` 等。

**旁路**（不参与主线，按需单跑）：

| 命令 | 用途 | 前置 |
|---|---|---|
| `python -m scripts.extract_bks` | RCLIB xlsx → `data/bks/bks_psplib.json` | —（产物已入库） |
| `bash eval_cpu.sh` | 仅评估已有模型（`MODEL_DIR` 指向 run 目录；默认评 `splits.json` 全部 evaluation 组） | S5 的 `final_model.zip`；**不需要 S3**（读参照规则之前就已 return） |
| `bash search_cpu.sh` | 跨 seed 采样评估（模型须放 `outputs/experiments/ppo/seed<N>/final_model.zip`） | 多 seed 的 S5 + 覆盖所评组的参照规则 |
| `python -m scripts.compare_ppo_results` | 两轮 PPO 的 `ppo_eval_summary.csv` 逐实例对比 | 两次 S5 |
| `python -m scripts.visualize_instance data/psplib/j30/j3010_1.sm` | 单实例 Gantt + AON 图 | — |

---

## 4. 从零到出表（完整命令块）

```bash
conda activate rl

# S0 自检
python -m pytest -q -W error

# S1 训练池 + 协议（协议不变则跳过；manifest 即协议）
python -m scripts.generate_pool --mode psp-grid --replicates 8,4,2,1 \
  --validation-fraction 0.2 --widen --workers 8 \
  --output data/generated/psp_grid_bal --manifest data/generated/psp_grid_bal.json
cp data/generated/psp_grid_bal.json splits.json

# S2 协议诊断（--params 源，缺了没有 regime 表）
python -m scripts.instance_stats --data-root data --workers 8

# S3 训练池参照规则（--ref-rules 源，缺了 S5 起不来）
python -m scripts.baselines --data-root data --instance-workers 8 \
  --output-csv outputs/rules_psp_grid/makespan_summary.csv

# S4 test 基线（rules 秒级，GA 小时级）
SUITES=psplib_j30,psplib_j60,psplib_j90,psplib_j120
python -m scripts.baselines --data-root data --suites "$SUITES" --instance-workers 8 \
  --seed 17 --output-csv outputs/rules_psplib/makespan_summary.csv
python -m scripts.run_ga --data-root data --suites "$SUITES" --instance-workers 8 \
  --seed 17 --population 50 --generations 200 --output-csv outputs/ga_psplib/ga.csv
python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 40 \
  --seed 17 --eval-suites "$SUITES" --eval-workers 8 --output-dir outputs/gphh_psplib/v1

# S5 PPO
python -m scripts.bench_ppo --caps 122 302 --threads 8 16 20 --batch-sizes 512 1024 4096
WAIT_FOR_TRAINING=1 bash train_cpu.sh          # 或 bash train_a800.sh

# S6 汇总（PPO 训完后带上 --ppo 重跑）
python -m scripts.aggregate_results \
  --rules outputs/rules_psplib/makespan_summary.csv \
  --ga    outputs/ga_psplib/ga.csv \
  --gphh  outputs/gphh_psplib/v1/eval_summary.csv \
  --ppo   outputs/experiments/ppo/<run>/ppo_eval_summary.csv \
  --bks   data/bks/bks_psplib.json \
  --params outputs/instance_stats/instances.csv \
  --out-dir outputs/aggregate_psplib
```

---

## 5. 产物路径一览

| 阶段 | 产物 | 规范路径 |
|---|---|---|
| S1 | 生成实例 + 协议 | `data/generated/psp_grid_bal/**/*.rcp`、`data/generated/psp_grid_bal.json`、`splits.json` |
| S2 | 诊断三件套 | `outputs/instance_stats/{instances,summary,coverage}.csv` |
| S3 | 训练池参照规则 | `outputs/rules_psp_grid/makespan_summary.csv` |
| S4 | rules / GA / GPHH | `outputs/rules_<suite>/…`、`outputs/ga_<suite>/…`、`outputs/gphh_<suite>/<trial>/…` |
| S5 | PPO run | `outputs/experiments/ppo/<run>/`（模型 / checkpoints / tensorboard / eval CSV）；日志 `logs/ppo/*.log` |
| S6 | 汇总三件套 | `outputs/aggregate_<scope>/{merged_detail,summary_by_suite,summary_by_regime}.csv` |
| 旁路 | 跨 seed / 可视化 | `outputs/experiments/ppo/{seed<N>,inference_search}/`、`outputs/visualizations/<stem>/` |

`outputs/`、`logs/` 整体不入版本库；冒烟/调试产物（含数百 MB 失效模型）属一次性，不保留。

---

## 6. 排错速查

| 症状 | 根因 | 处置 |
|---|---|---|
| S5 报 `--ref-rules not found` | S3 未跑 | 先跑 S3；或 `REF_RULES=` 指向已有覆盖 CSV |
| S5 报 `does not cover N validation instances` | 参照 CSV 与当前 `splits.json` 的 validation 分片不一致 | 用当前协议重跑 S3（instance_id 必须逐字符对齐） |
| S6 没有 `summary_by_regime.csv` | 未传 `--params` 或 S2 未跑 | 跑 S2，再带 `--params outputs/instance_stats/instances.csv` 重跑 S6 |
| `instance_stats` 写 `summary.csv` 时 `KeyError: 'OS'` | `instance_parameters` 漏返回 `OS` | 修 `scripts/instance_stats.py` 的返回字典 |
| `train_cpu.sh` 立即 `exit 2` | `RUN_DIR` 已存在 `final_model.zip` | 换 `RUN_DIR` 或 `ALLOW_OVERWRITE_BASELINE=1` |
| 基线只跑了几十个实例 | `--max-instances` 传了小值 | 全量用 `--max-instances 0`（默认） |
| GA 结果与之前不同 | 重跑了 GA（小时级）或 seed 变了 | 默认跳过已存在输出；要重算显式 `ALLOW_OVERWRITE=1` |
| 训练吞吐远低于预期 | batch / 线程未按本机调 | 先跑 `bench_ppo` 扫描，再定 `BATCH_SIZE` / `TORCH_THREADS` |
