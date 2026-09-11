# RCPSP x PPO 执行流程

> 本文只维护数据协议、模块边界和阶段依赖。可复制执行的命令、通过判据和排错信息统一放在
> [docs/MAINLINE.md](MAINLINE.md)；仓库概览和常用入口见根目录 [README.md](../README.md)。
> 更新日期：2026-09-10。

## 1. 数据协议

| 数据 | 数量 | 位置 | 作用 |
|---|---:|---|---|
| PSPLIB j30-j120 | 2040 | `data/psplib/` | 训练期不可见的 test，使用 BKS 计算 gap |
| 生成池 `psp_grid_bal` | 1216 | `data/generated/psp_grid_bal/` | PPO/GPHH 的训练池，`train=1088`、`validation=128` |
| BKS | 2040 条 | `data/bks/bks_psplib.json` | PSPLIB 汇总和 gap 计算 |
| 协议清单 | 1 | `splits.json` | 唯一的 train/validation/evaluation 切分来源 |

协议固定为：

- `splits.json` 的顶层分组是 `train`、`validation` 和 `evaluation`。入口不得用目录扫描结果替代切分清单。
- 生成池按 n=30/60/90/120 使用 8/4/2/1 个副本；validation 每种规模 32 个实例，并按 RF/RS/NC 分层。
- PSPLIB evaluation 包含 `psplib_j30`、`psplib_j60`、`psplib_j90`、`psplib_j120`，共 2040 个实例。
- 文件解析统一走 `src.data.parsers.load_instance`，转换到调度内核统一走
  `src.data.adapter.load_core_instance`；稳定实例标识是数据根下的相对路径去扩展名。
- 训练、验证和测试共用 122 活动、4 资源的模型上限；图的最大入度/出度上限分别为 20。
- `data/bks/bks_psplib.json` 是汇总脚本的 BKS 输入。原始 xlsx 仅由 `scripts.extract_bks` 重新生成该 JSON 时使用。

## 2. 阶段依赖

```text
S0 pytest
   |
S1 generate_pool ----+-----------------------+
   |                 |                       |
   +--> S2 stats     +--> S3 rules           +--> S4 test baselines
                         |                       |
                         +----------+------------+
                                    v
                              S5 PPO training
                                    |
                                    v
                              S6 aggregation
```

| 阶段 | 入口 | 前置 | 主要产物 |
|---|---|---|---|
| S0 自检 | `pytest` | — | — |
| S1 生成训练池 | `python -m scripts.generate_pool` | PSPLIB 数据、RCPLIB 参数表 | `.rcp`、manifest、`splits.json` |
| S2 协议诊断 | `python -m scripts.instance_stats` | S1 | `outputs/instance_stats/*.csv` |
| S3 参照规则 | `python -m scripts.baselines`，默认套件为 `psp_grid` | S1 | `outputs/rules_psp_grid/makespan_summary.csv` |
| S4 test 基线 | `baselines`、`run_ga`、可选 `run_gphh` | S1；GPHH 读取 train split | `outputs/rules_*`、`outputs/ga_*`、`outputs/gphh_*` |
| S5 PPO | `train_cpu.sh` 或 `train_a800.sh` | S1 + S3 | 模型、checkpoint、`ppo_eval_summary.csv` |
| S6 汇总 | `python -m scripts.aggregate_results` | S2 + S4；有 PPO 时加 S5 | `merged_detail.csv`、`summary_by_suite.csv`、可选 `summary_by_regime.csv` |

依赖约束：

- S3 的规则 CSV 必须覆盖 `splits.json` 的全部 validation 实例，PPO 才能进行验证集择优。
- S2 生成的 `instances.csv` 是 S6 `--params` 的唯一来源；没有它仍可汇总，但不会生成 regime 表。
- S4 与 S5 可并行，但 S5 不能绕过 S3。
- `run_test_baselines.sh` 只封装 S2 + S4 + S6，不生成训练池、不生成 S3 参照规则，也不训练 PPO。
- S5 的默认关键路径 shaping 只改变 makespan 惩罚的时间分配；其 episode 累计目标仍是
  makespan 的单调变换，`CRITICAL_PATH_SHAPING=0` 可用于无 shaping 对照实验。

## 3. 代码边界

```text
src/data/           解析、格式适配、协议 loader、ProGen 风格生成器
src/core/           Instance、串行/并行 SGS、规则、GA、GPHH
src/envs/           RCPSP Gym 环境、批量环境、观测和 SB3 wrapper
src/training/       GIN 特征提取、PPO、环境工厂、训练 callback
src/visualization/  AON 和 Gantt 图
scripts/            数据准备、基线运行、训练/评估、汇总和诊断入口
tests/              解析、内核、规则、环境、模型和脚本回归测试
```

维护约定：

- 新脚本优先复用 `scripts.common` 的套件表、实例发现、并行执行和 CSV 输出，不复制这些逻辑。
- 调度算法只放在 `src.core.rcpsp`；解析器不实现调度，脚本不维护第二套实例转换逻辑。
- 训练和评估都通过 `src.data.instances.loader_for` 读取协议中的路径，并使用 `instance_id` 作为缓存/合并键。
- 新增公共函数必须有调用方或测试；一次性诊断代码放在脚本入口，不沉入核心模块。

## 4. 旁路入口

| 入口 | 用途 | 前置 |
|---|---|---|
| `python -m scripts.extract_bks` | 从 RCPLIB xlsx 生成 BKS JSON | 原始 xlsx |
| `python -m scripts.bench_ppo` | 扫描目标机器的 PPO batch/线程吞吐 | 可运行的训练环境 |
| `bash eval_cpu.sh` | 评估已有模型；默认自动选择 `cpu_runs` 下最新且包含 `checkpoints/best_model.zip` 的 run，也可用 `MODEL_DIR` 覆盖 | 所选 run 的模型与含 PPO/LST/BKS gap 的 `ppo_eval_summary.csv` |
| `bash search_cpu.sh` | 采样评估；默认自动选择最新 CPU run 的 `checkpoints/best_model.zip`，也兼容 `seed<N>` 布局 | 所选 run 的 `inference_search/` |
| `python -m scripts.compare_ppo_results` | 对比两次 PPO 评估 | 两个评估 CSV |
| `python -m scripts.visualize_instance` | 输出单实例 AON/Gantt 图 | 一个 `.sm` 或 `.rcp` 文件 |

这些入口不会改变主协议。详细参数和示例统一见 [MAINLINE.md](MAINLINE.md)。

## 5. 测试基线

当前测试集共 65 项：64 项快测，1 项 `slow` 全语料解析回归；slow 测试会遍历主协议的 3256 个实例文件。

```bash
python -m pytest -q -m 'not slow'  # 64 passed, 1 deselected
python -m pytest -q                # 全部 65 项
python -m pytest -q -W error       # 全部测试且警告视为失败
```
