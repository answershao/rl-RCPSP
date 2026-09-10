# rl-RCPSP

单项目**资源受限项目调度（RCPSP）× PPO 强化学习**研究代码库：用 SB3 PPO + 图同构网络（GIN）
逐活动学习调度策略，与**优先规则**（串行/并行 SGS × FIFO/SPT/…/LST/…/WCS）、**GA**（随机键）、
**GPHH**（表达式树超启发式）基线对比。数据为 4430 个公开单项目算例，PSPLIB 主基准以 BKS 精确 gap 度量。

> 数据划分协议、方法口径、端到端流程与重构路线图：见 **[docs/EXECUTION_FLOW.md](docs/EXECUTION_FLOW.md)**；
> 逐步操作手册（S0–S6 命令 / 通过判据 / 排错）：见 **[docs/MAINLINE.md](docs/MAINLINE.md)**。
> 本文档只给「怎么跑、产物放哪」的速查。

## 环境与测试

```bash
conda activate rl                      # py3.12 + torch2.5 + SB3 2.7 + gymnasium 1.2 + numpy2
python -m pytest -q                    # 快测（≈40s，含 SB3 短训）
python -m pytest -q -W error           # R3 约定：无警告级失败
python -m pytest -m slow               # 慢项：4430 全语料解析回归（≈4min）
python -m pytest -m "not slow"         # 仅快测
```

测试体系：`pytest.ini` + 根 `conftest.py`（sys.path）+ `tests/conftest.py`（fixture）；
全语料解析标 `@pytest.mark.slow`。

## 仓库结构

```
src/
  data/          parsers.py（.sm/.rcp 解析）· adapter.py（→core Instance 唯一入口
                 load_core_instance）· instances.py（splits.json 协议 / loader_for）
  core/          rcpsp.py（单项目内核：Instance + 串行/并行 SGS + 校验）
                 rules.py（25 列优先规则）· ga.py · gphh.py · exact.py
  envs/          rcpsp_env.py · multi_instance.py · observation.py · sb3_env.py
  training/      ppo.py · features.py（GIN）· environments.py · callbacks.py
  visualization/ aon.py · gantt.py
  data/generator.py  ProGen 风格生成器（可控 n/RF/RS/NC）
scripts/         common.py（套件表/发现/进程池/CSV 公共件，单一事实源）
                 baselines run_ga run_gphh train_ppo search_ppo
                 bench_ppo（训练吞吐扫描）compare_ppo_results extract_bks
                 aggregate_results visualize_instance
                 instance_stats（套件参数/覆盖度诊断）· generate_pool（生成训练池）
tests/           14 个 test_*.py（65 用例）+ conftest
splits.json      唯一切分协议（生成池 psp_grid_bal）
```

## 复现入口速查

| 步骤 | 命令 | 产物 |
|---|---|---|
| 生成协议/训练池 | `python -m scripts.generate_pool --mode psp-grid --replicates 8,4,2,1 --validation-fraction 0.2 --widen --workers 8 --output data/generated/psp_grid_bal --manifest data/generated/psp_grid_bal.json`（生成后把 manifest 拷为 `splits.json` 即当前协议） | 生成 `.rcp` + `read_protocol` 兼容 manifest，可直接喂 `--splits`；`--replicates` 支持单值或按 n=30/60/90/120 的列表（列表配平决策状态数，单实例成本 ≈n²） |
| 协议诊断（S2，regime 参数源） | `python -m scripts.instance_stats --data-root data --workers 8` | `outputs/instance_stats/{instances,summary,coverage}.csv`（各套件 n/K/RF/RS/NC/OS/CP + 与训练池的参数与特征级覆盖；`instances.csv` 是汇总阶段 `--params` 的输入） |
| 训练池参照规则（S3） | `python -m scripts.baselines --data-root data --instance-workers 8 --output-csv outputs/rules_psp_grid/makespan_summary.csv` | rules CSV，覆盖 train+validation 全部 1216 个生成实例；**PPO 训练的 `--ref-rules` 默认读它，缺了训练直接报错** |
| 规则基线 | `python -m scripts.baselines --data-root data --suites psplib_j30 --instance-workers 8 --seed 17 --output-csv outputs/rules_j30/makespan_summary.csv` | rules CSV（30 列，25 个 makespan 方法列） |
| GA | `python -m scripts.run_ga --data-root data --suites psplib_j30 --instance-workers 8 --seed 17 --output-csv outputs/ga_j30/ga.csv` | ga CSV（`ga_makespan` 等，默认 50×200） |
| GPHH | `python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 60 --seed 17 --eval-suites psplib_j30 --eval-workers 8 --output-dir outputs/gphh_j30/trial1` | `best_rule.txt` + `eval_summary.csv` + `history.csv` + `run_meta.json` |
| PPO 训练 | `bash train_a800.sh`（GPU，含 --eval-all）／`bash train_cpu.sh`（CPU，`EVAL_ALL=1` 时含评估） | `final_model.zip` + `ppo_eval_summary.csv` |
| PPO 训练吞吐扫描 | `python -m scripts.bench_ppo --caps 32 302 --threads 8 16 20 --batch-sizes 512 1024 4096` | 终端表格 / 可选 `--output-csv`（rollout·update·total fps） |
| PPO 评估已有模型 | `bash eval_cpu.sh`（MODEL_DIR 指向 run 目录，评估 splits.json 全部 evaluation 组） | 同上目录追加 `ppo_eval_summary.csv` |
| 跨 seed 搜索 | `bash search_cpu.sh`（模型须放 `outputs/experiments/ppo/seedN/final_model.zip`） | inference_search 结果 |
| 统一汇总出表 | `python -m scripts.aggregate_results --rules … --ga … --gphh … --ppo … --bks data/bks/bks_psplib.json --params outputs/instance_stats/instances.csv --out-dir outputs/aggregate_<scope>` | `merged_detail.csv` + `summary_by_suite.csv` + `summary_by_regime.csv`（gap vs BKS、below-BKS 告警、RF×RS 分档） |
| 单实例可视化 | `python -m scripts.visualize_instance data/psplib/j30/j3010_1.sm` | `outputs/visualizations/j3010_1/{gantt,aon}.png` |
| **一键步骤 2+4** | `bash run_test_baselines.sh`（test=PSPLIB 全量：规则→GA→GPHH→aggregate 合并出表；`WITH_PPO=1` 在 PPO 训完后并入 ppo 列；已存在的阶段输出自动跳过，`ALLOW_OVERWRITE=1` 重算；`SMOKE_MAX_INSTANCES=2` 冒烟） | `outputs/{rules,ga,gphh}_psplib/…` + `outputs/aggregate_psplib/{merged_detail,summary_by_suite}.csv` |

`--suites` 合法 id：`psplib_j30/j60/j90/j120`、`rg30`、`rg300`、`patterson`、`psp_grid`（= 现役训练池，`scripts/common.py::SUITE_SPECS`）。
**默认值就是 `psp_grid`**（`baselines`/`run_ga` 不带 `--suites` 时跑训练池，供 S3 用）；跑 benchmark 请显式指定，如 `--suites psplib_j30,psplib_j60,psplib_j90,psplib_j120`。`run_gphh --eval-suites` 的默认值则是全部评测套件（`EVALUATION_SUITES`，不含任何训练池）。

> 上表按**依赖顺序**排列（S0 自检 → S1 生成协议 → S2 诊断 → S3 训练池参照规则 → S4 基线 → S5 PPO → S6 汇总）。
> **逐步操作手册（含每步通过判据与排错表）见 [docs/MAINLINE.md](docs/MAINLINE.md)**；
> 阶段定义与一键封装范围见 [docs/EXECUTION_FLOW.md](docs/EXECUTION_FLOW.md) §2。

## 输出目录规范

| 产物 | 规范路径 |
|---|---|
| 规则/GA CSV | `outputs/rules_<suite>/…`、`outputs/ga_<suite>/…`（`<suite>` 取主要套件名） |
| GPHH | `outputs/gphh_<suite>/<trial>/…` |
| 统一汇总 | `outputs/aggregate_<scope>/…` |
| PPO 训练 run | `outputs/experiments/ppo/<run>/`（模型、checkpoints、tensorboard、eval CSV） |
| PPO 跨 seed 存档 | `outputs/experiments/ppo/seed<seed>/final_model.zip`（search_ppo 的输入约定） |
| 跨 seed 搜索 | `outputs/experiments/ppo/inference_search/` |
| 可视化 | `outputs/visualizations/<instance_stem>/` |
| 运行日志 | `logs/ppo/*.log`（train/eval/search 均 nohup 后台写此） |

冒烟/调试产物（`outputs/*_smoke`、`outputs/experiments/ppo/<debug_run>/`，含数百 MB 的失效模型）为一次性产物，不保留，可随时清理。

## 关键口径（与 docs/EXECUTION_FLOW.md §1 同源）

- `splits.json` 是**唯一切分清单**（2026-09-10 起为生成池协议）：训练=train
  （1148，PSPLIB 因子网格生成实例，按 n=30/60/90/120 以 8/4/2/1 配平）、
  训练期验证=validation（68，按规模 20% 格子抽取）、
  test=PSPLIB（主测试集，训练期不可见）；rg300/patterson 为可选泛化/补充参考，不进主对比表。
  旧 RG30-only 协议已退役（`data/oras` 的 RG30 数据保留，仅供 `--suites rg30` 基线）。
- **参数覆盖度是可测量的，不是口号**：PSPLIB j30-j120 是 4 RF × (4|5) RS × 3 NC 的因子设计，
  旧 RG30 训练池在参数空间里是一个点（RF 恒 0.75、RS∈[0.003,0.046]、n 恒 30），
  j30-j120 的 RF×RS 联合覆盖只有 0–15%。用 `scripts/instance_stats` 量化、
  `scripts/generate_pool` 在同一网格上生成新实例补齐——现役训练池
  `data/generated/psp_grid_bal` 的 RF×RS 联合覆盖升到 92–100%。
- **归一化口径（2026-09-10 修正）**：静态时长特征用 **per-instance 最大值**作分母
  （`duration/max(duration)`、`downstream/max(downstream)`），不再除以 duration 总和——
  后者 ∝1/n，曾让 j120 的 dur 特征交叠掉到 0.22。修正后 j30-j120 的两个时长特征交叠
  0.97–0.99、demand/capacity 0.85–0.96；跨规模分布齐平，无 n 相关漂移。
  动态时间特征（ready/start/finish 等）仍除以 horizon（安全上界），分子分母同 ∝n，无此问题。
  ⚠️ 特征语义已变，2026-09-10 前的 checkpoint 一律不可用。
- **评估按 regime 分档**：`scripts/aggregate_results.py --params outputs/instance_stats/instances.csv`
  额外产出 `summary_by_regime.csv`（suite × RF 档 × RS 四分位 × 方法）。
  池化均值会掩盖紧/松档差异——j30 上 GA 的 gap 从最紧四分位 1.91% 到最松 0.03%（64 倍）。
- 解析唯一入口 `src.data.adapter.load_core_instance`；实例唯一标识 = 数据根相对路径去扩展名。
- 模型全局 cap：302 活动 / 4 资源 / `MAX_SUCCESSORS=96` + `MAX_PREDECESSORS=96`
  （全语料实测上界 89 / 91，共用 96 余量；j30→RG300 零样本单模型）。
- GIN 消息按邻居数**取均值**（非求和）：求和会按度线性放大，RG300 入度最高 91（j30 仅 3）
  且编码器末端是 ReLU 无抵消，两层后嵌入幅度炸到 1883×；归一化后跨套件极差 1.23×。
  节点自身的入/出度改由 `predecessor_counts` / `successor_counts` 显式特征喂入，
  因此没有丢图结构信息。critic 图池化同理用 `[节点均值 | 节点最大 | global]`，不含随节点数
  增长的求和项。
- **训练图与评估图可以不同**：`train_cpu.sh`/`train_a800.sh` 默认带 `--train-max-activities auto`
  （训练图取训练池最大活动数，当前池 → 122），训练在小图上跑、保存前用 `src.training.ppo.widen_policy`
  把权重迁到 302 图。RCPSP 策略的 30 个可训练参数形状与活动数无关，同一实例下两个图的前向
  数值逐位相同（logits / value 差 0.0），因此只省算力、不改学到的策略。
  ⚠️ **函数等价 ≠ 训练等价**：动作空间是 `Discrete(max_activities)`，`torch.multinomial`
  的 RNG 消耗随类别数变化，随机 rollout 从第 2 个动作起就分叉。所以：
  ① `widen_policy` 前后（同一份权重）可用 `evaluate_paths`（本就是 `deterministic=True`）
  逐实例比对，结果应完全一致；② 但"cap=32 训一次"与"cap=302 训一次"是两次独立训练，
  只能当 A/B 对照，不是复现。设 `TRAIN_MAX_ACTIVITIES=302` 可关闭。
- 训练耗时主要在 PPO 的 update（约 86%），环境采样不是瓶颈；用 `scripts/bench_ppo.py` 在目标
  机器上定 batch/线程，勿直接套用其他机器的结论。
- BKS：`data/bks/bks_psplib.json`（由 RCPLIB xlsx 合成，勿单独引用平凡兜底 UB 列）。
