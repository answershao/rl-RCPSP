# rl-RCPSP

单项目**资源受限项目调度（RCPSP）× PPO 强化学习**研究代码库：用 SB3 PPO + 图同构网络（GIN）
逐活动学习调度策略，与**优先规则**（串行/并行 SGS × FIFO/SPT/…/LST/…/WCS）、**GA**（随机键）、
**GPHH**（表达式树超启发式）基线对比。数据为 4430 个公开单项目算例，PSPLIB 主基准以 BKS 精确 gap 度量。

> 数据划分协议、方法口径、端到端流程与重构路线图：见 **[docs/EXECUTION_FLOW.md](docs/EXECUTION_FLOW.md)**。
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
scripts/         common.py（套件表/发现/进程池/CSV 公共件，单一事实源）
                 make_splits baselines run_ga run_gphh train_ppo search_ppo
                 compare_ppo_results extract_bks aggregate_results visualize_instance
tests/           12 个 pytest 文件（35 用例）
splits.json      唯一切分协议（seed 20260909，勿手改，用 scripts/make_splits.py 重新生成）
```

## 复现入口速查

| 步骤 | 命令 | 产物 |
|---|---|---|
| （重新）生成协议 | `python -m scripts.make_splits --data-root data --seed 20260909 --val-fraction 0.1 --output splits.json` | `splits.json`（当前版本逐字节一致） |
| 规则基线 | `python -m scripts.baselines --data-root data --suites psplib_j30 --instance-workers 8 --seed 17 --output-csv outputs/rules_j30/makespan_summary.csv` | rules CSV（30 列，25 个 makespan 方法列） |
| GA | `python -m scripts.run_ga --data-root data --suites psplib_j30 --instance-workers 8 --seed 17 --output-csv outputs/ga_j30/ga.csv` | ga CSV（`ga_makespan` 等，默认 50×200） |
| GPHH | `python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 60 --seed 17 --eval-suites psplib_j30 --eval-workers 8 --output-dir outputs/gphh_j30/trial1` | `best_rule.txt` + `eval_summary.csv` + `history.csv` + `run_meta.json` |
| PPO 训练 | `bash train_a800.sh`（GPU，含 --eval-all）／`bash train_cpu.sh`（CPU，`EVAL_ALL=1` 时含评估） | `final_model.zip` + `ppo_eval_summary.csv` |
| PPO 评估已有模型 | `bash eval_cpu.sh`（MODEL_DIR 指向 run 目录，评估 splits.json 全部 evaluation 组） | 同上目录追加 `ppo_eval_summary.csv` |
| 跨 seed 搜索 | `bash search_cpu.sh`（模型须放 `outputs/experiments/ppo/seedN/final_model.zip`） | inference_search 结果 |
| 统一汇总出表 | `python -m scripts.aggregate_results --rules … --ga … --gphh … --ppo … --bks data/bks/bks_psplib.json --out-dir outputs/aggregate_<scope>` | `merged_detail.csv` + `summary_by_suite.csv`（gap vs BKS、below-BKS 告警） |
| 单实例可视化 | `python -m scripts.visualize_instance data/psplib/j30/j3010_1.sm` | `outputs/visualizations/j3010_1/{gantt,aon}.png` |

`--suites` 合法 id：`psplib_j30/j60/j90/j120`、`rg30`、`rg300`、`patterson`（`scripts/common.py::SUITE_SPECS`）。

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

历史遗留的 `outputs/*_smoke` 为临时冒烟产物，可随时清理。

## 关键口径（与 docs/EXECUTION_FLOW.md §1 同源）

- `splits.json` 是**唯一切分清单**：训练=rg30_train(1620)、训练期验证=rg30_validation(180)、
  评估=evaluation 各组；PSPLIB 全体训练期不可见。
- 解析唯一入口 `src.data.adapter.load_core_instance`；实例唯一标识 = 数据根相对路径去扩展名。
- 模型全局 cap：302 活动 / 4 资源 / MAX_SUCCESSORS=96（j30→RG300 零样本单模型）。
- BKS：`data/bks/bks_psplib.json`（由 RCPLIB xlsx 合成，勿单独引用平凡兜底 UB 列）。
