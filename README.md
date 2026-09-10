# rl-RCPSP

单项目**资源受限项目调度（RCPSP）× PPO 强化学习**研究代码库：用 SB3 PPO + 图同构网络（GIN）
逐活动学习调度策略，与**优先规则**（串行/并行 SGS × FIFO/SPT/…/LST/…/WCS）、**GA**（随机键）、
**GPHH**（表达式树超启发式）基线对比。主协议包含 1216 个 generated 训练/验证实例和 2040 个 PSPLIB 测试实例，以 BKS 精确 gap 度量。

> 数据划分协议、模块边界和端到端流程：见 **[docs/EXECUTION_FLOW.md](docs/EXECUTION_FLOW.md)**；
> 逐步操作手册（S0–S6 命令 / 通过判据 / 排错）：见 **[docs/MAINLINE.md](docs/MAINLINE.md)**。
> 本文档只给「怎么跑、产物放哪」的速查。

## 环境与测试

```bash
conda activate rl                      # py3.12 + torch2.5 + SB3 2.7 + gymnasium 1.2 + numpy2
python -m pytest -q -m 'not slow'      # 快测（含 SB3 短训）
python -m pytest -q                    # 全量（含语料解析回归）
python -m pytest -q -W error           # 全量且警告级失败
python -m pytest -m slow               # 慢项：3256 个主协议实例解析回归
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
                 rules.py（25 列优先规则）· ga.py · gphh.py
  envs/          rcpsp_env.py · multi_instance.py · observation.py · sb3_env.py
  training/      ppo.py · features.py（GIN）· environments.py · callbacks.py
  visualization/ aon.py · gantt.py
  data/generator.py  ProGen 风格生成器（可控 n/RF/RS/NC）
scripts/         common.py（套件表/发现/进程池/CSV 公共件，单一事实源）
                 baselines run_ga run_gphh train_ppo search_ppo
                 bench_ppo（训练吞吐扫描）compare_ppo_results extract_bks
                 aggregate_results visualize_instance
                 instance_stats（套件参数/覆盖度诊断）· generate_pool（生成训练池）
tests/           13 个 test_*.py（65 用例）+ conftest
splits.json      唯一切分协议（生成池 psp_grid_bal）
```

## 复现入口速查

| 步骤 | 命令 | 产物 |
|---|---|---|
| 生成协议/训练池 | `python -m scripts.generate_pool --mode psp-grid --replicates 8,4,2,1 --validation-per-size 32 --widen --workers 8 --output data/generated/psp_grid_bal --manifest data/generated/psp_grid_bal.json`（生成后把 manifest 拷为 `splits.json` 即当前协议） | 生成 `.rcp` + `read_protocol` 兼容 manifest，可直接喂 `--splits`；`--replicates` 支持单值或按 n=30/60/90/120 的列表（列表配平决策状态数，单实例成本 ≈n²）。⚠️ 实例 bytes 可复现（实测 1216/1216 一致）但**当前 validation 划分不可复现**（108/128），切分以 git 里的 `splits.json` 为准；协议一改，`psp_grid_bal.json` / `.specs.json` 两个派生文件必须同步重生，见 S1 判据 |
| 协议诊断（S2，regime 参数源） | `python -m scripts.instance_stats --data-root data --workers 8` | `outputs/instance_stats/{instances,summary,coverage}.csv`（各套件 n/K/RF/RS/NC/OS/CP + 与训练池的参数与特征级覆盖；`instances.csv` 是汇总阶段 `--params` 的输入） |
| 训练池参照规则（S3） | `python -m scripts.baselines --data-root data --instance-workers 8 --output-csv outputs/rules_psp_grid/makespan_summary.csv` | rules CSV，覆盖 train+validation 全部 1216 个生成实例；**PPO 训练的 `--ref-rules` 默认读它，缺了训练直接报错** |
| 规则基线 | `python -m scripts.baselines --data-root data --suites psplib_j30 --instance-workers 8 --seed 17 --output-csv outputs/rules_j30/makespan_summary.csv` | rules CSV（30 列，25 个 makespan 方法列） |
| GA | `python -m scripts.run_ga --data-root data --suites psplib_j30 --instance-workers 8 --seed 17 --output-csv outputs/ga_j30/ga.csv` | ga CSV（`ga_makespan` 等，默认 50×200） |
| GPHH | `python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 60 --seed 17 --eval-suites psplib_j30 --eval-workers 8 --output-dir outputs/gphh_j30/trial1` | `best_rule.txt` + `eval_summary.csv` + `history.csv` + `run_meta.json` |
| PPO 训练 | `bash train_a800.sh`（GPU）／`bash train_cpu.sh`（CPU；默认训后评测） | `final_model.zip` + `ppo_eval_summary.csv` |
| PPO 训练吞吐扫描 | `python -m scripts.bench_ppo --caps 122 --threads 8 16 20 --batch-sizes 512 1024 4096` | 终端表格 / 可选 `--output-csv`（rollout·update·total fps） |
| PPO 评估已有模型 | `bash eval_cpu.sh`（MODEL_DIR 指向 run 目录，评估 PSPLIB j30-j120） | 同上目录追加 `ppo_eval_summary.csv` |
| 跨 seed 搜索 | `bash search_cpu.sh`（模型须放 `outputs/experiments/ppo/seedN/final_model.zip`） | inference_search 结果 |
| 统一汇总出表 | `python -m scripts.aggregate_results --rules … --ga … --gphh … --ppo … --bks data/bks/bks_psplib.json --params outputs/instance_stats/instances.csv --out-dir outputs/aggregate_<scope>` | `merged_detail.csv` + `summary_by_suite.csv` + `summary_by_regime.csv`（gap vs BKS、below-BKS 告警、RF×RS 分档） |
| 单实例可视化 | `python -m scripts.visualize_instance data/psplib/j30/j3010_1.sm` | `outputs/visualizations/j3010_1/{gantt,aon}.png` |
| **一键 S2+S4+S6** | `bash run_test_baselines.sh`（test=PSPLIB 全量：规则→GA→可选 GPHH→aggregate；`WITH_PPO=1` 在 PPO 训完后并入 ppo 列；已存在的阶段输出自动跳过，`ALLOW_OVERWRITE=1` 重算；`SMOKE_MAX_INSTANCES=2` 冒烟） | `outputs/{rules,ga,gphh}_psplib/…` + `outputs/aggregate_psplib/{merged_detail,summary_by_suite}.csv` |

`--suites` 合法 id：`psplib_j30/j60/j90/j120`、`psp_grid`（= 训练池，`scripts/common.py::SUITE_SPECS`）。
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
  （1088，PSPLIB 因子网格生成实例，按 n=30/60/90/120 以 8/4/2/1 生成）、
  训练期验证=validation（128，每种规模各 32 个 RF/RS/NC 分层单元）、
  test=PSPLIB j30/j60/j90/j120（训练期不可见）。
- **参数覆盖度是可测量的，不是口号**：PSPLIB j30-j120 是 4 RF × (4|5) RS × 3 NC 的因子设计，
  用 `scripts/instance_stats` 量化、`scripts/generate_pool` 在同一网格上生成训练池，
  `data/generated/psp_grid_bal` 的 RF×RS 联合覆盖升到 92–100%。
- **归一化口径（2026-09-10 修正）**：静态时长特征用 **per-instance 最大值**作分母
  （`duration/max(duration)`、`downstream/max(downstream)`），不再除以 duration 总和——
  后者 ∝1/n，曾让 j120 的 dur 特征交叠掉到 0.22。修正后 j30-j120 的两个时长特征交叠
  0.97–0.99、demand/capacity 0.85–0.96；跨规模分布齐平，无 n 相关漂移。
- **时间型特征改用实例级 `time_scale`（2026-09-10 二次修正）**：动态活动特征的 6 维、
  全局 `current_time`、以及 `resource_profile` 的 16 个 bin，全部改除以
  `RCPSPEnv.time_scale`（= LST 优先规则跑一条串行 SGS 的 makespan，每实例算一次；
  j30 0.5ms / j120 3.5ms / RG300 60ms），不再除以 horizon(=Σd)。Σd 是安全上界但比真实
  makespan 大 2.3–5.3 倍（j120 实测 makespan/Σd=0.23），旧口径下 128 个 profile 通道里
  124 个恒为 0（现在 16/16 箱非零）。`time_scale` 是**参照不是上界**（随机策略实测 10/12
  实例越界），越界特征被 clip 到 1.0，profile 末位 bin 加宽兜住溢出区。
  **例外**：`makespan_increment` 除以实例内 **max(duration)** 而非 `time_scale`——
  插入代价的上界就是最大单个活动，这样天花板恒为 1.0、跨套件可比（除 `time_scale` 时
  天花板只有 0.09–0.20）；也不能除自身时长，那会塌成 {0,1}（无延迟时恰等于自身时长）。
- **reward 与 episode 指标同样除以 `time_scale`**（不再是 Σd）：γ=1 下回报精确等于
  `-makespan/time_scale` ∈ 约 [−2, −0.9]，value target 落在 O(1)；除 Σd 时它是个近乎
  常数（≈0.3）的弱信号，实测 `explained_variance` 在 +0.55/−1.60 间摆动。
  ⚠️ 特征语义 + 回报口径都变过，2026-09-10 二次修正前的 checkpoint 一并作废。
- **静态 LST 特征（2026-09-10 新增）**：`StaticGraphCache` 增加 `slack_ratios`
  （=(LST−EST)/CP，用**关键路径长**而非 Σd 当deadline，故 ∈[0,1] 且尺度无关）与
  `on_critical_path`（slack==0 的 0/1 位）。两者都是资源无关的 CPM 量，静态缓存一次算好，
  经 `activity_input_dim` 7→9 喂进编码器。LST 是 RCPSP 信息量最大的单一结构特征
  （MSLK/WCS/GPHH 终端都建立在它上面）。
- **评估按 regime 分档**：`scripts/aggregate_results.py --params outputs/instance_stats/instances.csv`
  额外产出 `summary_by_regime.csv`（suite × RF 档 × RS 四分位 × 方法）。
  池化均值会掩盖紧/松档差异——j30 上 GA 的 gap 从最紧四分位 1.91% 到最松 0.03%（64 倍）。
- 解析唯一入口 `src.data.adapter.load_core_instance`；实例唯一标识 = 数据根相对路径去扩展名。
- 模型全局 cap：122 活动 / 4 资源 / `MAX_SUCCESSORS=20` + `MAX_PREDECESSORS=20`
  （generated + PSPLIB 全语料实测最大出度 19、最大入度 18）。
- GIN 消息按邻居数**取均值**（非求和），避免节点嵌入随图度数放大。
  节点自身的入/出度改由 `predecessor_counts` / `successor_counts` 显式特征喂入，
  因此没有丢图结构信息。critic 图池化同理用 `[节点均值 | 节点最大 | global]`，不含随节点数
  增长的求和项。每层 GIN 用 **`LayerNorm(x + update(...))`**（2026-09-10 新增）：均值聚合
  只压掉度数因子，残差流没有归一的话深度与度数仍会累积（实测 6 层后激活 max 从 2.4
  涨到 142），LayerNorm 后同一套权重才可跨套件迁移。
- **无效动作是零奖励 + 步数预算，不再是惩罚项**（2026-09-10 改）：`INVALID_ACTION_PENALTY`
  由 −1.0 改为 0.0（−1.0 比整集回报还大，掩码一旦失效会毁掉 value function），
  并加 `STEP_BUDGET_FACTOR=2` 的步数上限——持续被拒的动作会把 episode 截断
  （`truncated=True`，非终止）而不是无限循环。截断也会带上 `TERMINAL_METRICS` 三个键，
  否则 SB3 的 Monitor 会在 `info[key]` 上抛 KeyError。
- generated 训练/验证和 PSPLIB 测试统一使用 122 节点 cap，不执行模型扩宽。
- 训练耗时主要在 PPO 的 update（约 86%），环境采样不是瓶颈；用 `scripts/bench_ppo.py` 在目标
  机器上定 batch/线程，勿直接套用其他机器的结论。
- PPO 默认使用 `critical-path-shaping=0.5`：根据已排活动的关键路径下界重新分配
  makespan 的信用，episode 总目标仍等价于最小化 makespan；设为 `0` 可关闭。
- PPO 默认 `gamma=1.0`（2026-09-10 起，原 0.999）：episode 回报精确等于
  `-makespan/time_scale`。γ<1 会把终止增量按 γ^(n-1) 衰减（j30 0.97 → j120 0.89 →
  RG300 0.74），混合规模训练池下 critic 实际在拟合一族彼此不同的目标；episode 长度
  固定、转移确定、GAE 逐 rollout 重置，γ=1 无副作用。
- PPO 默认 `n_epochs=3`（2026-09-10 起，原 5）：update 耗时对 n_epochs 严格线性，
  而实测 `train/approx_kl`≈1e-6/轮，比 `target_kl=0.02` 低 4 个数量级，KL 早停永不触发。
- PPO 默认 `batch_size=1024`（2026-09-10 起，原 4096）：`16*256=4096` 原来**恰好等于**
  batch_size，等于每 rollout 只有 1 个 minibatch、3 个梯度步。update 成本 ≈
  n_epochs × rollout × 单样本成本，minibatch 数只是二阶效应（实测 512 与 1024 差 <5%），
  所以 batch 1024 是"同样本预算下 3→12 个优化步"，不是"更快"。
- `train_cpu.sh` 默认开 `TORCH_COMPILE=1`（`--compile-mode default`，CPU 上不用
  `reduce-overhead`——那是 CUDA graphs 模式）。启动时 `scripts/train_ppo` 会用
  一次性探针验证 inductor 后端：**探针期间 `suppress_errors` 关闭**，否则 dynamo 会静默
  降级成 eager 让探针误报成功；探针通过后才打开 `suppress_errors` 作为第二道网。
  后端不可用（典型：有 g++ 但没有 `omp.h`）时只打印原因并退回 eager，不再中途终止训练。
  代价：失败探针本身约 100 s 启动开销，`TORCH_COMPILE=0` 可跳过。
- BKS：`data/bks/bks_psplib.json`（由 RCPLIB xlsx 合成，勿单独引用平凡兜底 UB 列）。
