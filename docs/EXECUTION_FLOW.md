# RCPSP × PPO —— 执行流程与重构路线图

> 项目：资源受限项目调度（单项目 RCPSP）× PPO 强化学习，与优先规则 / GA / GPHH 基线对比。
> 文档用途：把「数据协议 → 各方法入口 → 汇总出表」的端到端流程固定下来，并给出后续
> 「分层 + 命名收敛」重构的路线图（测试先行）。更新日期：2026-09-10（R0–R6 已达成，
> 65 passed）。

---

## 1. 数据与协议（已定稿，不可改口径）

| 套件 | 数量 | 格式 | 位置 | 角色 |
|---|---|---|---|---|
| PSPLIB j30–j120 | 2040 | `.sm` | `data/psplib/` | **test（主测试集）**，BKS 已知 → 精确 gap；训练全程不可见 |
| 生成池 psp_grid_bal | 1216 | `.rcp` | `data/generated/psp_grid_bal/` | **训练池**（PSPLIB 名义因子网格上的新实例；336 格 × 按 n=30/60/90/120 配平 8/4/2/1，`scripts/generate_pool.py` 生成，train 1088 + val 128） |
| BKS | — | xlsx | `data/bks/RCPLIB (Parameters and BKS).xlsx` | 最优值来源（合成 → json）+ PSPLIB 名义因子设计（`All` sheet → `scripts/psplib_design.py`） |

固定约定：

- **`splits.json` 是唯一切分清单**（2026-09-10 起 = 生成池协议）：`train`(1088) /
  `validation`(128，每种规模 32) / `evaluation.{psplib_j30,j60,j90,j120}`。所有入口只读它取数，
  禁止自行扫描目录当切分。
- **解析唯一入口**：`src/data/parsers.py`（.sm/.rcp → RCPSPInstance）→ `src/data/adapter.py`：
  `load_core_instance(path, name=None)`（→ core `Instance`）与 `to_core_instance`。
- 实例唯一标识 = **相对路径去扩展名**（`instance_id`）。
- 全局模型 cap：**122 活动 / 4 资源 / MAX_SUCCESSORS=20 + MAX_PREDECESSORS=20**。
  GIN 消息按邻居数取均值（非求和）；入/出度以显式特征喂入。
- BKS 缓存：`data/bks/bks_psplib.json`（`scripts/extract_bks.py` 从 PSPLIB sheet 的 160 个 UB 类列取
  min 合成）。⚠️ 勿单独引用 `UB-LAA/LAB/LSA/LSB/LPA/LPB-*` 列（恒等于工期总和的平凡兜底 UB）。
- **训练池也是登记套件**：`SUITE_SPECS` 增 `psp_grid` → `data/generated/psp_grid_bal`（role=`training-pool`），
  唯一用途是让 `scripts/baselines` 能产出 S3 的 validation 参照规则。`DEFAULT_SUITES` 即 `psp_grid`
  （`baselines`/`run_ga` 不带 `--suites` 时默认跑训练池，S3 因此不需要额外参数）；
  `EVALUATION_SUITES`（role=`final-evaluation` 的 6 个）是 `run_gphh --eval-suites` 的默认值 ——
  **任何训练池都不会进评测默认值**，避免"在拟合唱片上评测"。

## 2. 端到端执行顺序

> 本节只定义**阶段划分与依赖**（口径层）；可复制粘贴的逐步命令、每步通过判据与排错速查
> 见 **[docs/MAINLINE.md](MAINLINE.md)**（操作层），两者同源、不重复维护命令细节。

脚本按**依赖**排列：S1 是切分协议的唯一来源，S2 / S3 分别是 S6 / S5 的前置；旁路脚本（§2.2）不参与主线。

### 2.1 主线（按序执行）

| 序 | 阶段 | 入口 | 输入依赖 | 产物 |
|---|---|---|---|---|
| **S0** | 环境自检 | `python -m pytest -q -W error` | — | 65 passed |
| **S1** | 生成训练池 + 协议 | `python -m scripts.generate_pool --mode psp-grid --replicates 8,4,2,1 --validation-per-size 32 --widen --workers 8 --output data/generated/psp_grid_bal --manifest data/generated/psp_grid_bal.json`，生成后把 manifest 拷为 `splits.json` | `src/data/generator.py` + RCPLIB xlsx 名义设计 | 1216 `.rcp` + `splits.json`（seed 20260910 可复现；协议不变则跳过） |
| **S2** | 协议诊断（regime 参数源） | `python -m scripts.instance_stats --data-root data --workers 8` | `splits.json` | `outputs/instance_stats/{instances,summary,coverage}.csv` |
| **S3** | 训练池参照规则 | `python -m scripts.baselines --data-root data --instance-workers 8 --output-csv outputs/rules_psp_grid/makespan_summary.csv`（`--suites` 已默认 `psp_grid`） | `psp_grid` 套件（已登记）+ `splits.json` | 供 S5 择优用的 `serial_LST` 参照列 |
| **S4** | 基线（test 侧） | `python -m scripts.baselines` → `scripts.run_ga` → `scripts.run_gphh`（`--suites psplib_j30,psplib_j60,psplib_j90,psplib_j120`） | —（GPHH 另需 S1 的 train 分片） | `outputs/{rules,ga,gphh}_psplib/…` |
| **S5** | PPO 训练 | 先用 `python -m scripts.bench_ppo` 在目标机定 batch/线程，再 `bash train_cpu.sh` / `train_a800.sh`（内部即 `scripts.train_ppo`） | **S1 + S3** | `outputs/experiments/ppo/<run>/{final_model.zip,ppo_eval_summary.csv}` |
| **S6** | 汇总出表 | `python -m scripts.aggregate_results --rules … --ga … --gphh … [--ppo …] --bks data/bks/bks_psplib.json --params outputs/instance_stats/instances.csv --out-dir outputs/aggregate_psplib` | S2 + S4（有 PPO 时 + S5） | `merged_detail.csv` + `summary_by_suite.csv` + `summary_by_regime.csv` |

> `scripts/train_ppo.py --ref-rules` 默认 `outputs/rules_psp_grid/makespan_summary.csv`（= S3 产物），
> 并要求它覆盖全部 128 个 validation 实例 —— **S3 必须先跑，否则 S5 直接报错**（`FileNotFoundError`，
> 或覆盖不全时 `--ref-rules does not cover N validation instances`）。参照列默认取 `serial_LST`
> （`--ref-rule`）。`bash eval_cpu.sh`（`--evaluate-only`）不受影响：它在读参照规则之前就已 return。

一键封装：`bash run_test_baselines.sh` = **S2 + S4 + S6**（分阶段开关 `RUN_STATS/RUN_RULES/RUN_GA/RUN_GPHH/RUN_AGGREGATE`；`WITH_PPO=1` 并入 PPO 列；已存在的阶段输出自动跳过，`ALLOW_OVERWRITE=1` 重算）。S1 / S3 / S5 不在一键脚本内。

### 2.2 旁路脚本（不参与主线）

| 脚本 | 用途 | 前置 |
|---|---|---|
| `python -m scripts.extract_bks` | RCPLIB xlsx → `data/bks/bks_psplib.json`（S6 的 BKS 输入；产物已入库，通常无需重跑） | — |
| `bash eval_cpu.sh` | 仅评估已有模型，默认评测 `splits.json` 的全部 evaluation 组 | S5 的 `final_model.zip` |
| `bash search_cpu.sh` | 跨 seed 采样评估，读 `outputs/experiments/ppo/seed<N>/final_model.zip` | S5（多 seed）+ 参照规则覆盖所评组 |
| `python -m scripts.compare_ppo_results` | 两轮 PPO 的 `ppo_eval_summary.csv` 逐实例对比 | 两次 S5 |
| `python -m scripts.visualize_instance` | 单实例 Gantt + AON 图 | — |

### 2.3 各入口速查（参数细节）

| 入口 | 取数 | 关键参数 | 产物 |
|---|---|---|---|
| `python -m scripts.generate_pool` | PSPLIB + RCPLIB xlsx | `--mode/--replicates/--validation-per-size/--widen/--workers` | 生成池 `.rcp` + manifest（当前 `splits.json` 的来源） |
| `python -m scripts.instance_stats` | `--splits` + 数据根 | `--workers/--output-dir/--rf-tol/--extra-pool` | `instances.csv` / `summary.csv` / `coverage.csv` |
| `python -m scripts.baselines` | `--data-root/--suites` | `--max-instances 0`=全量、`--instance-workers` | `makespan_summary.csv` |
| `python -m scripts.run_ga` | 同 baselines | `--population 50 --generations 200 --seed` | `ga.csv` |
| `python -m scripts.run_gphh` | `--splits`(训练) + eval-suites | `--train-instances/--population/--generations/--eval-workers` | `best_rule.txt`/`eval_summary.csv` |
| `python -m scripts.bench_ppo` | `--splits` | `--caps/--threads/--batch-sizes/--repeats` | 终端吞吐表（可选 `--output-csv`） |
| `python -m scripts.train_ppo` | `--splits` | `--total-timesteps/--n-envs/--max-activities/--max-resources/--eval-suites/--output-dir` | `ppo_eval_summary.csv`、模型 |
| `python -m scripts.aggregate_results` | 上述 CSV | `--rules/--ga/--gphh/--ppo/--bks/--params/--out-dir` | `merged_detail.csv` + `summary_by_suite.csv` + `summary_by_regime.csv`（RF 档 × RS 四分位） |

## 3. 测试体系（pytest，本机 Mac 标准）

- 一条命令全量回归（含 3256 个主协议实例解析、SB3 短训）：
  ```bash
  source /Users/fn/miniconda3/etc/profile.d/conda.sh && conda activate rl
  python -m pytest -q          # 65 passed（≈40s 快测 + ≈4min 全语料 slow）
  python -m pytest -q -W error # 无警告级失败（R3：第三方良性告警已在源点精准抑制）
  python -m pytest -m slow     # 仅慢项（全语料解析）
  ```
- 约定：`pytest.ini`（testpaths/markers）+ 根 `conftest.py`（sys.path）+ `tests/conftest.py`
  （repo_root / j30 样例 fixture）；快测默认跑，全语料解析标 `@pytest.mark.slow`。
- 覆盖：解析结构（.sm/.rcp）、规则列契约与确定性、GA/GPHH 可复现+单调+解码有效、
  环境（env/multi-instance/sb3 短训）、可视化、全语料解析回归。
- 2026-09-09 已清理：早期原型 `src/rcpsp_env.py`、`src/baselines.py` 及其专属测试
  （test_env/test_baselines/test_diag_quality）已删除 —— 功能由 `src/core/rules.py` 与
  `src/envs/rcpsp_env.py` 正式栈接管；解析统一走 `src/data/parsers.py`（.sm/.rcp → RCPSPInstance）。
- 2026-09-09 R1/R2 已收敛：`parsers.py` 不实现调度（遗留 `sgs_schedule` 已删），
  串行/并行 SGS 唯一实现位于 `src/core/rcpsp.py`；文件→core `Instance` 的唯一入口为
  `src/data/adapter.py::load_core_instance`（协议名用 `src/data/instances.py::loader_for`）。
- 2026-09-09 R3 已收敛：`src/py.typed` 标记；非法规则/非法 scheme 的 ValueError 带可用集合；
  `ppo.evaluate_paths` 的 static-cache name 一致性校验语义注释化（stem vs 协议 instance_id）；
  SB3 env_checker 对 2D 观测矩阵的形态提示在测试源点按消息精准抑制（GIN 策略有意为之）。

## 4. 重构路线图（分层 + 命名收敛，测试先行）

现状骨架（6880 行）：

```
src/  data/  parsers.py（解析）· adapter.py（适配）· instances.py（协议）
      core/     rcpsp.py(SGS+校验+Instance) rules.py ga.py gphh.py exact.py
      envs/     rcpsp_env.py multi_instance.py observation.py sb3_env.py
      training/ ppo.py features.py environments.py callbacks.py
      visualization/ aon.py gantt.py
scripts/  common（套件表/发现/进程池/CSV 公共件）· generate_pool psplib_design
          instance_stats baselines run_ga
          run_gphh train_ppo search_ppo bench_ppo compare_ppo_results extract_bks
          aggregate_results visualize_instance
tests/    14 个 test_*.py（65 用例）
```

> R1 已收敛：目录 `src/data`（解析/适配/协议）+ `src/envs`；`ActivityId` 由
> `(project, activity)` 元组收敛为单项目 0-indexed `int`（dummy source=0，sink=n-1）；
> 模块/类名去除 RC-MPSP 表述（`rcmpsp.py`→`rcpsp.py`、`RCMPSPEnv`→`RCPSPEnv`、
> `to_rcmpsp_instance`→`to_core_instance` 等）。

目标布局提案（每阶段独立提交、pytest 全绿后再进下一阶段）：

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **R0** ✅ | 清理 MPLIB2/.rcmp 接口；删除早期原型；测试统一 pytest | 34 passed（已达成） |
| **R1 分层收敛** ✅ | 目录重排为 `src/data`（解析/适配/协议）、`src/core`（SGS/rules/ga/gphh/exact）、`src/envs`、`src/training`；`core/rcpsp.py` 语义收敛为单项目调度内核（`ActivityId` 元组→int、模块名/类名去除多项目 RC-MPSP 表述） | `pytest -q` 全绿（34 passed）；规则/GA 在固定 5 个 j30 上 makespan 与重构前一致 |
| **R2 去重** ✅ | 移除 `src/data/parsers.py` 内遗留 `sgs_schedule`（统一走 core SGS）；删死代码 `load_bks`；public API 收口到 `src/data/adapter.py::load_core_instance`（scripts/baselines·run_ga·run_gphh、tests×3、instances.py 的 `load_instance+to_core_instance` 两步曲全部单点化） | `pytest -q` 全绿（34 passed）；grep 串行 SGS 仅剩 `core/rcpsp.py::generate_schedule`；固定 5 个 j30 数值与 R1 基线一致 |
| **R3 健壮性** ✅ | 类型标注（`src/py.typed`、parsers 字段/返回注解）；docstring（`evaluate_paths` 的 static-cache name 校验语义注释化：stem vs 协议 instance_id）；异常路径（非法规则/非法 scheme 的 ValueError 带可用集合；空清单/超界原有校验保留） | 全绿（35 passed）+ `python -m pytest -W error` 无警告级失败；SB3 env_checker 对 2D 观测的良性提示在测试源点按消息精准抑制；固定 j30 数值与 R1 基线一致 |
| **R4 scripts 薄化** ✅ | 新增 `scripts/common.py`（SUITE_SPECS 含 role / natural_key / find_instances / relative_posix / resolve_suite_ids / map_jobs spawn 驱动 / write_csv / print_suite_means 单一事实源），baselines·run_ga·run_gphh·**make_splits** 四个脚本的三份 SUITE_SPECS + 三份 find_instances + 三个进程池循环/CSV 写出全部单点化 | `pytest -q` 全绿 + `-W error` 干净；baselines/run_ga/gphh 冒烟输出与改造前 **列名/列序一致、行值一致**（仅 ga_seconds 计时列除外）；`make_splits` 重新生成 splits.json 逐字节一致；spawn 多进程路径冒烟通过 |
| **R5 收尾** ✅ | argparse 自省校验 shell×4 + launch.json×8 任务全部参数合法（零漂移）；补根 `README.md`（环境/测试/结构/复现命令/输出目录规范）；本流程文档随仓库同步 | 冒烟命令与 R0 结果数值一致 |
| **R6 主线对齐** ✅ | ①`observation` 时长特征改 per-instance 最大值归一（消除 ∝1/n 伪影，j120 dur 交叠 0.22→0.99）；②协议键正名 `train`/`validation`（`read_protocol` 兼容旧键）；③退役 `scripts/make_splits.py`（协议由 `generate_pool` 接管）；④`aggregate_results --params` 增 `summary_by_regime.csv`（RF 档 × RS 四分位） | `pytest -W error` 61 passed；`instance_stats` 重算：j30-j120 时长特征交叠 0.97–0.99、RF×RS 联合覆盖 92–100%；regime 汇总复现 j30 GA gap 1.91%(紧)→0.03%(松) |

贯穿原则：**任何重构步骤都以 `pytest -q` + 一个「基线数值对齐」冒烟（固定实例的 makespan/gap 不变）作为放行条件**；涉及命名/结构的改动独立小步提交，避免一次大爆炸。

## 5. 典型跑法（结果复现）

> 产物路径规范、PPO shell 用法（train_a800.sh/train_cpu.sh/eval_cpu.sh/search_cpu.sh 的
> 环境变量覆盖）与仓库门面见根 `README.md`。本小节只列命令骨架。

```bash
# S2 诊断（提供 aggregate 的 --params，缺了就没有 regime 分档表）
python -m scripts.instance_stats --data-root data --workers 8

# S4 基线 → S6 汇总（j30 全量 480 实例示例）
python -m scripts.baselines --data-root data --suites psplib_j30 --instance-workers 8 \
  --seed 17 --output-csv outputs/rules_j30/makespan_summary.csv
python -m scripts.run_ga --data-root data --suites psplib_j30 --instance-workers 8 \
  --seed 17 --output-csv outputs/ga_j30/ga.csv
python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 60 \
  --seed 17 --eval-suites psplib_j30 --eval-workers 8 --output-dir outputs/gphh_j30/trial1
python -m scripts.aggregate_results --rules outputs/rules_j30/makespan_summary.csv \
  --ga outputs/ga_j30/ga.csv --gphh outputs/gphh_j30/trial1/eval_summary.csv \
  --bks data/bks/bks_psplib.json --params outputs/instance_stats/instances.csv \
  --out-dir outputs/aggregate_j30

# S5 PPO（GPU 机 train_a800.sh / CPU 机 train_cpu.sh；评估已有模型 eval_cpu.sh；跨 seed search_cpu.sh）
bash train_a800.sh
```
