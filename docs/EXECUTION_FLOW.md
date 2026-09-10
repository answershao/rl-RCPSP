# RCPSP × PPO —— 执行流程与重构路线图

> 项目：资源受限项目调度（单项目 RCPSP）× PPO 强化学习，与优先规则 / GA / GPHH 基线对比。
> 文档用途：把「数据协议 → 各方法入口 → 汇总出表」的端到端流程固定下来，并给出后续
> 「分层 + 命名收敛」重构的路线图（测试先行）。更新日期：2026-09-09（R0–R4 已达成，
> 35 passed；R5 收尾待办）。

---

## 1. 数据与协议（已定稿，不可改口径）

| 套件 | 数量 | 格式 | 位置 | 角色 |
|---|---|---|---|---|
| PSPLIB j30–j120 | 2040 | `.sm` | `data/psplib/` | **test（主测试集）**，BKS 已知 → 精确 gap；训练全程不可见 |
| 生成池 psp_grid_bal | 1216 | `.rcp` | `data/generated/psp_grid_bal/` | **训练池**（PSPLIB 名义因子网格上的新实例；336 格 × 按 n=30/60/90/120 配平 8/4/2/1，`scripts/generate_pool.py` 生成，train 1148 + val 68） |
| RG30（Set 1–5） | 1800 | `.rcp` | `data/oras/RCPSP/RG30/` | 历史训练池（协议已退役；数据保留，仅 `--suites rg30` 基线可用） |
| RG300 | 480 | `.rcp` | `data/oras/RCPSP/RG300/` | 可选泛化参考（跨规模 302 活动；非主 test） |
| Patterson | 110 | `.rcp` | `data/oras/RCPSP/Patterson/` | 可选补充参考（非主 test） |
| BKS | — | xlsx | `data/bks/RCPLIB (Parameters and BKS).xlsx` | 最优值来源（合成 → json）+ PSPLIB 名义因子设计（`All` sheet → `scripts/psplib_design.py`） |

固定约定：

- **`splits.json` 是唯一切分清单**（2026-09-10 起 = 生成池协议）：`train`(1148) /
  `validation`(68) / `evaluation.{psplib_j30,j60,j90,j120,rg300,patterson}`。所有入口只读它取数，
  禁止自行扫描目录当切分。（`read_protocol` 仍兼容旧键名 `rg30_train`/`rg30_validation`。）
- **解析唯一入口**：`src/data/parsers.py`（.sm/.rcp → RCPSPInstance）→ `src/data/adapter.py`：
  `load_core_instance(path, name=None)`（→ core `Instance`）与 `to_core_instance`。
- 实例唯一标识 = **相对路径去扩展名**（`instance_id`，RG30 跨 Set 有同名 stem，不可只用文件名）。
- 全局模型 cap：**302 活动 / 4 资源 / MAX_SUCCESSORS=96 + MAX_PREDECESSORS=96**（单模型零样本覆盖 j30→RG300）。
  GIN 消息按邻居数取均值（非求和），否则 RG300 稠密图会让嵌入幅度爆炸；入/出度以显式特征喂入。
- BKS 缓存：`data/bks/bks_psplib.json`（`scripts/extract_bks.py` 从 PSPLIB sheet 的 160 个 UB 类列取
  min 合成）。⚠️ 勿单独引用 `UB-LAA/LAB/LSA/LSB/LPA/LPB-*` 列（恒等于工期总和的平凡兜底 UB）。

## 2. 端到端执行流程

```
 data/  ──scripts/generate_pool.py──▶ data/generated/psp_grid_bal ──▶ splits.json
      │
      ├─ scripts/baselines.py ───────────────▶ rules CSV（25 列 makespan：serial/parallel × 11 规则 + WCS）
      ├─ scripts/run_ga.py ──────────────────▶ ga CSV（ga_makespan，默认 50×200）
      ├─ scripts/run_gphh.py ────────────────▶ best_rule.txt + eval_summary.csv（训练只读 train 分片）
      ├─ scripts/train_ppo.py ───────────────▶ outputs/experiments/ppo/<run>/ppo_eval_summary.csv
      │     （训练=train 分片（生成池）；验证=validation 按 serial_LST gap 择优 checkpoint；
      │       final_model.zip = 训练结束恢复 best 验证权重后的落盘，即 best model）
      │        评估对象：主 test=PSPLIB 各规模；rg300/patterson 为可选泛化/补充参考
      │        shell：train_a800.sh / train_cpu.sh（训练+评估）、eval_cpu.sh（仅评估已有模型）
      │        search_ppo.py（跨 seed：outputs/experiments/ppo/seedN/final_model.zip）→ shell：search_cpu.sh
      │
 data/bks/xlsx ──scripts/extract_bks.py──▶ bks_psplib.json
      │
      ▼
 scripts/aggregate_results.py --rules … --ga … --gphh … --ppo … --bks …
      ▶ merged_detail.csv（每实例宽表 + gap 列） + summary_by_suite.csv（方法×套件 gap / 达 BKS 数）
```

各入口速查：

| 入口 | 取数 | 关键参数 | 产物 |
|---|---|---|---|
| `python -m scripts.generate_pool` | PSPLIB + RCPLIB xlsx | `--mode/--replicates/--validation-fraction/--widen/--workers` | 生成池 `.rcp` + manifest（当前 `splits.json` 的来源） |
| `python -m scripts.baselines` | `--data-root/--suites` | `--max-instances 0`=全量、`--instance-workers` | `makespan_summary.csv` |
| `python -m scripts.run_ga` | 同 baselines | `--population 50 --generations 200 --seed` | `ga.csv` |
| `python -m scripts.run_gphh` | `--splits`(训练) + eval-suites | `--train-instances/--population/--generations/--eval-workers` | `best_rule.txt`/`eval_summary.csv` |
| `python -m scripts.train_ppo` | `--splits` | `--total-timesteps/--n-envs/--max-activities/--max-resources/--eval-suites/--output-dir` | `ppo_eval_summary.csv`、模型 |
| `python -m scripts.aggregate_results` | 上述 CSV | `--rules/--ga/--gphh/--ppo/--bks/--params/--out-dir` | `merged_detail.csv`+`summary_by_suite.csv`+`summary_by_regime.csv`（RF 档 × RS 四分位） |

辅助：`scripts/extract_bks.py`、`scripts/visualize_instance.py`（.sm/.rcp → Gantt/AON）、
`scripts/search_ppo.py`（跨 seed×采样评估）、`scripts/compare_ppo_results.py`（两轮 PPO 对比）。

## 3. 测试体系（pytest，本机 Mac 标准）

- 一条命令全量回归（含 4430 全语料解析、SB3 短训）：
  ```bash
  source /Users/fn/miniconda3/etc/profile.d/conda.sh && conda activate rl
  python -m pytest -q          # 44 passed（≈40s 快测 + ≈4min 全语料 slow）
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
tests/    14 个 pytest 文件（61 用例）
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
| **R5 收尾** ✅ | argparse 自省校验 shell×4 + launch.json×9 任务全部参数合法（零漂移）；补根 `README.md`（环境/测试/结构/复现命令/输出目录规范）；本流程文档随仓库同步 | 冒烟命令与 R0 结果数值一致 |
| **R6 主线对齐** ✅ | ①`observation` 时长特征改 per-instance 最大值归一（消除 ∝1/n 伪影，j120 dur 交叠 0.22→0.99）；②协议键正名 `train`/`validation`（`read_protocol` 兼容旧键）；③退役 `scripts/make_splits.py`（协议由 `generate_pool` 接管）；④`aggregate_results --params` 增 `summary_by_regime.csv`（RF 档 × RS 四分位） | `pytest -W error` 61 passed；`instance_stats` 重算：j30-j120 时长特征交叠 0.97–0.99、RF×RS 联合覆盖 92–100%；regime 汇总复现 j30 GA gap 1.91%(紧)→0.03%(松) |

贯穿原则：**任何重构步骤都以 `pytest -q` + 一个「基线数值对齐」冒烟（固定实例的 makespan/gap 不变）作为放行条件**；涉及命名/结构的改动独立小步提交，避免一次大爆炸。

## 5. 典型跑法（结果复现）

> 产物路径规范、PPO shell 用法（train_a800.sh/train_cpu.sh/eval_cpu.sh/search_cpu.sh 的
> 环境变量覆盖）与仓库门面见根 `README.md`。本小节只列命令骨架。

```bash
# 基线 → 汇总（j30 全量 480 实例示例）
python -m scripts.baselines --data-root data --suites psplib_j30 --instance-workers 8 \
  --seed 17 --output-csv outputs/rules_j30/makespan_summary.csv
python -m scripts.run_ga --data-root data --suites psplib_j30 --instance-workers 8 \
  --seed 17 --output-csv outputs/ga_j30/ga.csv
python -m scripts.run_gphh --data-root data --splits splits.json --train-instances 60 \
  --seed 17 --eval-suites psplib_j30 --eval-workers 8 --output-dir outputs/gphh_j30/trial1
python -m scripts.aggregate_results --rules outputs/rules_j30/makespan_summary.csv \
  --ga outputs/ga_j30/ga.csv --gphh outputs/gphh_j30/trial1/eval_summary.csv \
  --bks data/bks/bks_psplib.json --out-dir outputs/aggregate_j30

# PPO（GPU 机 train_a800.sh / CPU 机 train_cpu.sh；评估已有模型 eval_cpu.sh；跨 seed search_cpu.sh）
bash train_a800.sh
```
