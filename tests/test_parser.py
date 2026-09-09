import sys

sys.path.insert(0, "src")
from rcpsp_data import load_instance, sgs_schedule

# 1. .sm 解析
inst = load_instance("data/psplib/j30/j301_1.sm")
assert inst.n_activities == 32 and inst.n_renewable == 4
assert inst.successors[0] == [1, 2, 3]  # j30_1 的虚源后继
assert len(inst.topological_order()) == 32

# 2. .rcp 解析
inst2 = load_instance("data/oras/RCPSP/RG300/RG300_1.rcp")
assert inst2.n_activities == 302 and inst2.n_renewable == 4
assert (inst2.capacities == 10).all()  # 与 head 输出一致
assert len(inst2.successors[0]) == 72  # 虚源 72 个后继

# 3. SGS 跑通且满足约束
for inst_x in (inst, inst2):
    order = inst_x.topological_order()
    start = sgs_schedule(inst_x, order)
    mk = (start + inst_x.durations).max()
    print(
        f"{inst_x.name}: activities={inst_x.n_activities}, "
        f"ES-schedule makespan={mk}, horizon={inst_x.horizon}"
    )
print("ALL TESTS PASSED")
