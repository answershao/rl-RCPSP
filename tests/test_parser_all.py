# tests/test_parser_all.py — full-corpus parse & sanity check
import sys, glob, time

sys.path.insert(0, "src")
from rcpsp_data import load_instance

files = sorted(glob.glob("data/psplib/**/*.sm", recursive=True)) + sorted(
    glob.glob("data/oras/**/*.rcp", recursive=True)
)

print(f"total files: {len(files)}")

bad, t0 = [], time.time()
for k, f in enumerate(files):
    try:
        inst = load_instance(f)
        order = inst.topological_order()
        assert len(order) == inst.n_activities
        # every duration >= 0, dummy source/sink have 0 duration
        assert inst.durations.min() >= 0
        assert inst.durations[0] == 0 and inst.durations[-1] == 0
        # demands never exceed capacity (feasibility sanity)
        assert (inst.demands <= inst.capacities).all()
    except Exception as e:
        bad.append((f, repr(e)))
    if (k + 1) % 500 == 0:
        print(f"  {k+1}/{len(files)} ({time.time()-t0:.0f}s)")

print(f"\nparsed OK: {len(files) - len(bad)}/{len(files)}  ({time.time()-t0:.0f}s)")
for f, e in bad[:20]:
    print("FAIL:", f, e)
if not bad:
    print("FULL CORPUS PASSED")
