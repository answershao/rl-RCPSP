"""Main-protocol corpus parse and structural sanity integration test."""

import time

import pytest

from src.data.parsers import load_instance

EXPECTED_COUNT = 3256  # 2040 PSPLIB test + 1216 generated train/validation


def _corpus_files(repo_root):
    sm = sorted((repo_root / "data/psplib").glob("**/*.sm"))
    rcp = sorted((repo_root / "data/generated/psp_grid_bal").glob("**/*.rcp"))
    return sm + rcp


@pytest.mark.slow
def test_full_corpus_parse_and_sanity(repo_root):
    files = _corpus_files(repo_root)
    assert len(files) == EXPECTED_COUNT, (
        f"corpus changed: found {len(files)} files, expected {EXPECTED_COUNT}"
    )

    bad = []
    t0 = time.time()
    for index, path in enumerate(files):
        try:
            inst = load_instance(path)
            order = inst.topological_order()
            assert len(order) == inst.n_activities
            assert inst.durations.min() >= 0
            assert inst.durations[0] == 0 and inst.durations[-1] == 0  # dummy source/sink
            assert (inst.demands <= inst.capacities).all()  # feasibility sanity
        except Exception as exc:  # noqa: BLE001 - report any per-file failure
            bad.append((str(path), repr(exc)))
        if (index + 1) % 1000 == 0:
            print(f"  {index + 1}/{len(files)} ({time.time() - t0:.0f}s)")

    assert not bad, f"{len(bad)} corpus failures, first: {bad[:5]}"
