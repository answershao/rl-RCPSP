"""Parser structure tests for .sm / .rcp.

Exercises ``src.data.parsers`` (the shared parse backend for the whole stack:
baselines, GA/GPHH and the RL data adapter).  Scheduling is *not* implemented
at the parser level — the third test verifies parsed instances are directly
schedulable by the canonical core kernel (``src.core.rcpsp.generate_schedule``)
through the ``src.data.adapter`` bridge.
"""

import pytest

from src.core.rcpsp import generate_schedule
from src.data.adapter import load_core_instance
from src.data.parsers import load_instance

J30_SM = "data/psplib/j30/j301_1.sm"
GENERATED_RCP = "data/generated/psp_grid_bal/n120/c0241_r00.rcp"


def test_sm_parse_j30(repo_root):
    inst = load_instance(repo_root / J30_SM)
    assert inst.n_activities == 32 and inst.n_renewable == 4
    # j301_1: dummy source fans out to the first three real jobs.
    assert inst.successors[0] == [1, 2, 3]
    assert len(inst.topological_order()) == 32


def test_rcp_parse_generated_instance(repo_root):
    inst = load_instance(repo_root / GENERATED_RCP)
    assert inst.n_activities == 122 and inst.n_renewable == 4
    assert len(inst.topological_order()) == 122


def test_parsed_instances_are_schedulable_by_core(repo_root):
    """Parse → adapt → core serial SGS end-to-end on both file formats.

    Guarantees parser output is consumable by the canonical kernel without any
    parser-local scheduling shim (R2: single serial SGS lives in core only).
    """
    for rel in (J30_SM, GENERATED_RCP):
        instance = load_core_instance(repo_root / rel)
        schedule = generate_schedule(instance)  # FIFO default priority
        assert schedule.makespan > 0, f"core SGS produced empty result for {rel}"
