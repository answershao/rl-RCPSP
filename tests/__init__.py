"""Tests for the RCPSP package."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Real single-project RCPSP instances shared by the environment / RL tests.
TEST_INSTANCE = REPO_ROOT / "data/psplib/j30/j3010_1.sm"
TEST_INSTANCE_2 = REPO_ROOT / "data/psplib/j30/j3010_2.sm"
