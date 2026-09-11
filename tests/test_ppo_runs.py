from pathlib import Path

from scripts.ppo_runs import latest_training_run, resolve_model_path


def test_latest_run_uses_timestamp_and_skips_incomplete_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "cpu_runs"
    complete = runs_root / "cpu_20260910_221202" / "checkpoints"
    newer_incomplete = runs_root / "cpu_20260911_103928" / "checkpoints"
    complete.mkdir(parents=True)
    newer_incomplete.mkdir(parents=True)
    (complete / "best_model.zip").touch()

    selected = latest_training_run(runs_root, Path("checkpoints/best_model.zip"))

    assert selected == complete.parent


def test_resolve_model_path_supports_seed_layout(tmp_path: Path) -> None:
    model = tmp_path / "models" / "seed17" / "final_model.zip"
    model.parent.mkdir(parents=True)
    model.touch()

    assert resolve_model_path(model.parents[1], Path("final_model.zip"), 17) == model
