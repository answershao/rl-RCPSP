from pathlib import Path

from scripts.search_ppo import DEFAULT_MODEL_FILE, resolve_search_model_paths


def test_search_default_uses_best_model_from_newest_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "cpu_runs"
    old_best = runs_root / "cpu_20260910_221202" / DEFAULT_MODEL_FILE
    new_best = runs_root / "cpu_20260911_143034" / DEFAULT_MODEL_FILE
    legacy_best = runs_root / "seed17" / DEFAULT_MODEL_FILE
    for model in (old_best, new_best, legacy_best):
        model.parent.mkdir(parents=True, exist_ok=True)
        model.touch()

    selected = resolve_search_model_paths(runs_root, DEFAULT_MODEL_FILE, [17])

    assert selected == {17: new_best}
