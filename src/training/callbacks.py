"""Stable-Baselines3 callbacks for RCPSP-specific training diagnostics."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


TERMINAL_METRICS = (
    "makespan",
    "normalized_makespan",
    "resource_utilization",
)


class RCPSPMetricsCallback(BaseCallback):
    """Log rolling task metrics from terminal environment ``info`` dictionaries."""

    def __init__(
        self,
        window_size: int = 100,
        checkpoint_dir: str | Path | None = None,
        early_stop_patience: int = 0,
        validation_interval: int = 1,
        validation_min_delta: float = 0.0,
        validation_evaluator: Callable[[Any], float] | None = None,
    ):
        super().__init__()
        if (
            early_stop_patience < 0
            or validation_interval < 1
            or validation_min_delta < 0
        ):
            raise ValueError(
                "patience and min-delta must be non-negative; interval must be positive"
            )
        if early_stop_patience and validation_evaluator is None:
            raise ValueError("early stopping requires a validation evaluator")
        self.window_size = window_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.early_stop_patience = early_stop_patience
        self.validation_interval = validation_interval
        self.validation_min_delta = validation_min_delta
        self.validation_evaluator = validation_evaluator
        self.best_validation_gap = np.inf
        self._patience_reference_gap = np.inf
        self.validation_evaluations = 0
        self._rollouts = 0
        self._last_evaluated_rollout = -1
        self._bad_validation_evaluations = 0
        self._stop_requested = False
        self._values = {name: deque(maxlen=window_size) for name in TERMINAL_METRICS}

    @property
    def best_model_path(self) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / "best_model.zip"

    def _on_training_start(self) -> None:
        self._evaluate_validation()

    def _on_rollout_start(self) -> None:
        if self._rollouts and self._rollouts % self.validation_interval == 0:
            self._evaluate_validation()

    def _on_step(self) -> bool:
        if self._stop_requested:
            return False
        for info in self.locals.get("infos", []):
            # ``makespan`` is the partial schedule frontier on ordinary
            # transitions. Monitor adds ``episode`` only when it terminates.
            if "episode" not in info:
                continue
            for name, values in self._values.items():
                value = info.get(name)
                if value is not None:
                    values.append(float(value))
        return True

    def _on_rollout_end(self) -> None:
        for name, values in self._values.items():
            if values:
                self.logger.record(f"rcpsp/{name}_mean", float(np.mean(values)))

        self._rollouts += 1

    def _evaluate_validation(self) -> None:
        if self.validation_evaluator is None:
            return
        gap = float(self.validation_evaluator(self.model))
        if not np.isfinite(gap):
            raise ValueError("validation evaluator returned a non-finite relative gap")
        self.validation_evaluations += 1
        self._last_evaluated_rollout = self._rollouts
        self.logger.record("validation/fifo_relative_gap", gap)
        if gap < self.best_validation_gap:
            self.best_validation_gap = gap
            if self.best_model_path is not None:
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_path))
            self.logger.record("validation/best_fifo_relative_gap", gap)

        significant_improvement = (
            gap < self._patience_reference_gap - self.validation_min_delta
        )
        if significant_improvement:
            self._patience_reference_gap = gap
            self._bad_validation_evaluations = 0
        elif self.early_stop_patience:
            self._bad_validation_evaluations += 1
            if self._bad_validation_evaluations >= self.early_stop_patience:
                self._stop_requested = True
                self.logger.record("validation/early_stop", 1)
                print(
                    "Early stopping: no validation FIFO-relative-gap improvement "
                    f"for {self.early_stop_patience} evaluations"
                )

    def _on_training_end(self) -> None:
        if self._last_evaluated_rollout != self._rollouts:
            self._evaluate_validation()
