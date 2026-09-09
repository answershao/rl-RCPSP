"""Gantt chart rendering for single-project RCPSP schedules."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.core.rcpsp import Instance, Schedule


def plot_gantt(
    instance: Instance,
    schedule: Schedule,
    output: str | Path,
    *,
    title: str | None = None,
    show_dummy: bool = True,
) -> Path:
    """Save a readable Gantt chart for one single-project schedule."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    activity_ids = sorted(instance.activities)
    color = plt.get_cmap("tab20")(0)

    height = max(6.0, len(activity_ids) * 0.22)
    fig, ax = plt.subplots(figsize=(16, height), constrained_layout=True)
    bar_height = 0.72
    labels: list[str] = []

    for y, activity_id in enumerate(activity_ids):
        activity = instance.activities[activity_id]
        start = schedule.starts[activity_id]
        labels.append(f"A{activity_id}")

        if activity.duration == 0:
            if show_dummy:
                ax.plot(start, y, marker="D", markersize=6, color=color, zorder=3)
        else:
            ax.barh(
                y,
                activity.duration,
                left=start,
                height=bar_height,
                color=color,
                edgecolor="black",
                linewidth=0.35,
            )
            if activity.duration >= max(3, schedule.makespan // 80):
                ax.text(
                    start + activity.duration / 2,
                    y,
                    str(activity_id),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    clip_on=True,
                )

    ax.set_yticks(range(len(activity_ids)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(left=0, right=max(schedule.makespan, 1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")
    ax.set_title(title or f"RCPSP Gantt chart: {instance.name} (Cmax={schedule.makespan})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
