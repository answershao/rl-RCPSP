"""Gantt chart rendering for RCMPSP schedules."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.core.rcmpsp import Instance, Schedule


def plot_gantt(
    instance: Instance,
    schedule: Schedule,
    output: str | Path,
    *,
    title: str | None = None,
    show_dummy: bool = True,
) -> Path:
    """Save a readable project-colored Gantt chart and return its path."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    activity_ids = sorted(instance.activities, key=lambda item: (item[0], item[1]))
    project_count = max((project for project, _ in activity_ids), default=0)
    cmap = plt.get_cmap("tab20", max(project_count, 1))
    colors = {project: cmap(project - 1) for project in range(1, project_count + 1)}

    height = max(6.0, len(activity_ids) * 0.22)
    fig, ax = plt.subplots(figsize=(16, height), constrained_layout=True)
    bar_height = 0.72
    labels: list[str] = []

    for y, activity_id in enumerate(activity_ids):
        activity = instance.activities[activity_id]
        start = schedule.starts[activity_id]
        labels.append(f"P{activity_id[0]}-A{activity_id[1]}")

        if activity.duration == 0:
            if show_dummy:
                ax.plot(start, y, marker="D", markersize=6, color=colors[activity_id[0]], zorder=3)
        else:
            ax.barh(
                y,
                activity.duration,
                left=start,
                height=bar_height,
                color=colors[activity_id[0]],
                edgecolor="black",
                linewidth=0.35,
            )
            if activity.duration >= max(3, schedule.makespan // 80):
                ax.text(
                    start + activity.duration / 2,
                    y,
                    str(activity_id[1]),
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
    ax.set_title(title or f"RCMPSP Gantt chart: {instance.name} (Cmax={schedule.makespan})")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(facecolor=colors[p], edgecolor="black", label=f"Project {p}") for p in colors],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
