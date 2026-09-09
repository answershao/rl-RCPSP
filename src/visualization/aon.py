"""Activity-on-Node (AON) network rendering for RCMPSP instances."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.core.rcmpsp import Instance, Schedule


def plot_aon(
    instance: Instance,
    output: str | Path,
    *,
    schedule: Schedule | None = None,
    title: str | None = None,
) -> Path:
    """Save an AON precedence graph with project-colored activity nodes."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.DiGraph()
    graph.add_nodes_from(instance.activities)
    for activity in instance.activities.values():
        graph.add_edges_from((activity.id, successor) for successor in activity.successors)

    # Longest predecessor distance gives a stable left-to-right precedence view.
    layer: dict[tuple[int, int], int] = {}
    for node in nx.topological_sort(graph):
        layer[node] = max((layer[pred] + 1 for pred in graph.predecessors(node)), default=0)

    project_count = max((project for project, _ in graph.nodes), default=0)
    cmap = plt.get_cmap("tab20", max(project_count, 1))
    colors = {project: cmap(project - 1) for project in range(1, project_count + 1)}
    positions: dict[tuple[int, int], tuple[float, float]] = {}
    project_offsets: dict[int, float] = {}
    next_project_y = 0.0
    for project in range(1, project_count + 1):
        project_nodes = sorted((node for node in graph if node[0] == project), key=lambda node: (layer[node], node[1]))
        by_layer: dict[int, list[tuple[int, int]]] = {}
        for node in project_nodes:
            by_layer.setdefault(layer[node], []).append(node)
        project_offsets[project] = next_project_y
        project_height = max((len(nodes) for nodes in by_layer.values()), default=1)
        next_project_y -= project_height + 2.5
        for current_layer, nodes in by_layer.items():
            for offset, node in enumerate(nodes):
                positions[node] = (current_layer, project_offsets[project] - offset * 0.65)

    max_nodes_per_layer = max((sum(1 for node in graph if layer[node] == current_layer) for current_layer in set(layer.values())), default=1)
    fig_height = max(6.0, (abs(next_project_y) + max_nodes_per_layer) * 0.55)
    fig, ax = plt.subplots(figsize=(max(12.0, max(layer.values(), default=1) * 0.8), fig_height), constrained_layout=True)

    regular = [node for node in graph if instance.activities[node].duration > 0]
    dummy = [node for node in graph if instance.activities[node].duration == 0]
    nx.draw_networkx_edges(graph, positions, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=8, width=0.6, alpha=0.45)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=regular,
        node_color=[colors[node[0]] for node in regular],
        node_shape="s",
        node_size=500,
        edgecolors="black",
        linewidths=0.5,
        ax=ax,
    )
    if dummy:
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=dummy,
            node_color=[colors[node[0]] for node in dummy],
            node_shape="D",
            node_size=440,
            edgecolors="black",
            linewidths=0.7,
            ax=ax,
        )

    labels = {}
    for node in graph:
        activity = instance.activities[node]
        text = f"P{node[0]}-A{node[1]}\nd={activity.duration}"
        if schedule is not None:
            text += f"\n[{schedule.starts[node]}-{schedule.finishes[node]}]"
        labels[node] = text
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=5.5, ax=ax)

    ax.set_title(title or f"RCMPSP AON network: {instance.name}")
    ax.axis("off")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
