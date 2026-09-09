"""Activity-on-Node (AON) network rendering for single-project RCPSP instances."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.core.rcpsp import Instance, Schedule


def plot_aon(
    instance: Instance,
    output: str | Path,
    *,
    schedule: Schedule | None = None,
    title: str | None = None,
) -> Path:
    """Save an AON precedence graph for one single-project instance."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    graph = nx.DiGraph()
    graph.add_nodes_from(instance.activities)
    for activity in instance.activities.values():
        graph.add_edges_from((activity.id, successor) for successor in activity.successors)

    # Longest predecessor distance gives a stable left-to-right precedence view.
    layer: dict[int, int] = {}
    for node in nx.topological_sort(graph):
        layer[node] = max((layer[pred] + 1 for pred in graph.predecessors(node)), default=0)

    by_layer: dict[int, list[int]] = {}
    for node in graph:
        by_layer.setdefault(layer[node], []).append(node)
    positions: dict[int, tuple[float, float]] = {}
    max_nodes_per_layer = max((len(nodes) for nodes in by_layer.values()), default=1)
    for current_layer, nodes in by_layer.items():
        nodes = sorted(nodes)
        for offset, node in enumerate(nodes):
            # Column x = topological layer; y spreads nodes out within the layer.
            positions[node] = (current_layer, -offset * 0.65)

    fig_height = max(6.0, max_nodes_per_layer * 0.65 + 1.0)
    fig, ax = plt.subplots(figsize=(max(12.0, max(layer.values(), default=1) * 0.8), fig_height), constrained_layout=True)

    regular = [node for node in graph if instance.activities[node].duration > 0]
    dummy = [node for node in graph if instance.activities[node].duration == 0]
    regular_color = "#1f77b4"
    dummy_color = "#d62728"
    nx.draw_networkx_edges(graph, positions, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=8, width=0.6, alpha=0.45)
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=regular,
        node_color=regular_color,
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
            node_color=dummy_color,
            node_shape="D",
            node_size=440,
            edgecolors="black",
            linewidths=0.7,
            ax=ax,
        )

    labels = {}
    for node in graph:
        activity = instance.activities[node]
        text = f"A{node}\nd={activity.duration}"
        if schedule is not None:
            text += f"\n[{schedule.starts[node]}-{schedule.finishes[node]}]"
        labels[node] = text
    nx.draw_networkx_labels(graph, positions, labels=labels, font_size=5.5, ax=ax)

    ax.set_title(title or f"RCPSP AON network: {instance.name}")
    ax.axis("off")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output
