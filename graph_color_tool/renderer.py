"""
renderer.py
Draws graphs as PNGs with non-overlapping nodes and overlays instructions
directly onto the image so models can be prompted by image-only.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import math
import random

import matplotlib.pyplot as plt
import networkx as nx

# Colours the LLM should use
COLORS4 = ["red", "blue", "green", "yellow"]


def _non_overlapping_layout(G: nx.Graph, tries: int = 200, min_dist: float = 0.15):
    """
    Compute a layout that avoids node overlap as much as practical.
    Start with spring_layout; jitter until min pairwise distance is satisfied.
    """
    pos = nx.spring_layout(G, seed=42, k=1.0 / math.sqrt(max(1, G.number_of_nodes())))
    nodes = list(G.nodes())
    def valid(p):
        coords = [p[n] for n in nodes]
        for i in range(len(coords)):
            xi, yi = coords[i]
            for j in range(i + 1, len(coords)):
                xj, yj = coords[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < (min_dist ** 2):
                    return False
        return True

    if valid(pos):
        return pos

    for _ in range(tries):
        # slight random jitter
        for n in nodes:
            dx = (random.random() - 0.5) * 0.05
            dy = (random.random() - 0.5) * 0.05
            x, y = pos[n]
            pos[n] = (x + dx, y + dy)
        if valid(pos):
            return pos

    return pos  # fallback (rare)


def draw_graph(
    G: nx.Graph,
    out_path: Path,
    *,
    add_instructions: bool = True,
    instructions: str | None = None,
    figsize=(7.5, 8.5),        # a little taller to fit the banner
    dpi: int = 180,
) -> None:
    """
    Render a simple node/edge graph with numbered circular nodes,
    straight 'stick' edges, and (optionally) an instruction banner.
    Guarantees no isolated vertices (assumed ensured by generator).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = G.number_of_nodes()

    if instructions is None:
        # concise, LLM-friendly instructions; they are on the image
        instructions = (
            f"Color this graph using only: Red, Blue, Green, Yellow.\n"
            f"Label exactly {n} vertices as: 'Vertex k: <Color>' for k=1..{n}.\n"
            f"If it cannot be 4-colored, write exactly: UNCOLORABLE"
        )

    pos = _non_overlapping_layout(G)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_axis_off()

    # Edges first (straight segments)
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.0)

    # Nodes as circles
    nx.draw_networkx_nodes(
        G, pos, node_size=800, node_color="#ffffff", edgecolors="#222222", linewidths=2.0, ax=ax
    )

    # Labels: 1..n
    labels = {v: str(v + 1) for v in G.nodes()}  # graph uses 0-based internally
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight="bold", ax=ax)

    # Instruction banner
    if add_instructions:
        # white rounded rectangle banner at the top
        ax.text(
            0.5, 1.03, instructions,
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=11, color="#111111",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#444444", linewidth=1.0)
        )

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
