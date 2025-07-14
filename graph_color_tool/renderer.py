"""
renderer.py
Render NetworkX graphs to clearer PNGs.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

COLORS4: dict[int, str] = {0: "red", 1: "blue", 2: "green", 3: "yellow"}


def draw_graph(G: nx.Graph, out_path: Path, layout_seed: int = 42) -> None:
    """Draw G with ample spacing and save to out_path (PNG)."""
    pos = nx.spring_layout(G, seed=layout_seed, k=0.9, iterations=100)

    plt.figure(figsize=(5, 5))       
    nx.draw_networkx_nodes(G, pos,
                           node_color="white",
                           edgecolors="black",
                           node_size=600)        
    nx.draw_networkx_edges(G, pos, width=1.5)
    nx.draw_networkx_labels(G, pos,
                            {v: v + 1 for v in G.nodes()},
                            font_size=12,
                            font_weight="bold")
    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
