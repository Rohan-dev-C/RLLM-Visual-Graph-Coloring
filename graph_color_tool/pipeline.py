"""
pipeline.py
Generate → render (with on-image instructions) → greedy + LLM → separate logs.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
from typing import TypedDict, Union, List

import networkx as nx

from graph_color_tool.generator import GraphGenerator
from graph_color_tool.renderer import draw_graph, COLORS4
from graph_color_tool.llm import LLMColourer


class GraphRecord(TypedDict):
    graph_img: str
    colouring: Union[dict[int, str], str]


def _greedy_colour(G: nx.Graph) -> Union[dict[int, str], str]:
    raw = nx.coloring.greedy_color(G, strategy="largest_first")
    if max(raw.values(), default=-1) >= len(COLORS4):
        return "uncolorable"
    return {v + 1: COLORS4[c] for v, c in raw.items()}


def run_batch(
    count: int = 5,
    n_vertices: int = 8,
    edge_prob: float = 0.3,
    fraction_uncolorable: float = 0.0,
    provider: str | None = None,
    model: str | None = None,
    device: int = -1,
    base_url: str | None = None,  # ignored now; kept for CLI compatibility
) -> Path:
    """
    Make `count` graphs with `n_vertices`, draw PNGs with embedded instructions,
    greedy-colour each, and optionally query an LLM *with image only*.
    """
    gen = GraphGenerator(
        n_vertices=n_vertices, edge_prob=edge_prob,
        fraction_uncolorable=fraction_uncolorable, seed=42
    )

    out_dir = Path("outputs") / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    greedy_records: List[GraphRecord] = []
    llm_records: List[GraphRecord] = []
    colourer = LLMColourer(provider, model, device) if provider else None

    for i in range(count):
        G = gen()
        img_name = f"graph_{i:03d}.png"
        img_path = out_dir / img_name

        # render with instructions on the image
        draw_graph(G, img_path, add_instructions=True)

        # Greedy always
        greedy = _greedy_colour(G)
        greedy_records.append(GraphRecord({"graph_img": img_name, "colouring": greedy}))

        # LLM (image only)
        if colourer:
            print(f"[INFO] Prompting {provider.upper()} with image {img_name}")
            llm_col = colourer.prompt_for_colouring(img_path)
            print(f"[RESULT] {provider.upper()} → {llm_col}\n")
            llm_records.append(GraphRecord({"graph_img": img_name, "colouring": llm_col}))

    with open(out_dir / "greedy_colourings.json", "w") as f:
        json.dump(greedy_records, f, indent=2)
    with open(out_dir / "llm_colourings.json", "w") as f:
        json.dump(llm_records, f, indent=2)

    return out_dir


if __name__ == "__main__":
    run_batch()
