"""
pipeline.py
Generate → render → greedy + LLM colouring → separate logs.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import networkx as nx
from typing import TypedDict, Union, List

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
    base_url: str | None = None,
) -> Path:
    """
    Build `count` random graphs, draw PNGs, then:
      • Always do greedy → greedy_colourings.json
      • If provider set, also prompt LLM with URL/base64 → llm_colourings.json
    """
    gen = GraphGenerator(
        n_vertices=n_vertices,
        edge_prob=edge_prob,
        fraction_uncolorable=fraction_uncolorable,
        seed=42,
    )
    out_dir = Path("outputs") / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    greedy_records: List[GraphRecord] = []
    llm_records:    List[GraphRecord] = []

    colourer = (
        LLMColourer(provider, model, device)
        if provider is not None
        else None
    )

    for i in range(count):
        G = gen()
        img_name = f"graph_{i:03d}.png"
        img_path = out_dir / img_name
        draw_graph(G, img_path)

        # 1) Greedy
        greedy = _greedy_colour(G)
        greedy_records.append(GraphRecord({
            "graph_img": img_name,
            "colouring": greedy
        }))

        # 2) LLM
        if colourer:
            # decide what to pass in
            if provider in ("openai", "gemini"):
                if not base_url:
                    raise ValueError("Must supply --base-url for image‐URL providers")
                ref = f"{base_url.rstrip('/')}/{out_dir.name}/{img_name}"
            else:
                ref = img_path

            print(f"[INFO] Prompting {provider.upper()} with {ref}")
            llm_col = colourer.prompt_for_colouring(ref)
            print(f"[RESULT] {provider.upper()} → {llm_col}\n")
            llm_records.append(GraphRecord({
                "graph_img": img_name,
                "colouring": llm_col
            }))

    # dump logs
    with open(out_dir / "greedy_colourings.json", "w") as f:
        json.dump(greedy_records, f, indent=2)
    with open(out_dir / "llm_colourings.json", "w") as f:
        json.dump(llm_records, f, indent=2)

    return out_dir


if __name__ == "__main__":
    run_batch()
