"""
pipeline.py
Generate → render (with on-image instructions) → LLM → evaluate colourings.
Writes:
  • llm_colourings.json  (raw LLM outputs, colouring as a string or "UNCOLORABLE")
  • graphs.json          (edges for each graph, 1-based)
  • llm_evaluation.json  (same list + isCorrect; final results block with accuracy)
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import re
from typing import TypedDict, Union, List, Dict, Tuple

import networkx as nx

from graph_color_tool.generator import GraphGenerator
from graph_color_tool.renderer import draw_graph, COLORS4
from graph_color_tool.llm import LLMColourer


class GraphRecord(TypedDict):
    graph_img: str
    colouring: str  # stored as a string like "{1: red, 2: blue, ...}" or "UNCOLORABLE"


def _edges_1based(G: nx.Graph) -> List[Tuple[int, int]]:
    """Return edges as 1-based pairs (u,v) with u < v for stable ordering."""
    edges = []
    for u, v in G.edges():
        a, b = (u + 1, v + 1)
        if a > b:
            a, b = b, a
        edges.append((a, b))
    edges.sort()
    return edges


# ---------- Parsing & validation ----------

_COLOR_SET = {"red", "blue", "green", "yellow"}
_PAIR_RE = re.compile(r"(\d+)\s*:\s*([A-Za-z]+)")

def parse_colour_string(s: str, n_vertices: int) -> Dict[int, str] | None:
    """
    Parse a string like "{1: red, 2: blue, ...}" into {1:'red', 2:'blue', ...}.
    - Case-insensitive colours; normalized to lowercase
    - Returns None if the string is "uncolorable" or cannot be parsed.
    - Extra vertices outside 1..n are ignored.
    """
    if not isinstance(s, str):
        return None
    if s.strip().lower().startswith("uncolorable"):
        return None

    mapping: Dict[int, str] = {}
    for m in _PAIR_RE.finditer(s):
        k = int(m.group(1))
        colour = m.group(2).lower()
        mapping[k] = colour

    if not mapping:
        return None

    mapping = {k: v for k, v in mapping.items() if 1 <= k <= n_vertices}
    if not mapping:
        return None
    return mapping


def check_colouring(
    edges_1: List[Tuple[int, int]],
    colouring: Dict[int, str] | None,
    n_vertices: int
) -> tuple[bool, List[Tuple[int, int]]]:
    """
    Return (is_correct, violating_edges).

    is_correct iff:
      - Every vertex 1..n has a colour in {red, blue, green, yellow}
      - For every edge (u,v), colouring[u] != colouring[v]

    violating_edges is a list of (u, v) vertex pairs (1-based, u < v) where both
    endpoints share the same colour. Missing/invalid colours also make is_correct
    False but are not listed as edges.
    """
    if colouring is None:
        return False, []

    # All vertices present with valid colour
    for v in range(1, n_vertices + 1):
        c = colouring.get(v)
        if c not in _COLOR_SET:
            return False, []

    # Find conflicting edges; keep actual vertex pairs
    bad_edges: List[Tuple[int, int]] = []
    for (u, v) in edges_1:
        if colouring.get(u) == colouring.get(v):
            bad_edges.append((u, v))

    return (len(bad_edges) == 0), bad_edges


# ---------- Batch pipeline ----------

def run_batch(
    count: int = 5,
    n_vertices: int = 8,
    edge_prob: float = 0.3,
    fraction_uncolorable: float = 0.0,
    provider: str | None = None,
    model: str | None = None,
    device: int = -1,
    base_url: str | None = None,  # ignored; kept for CLI compatibility
) -> Path:
    """
    Make `count` graphs with `n_vertices`, draw PNGs with embedded instructions,
    optionally query an LLM *with image only*, then evaluate the colourings.
    Prints CORRECT/WRONG per graph and writes evaluation JSON with accuracy.
    """
    gen = GraphGenerator(
        n_vertices=n_vertices, edge_prob=edge_prob,
        fraction_uncolorable=fraction_uncolorable, seed=42
    )

    out_dir = Path("outputs") / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm_records: List[GraphRecord] = []
    graphs_meta: List[dict] = [] 

    colourer = LLMColourer(provider, model, device) if provider else None

    for i in range(count):
        G = gen()
        img_name = f"graph_{i:03d}.png"
        img_path = out_dir / img_name
        draw_graph(G, img_path, add_instructions=False)
        edges1 = _edges_1based(G)
        graphs_meta.append({
            "graph_img": img_name,
            "n": n_vertices,
            "edges": edges1,
        })

        if colourer:
            print(f"[INFO] Prompting {provider.upper()} with image {img_name}")
            ans = colourer.prompt_for_colouring(img_path)

            if isinstance(ans, dict):
                parts = [f"{k}: {ans[k]}" for k in sorted(ans.keys())]
                s = "{ " + ", ".join(parts) + " }"
            else:
                s = ans

            print(f"[RESULT] {provider.upper()} → {s}")

            mapping = parse_colour_string(s, n_vertices)
            ok, bad_pairs = check_colouring(edges1, mapping, n_vertices)
            if ok:
                print(f"[VALIDATION] -> CORRECT\n")
            else:
                if bad_pairs:
                    pretty = ", ".join(f"EDGE BETWEEN {u} and {v}" for (u, v) in bad_pairs)
                    print(f"[VALIDATION] -> WRONG ({pretty})\n")
                else:
                    print(f"[VALIDATION] -> WRONG\n")

            llm_records.append(GraphRecord({"graph_img": img_name, "colouring": s}))
        else:
            llm_records.append(GraphRecord({"graph_img": img_name, "colouring": ""}))

    with open(out_dir / "llm_colourings.json", "w") as f:
        json.dump(llm_records, f, indent=2)
    with open(out_dir / "graphs.json", "w") as f:
        json.dump(graphs_meta, f, indent=2)

    evaluated_list, accuracy,correct,wrong,total = _evaluate_records(llm_records, graphs_meta)

    evaluated_with_results = evaluated_list + [{"results": {"accuracy": accuracy, "total": len(evaluated_list)}}]
    with open(out_dir / "llm_evaluation.json", "w") as f:
        json.dump(evaluated_with_results, f, indent=2)

    print(f"Evaluation accuracy: {accuracy:.3f}")
    print(f"Correct: {correct}, Wrong: {wrong}, Total: {total}")
    print(f"Outputs saved to {out_dir}")
    return out_dir


def _evaluate_records(
    llm_records: List[GraphRecord],
    graphs_meta: List[dict],
) -> tuple[list[dict], float]:
    """Return (evaluated_items, accuracy)."""
    meta_by_img = {m["graph_img"]: m for m in graphs_meta}

    evaluated: List[dict] = []
    correct = 0

    for item in llm_records:
        img = item["graph_img"]
        colouring_text = item["colouring"]
        meta = meta_by_img.get(img)
        if not meta:
            evaluated.append({
                "graph_img": img,
                "colouring": colouring_text,
                "isCorrect": False
            })
            continue

        n = int(meta["n"])
        edges1 = [tuple(e) for e in meta["edges"]]

        mapping = parse_colour_string(colouring_text, n)
        ok, _bad_pairs = check_colouring(edges1, mapping, n)

        evaluated.append({
            "graph_img": img,
            "colouring": colouring_text,
            "isCorrect": bool(ok)
        })
        if ok:
            correct += 1

    acc = (correct / len(evaluated)) if evaluated else 0.0
    return evaluated, acc, correct, len(evaluated) - correct, len(evaluated)
