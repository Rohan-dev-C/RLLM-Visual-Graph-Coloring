# scripts/generate_dataset.py
from __future__ import annotations
import argparse

from graph_color_tool.pipeline import run_batch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="number of graphs to generate")
    p.add_argument("--vertices", type=int, default=8, help="number of vertices per graph")
    p.add_argument("--p", type=float, default=0.3, help="edge probability")
    p.add_argument("--fraction-uncolorable", type=float, default=0.0, help="fraction of forced uncolorable graphs")
    p.add_argument("--provider", type=str, default=None, help="llm provider (openai, gemini, claude, deepseek, llama)")
    p.add_argument("--model", type=str, default=None, help="model id/name for the provider")
    p.add_argument("--device", type=int, default=-1, help="device index for local models")
    p.add_argument("--base-url", type=str, default=None, help="(kept for compatibility; ignored)")
    args = p.parse_args()

    out = run_batch(
        count=args.n,
        n_vertices=args.vertices,
        edge_prob=args.p,
        fraction_uncolorable=args.fraction_uncolorable,
        provider=args.provider,
        model=args.model,
        device=args.device,
        base_url=args.base_url,
    )
    print(f"Done. See {out}")

if __name__ == "__main__":
    main()
