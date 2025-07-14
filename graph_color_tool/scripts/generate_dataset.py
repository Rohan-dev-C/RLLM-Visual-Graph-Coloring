"""
generate_dataset.py
CLI for batch generation + multimodal colouring.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from graph_color_tool.pipeline import run_batch

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n",            type=int,   default=5,   help="Number of graphs")
    p.add_argument("--vertices",     type=int,   default=8,   help="Vertices per graph")
    p.add_argument("--p",            type=float, default=0.3, help="Edge probability")
    p.add_argument("--hard-fraction",type=float, default=0.0,
                   help="Fraction embedding a K5 (uncolourable)")
    p.add_argument("--provider",     type=str,
                   choices=["openai","claude","gemini","deepseek","llama"],
                   default=None, help="Backend to prompt")
    p.add_argument("--model",        type=Path,  default=None,
                   help="Model ID/path (for llama, path; else provider’s ID)")
    p.add_argument("--device",       type=int,   default=-1,
                   help="-1 for CPU, ≥0 for GPU")
    p.add_argument("--base-url",     type=str,   default=None,
                   help="Public URL root serving outputs/ (required for OpenAI)")
    args = p.parse_args()

    out = run_batch(
        count=args.n,
        n_vertices=args.vertices,
        edge_prob=args.p,
        fraction_uncolorable=args.hard_fraction,
        provider=args.provider,
        model=str(args.model) if args.model else None,
        device=args.device,
        base_url=args.base_url,
    )
    print("Dataset & colourings saved to", out)

if __name__ == "__main__":
    main()
