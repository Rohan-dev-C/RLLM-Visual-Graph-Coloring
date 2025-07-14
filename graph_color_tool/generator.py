"""
generator.py
Random graph generator.
Guarantees:
  • Graph is connected (no isolated vertices or disjoint components)
  • Optionally embeds a K5 clique so the graph is NOT 4-colourable.
"""
from __future__ import annotations
import random
import networkx as nx


class GraphGenerator:
    """
    Parameters
    ----------
    n_vertices : int
        Number of vertices in each graph (must be ≥5 to embed a K5).
    edge_prob : float
        Probability p for G(n, p) edges (outside any forced K5).
    fraction_uncolorable : float
        In [0,1]. Fraction of generated graphs that must contain a
        K5 clique (therefore need ≥5 colours).
    seed : int | None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_vertices: int = 8,
        edge_prob: float = 0.3,
        fraction_uncolorable: float = 0.0,
        seed: int | None = None,
    ) -> None:
        assert n_vertices >= 5, "n_vertices must be ≥5 to embed a K5 clique"
        assert 0.0 <= fraction_uncolorable <= 1.0, "fraction_uncolorable in [0,1]"
        self.n_vertices = n_vertices
        self.edge_prob = edge_prob
        self.frac_uncol = fraction_uncolorable
        self.rng = random.Random(seed)

    def _embed_k5(self, G: nx.Graph) -> None:
        """Connect five random vertices into a K5 clique in-place."""
        nodes = self.rng.sample(list(G.nodes()), 5)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                G.add_edge(u, v)

    def __call__(self) -> nx.Graph:
        """
        Generate graphs until one is connected (no isolates) and,
        if requested, embeds a K5 clique to force ≥5-colourability.
        """
        while True:
            seed = self.rng.randint(0, 1_000_000_000)
            G = nx.gnp_random_graph(self.n_vertices, self.edge_prob, seed=seed)
            if self.rng.random() < self.frac_uncol:
                self._embed_k5(G)
            if nx.is_connected(G):
                return G
