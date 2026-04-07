"""
HybridRetriever: fixed-policy baseline combining vector retrieval,
graph expansion, and RRF fusion.
This is the baseline the RL agent must learn to beat.
"""
from __future__ import annotations
import numpy as np
from mmkg.retrieval.index import KGIndex
from mmkg.utils import get_logger

log = get_logger(__name__)


class HybridRetriever:
    """
    Three retrieval modes, always combined with RRF fusion.
    The RL agent will learn WHEN to use which mode.
    """

    def __init__(self, index: KGIndex, top_k: int = 10, max_hops: int = 2):
        self.index = index
        self.top_k = top_k
        self.max_hops = max_hops

    # ── Vector retrieval ─────────────────────────────────────────────

    def vector_retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Dense retrieval over FAISS index."""
        k = k or self.top_k
        q_emb = self.index.encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        scores, positions = self.index.index.search(q_emb, k)
        results = []
        for pos, score in zip(positions[0], scores[0]):
            if pos < 0:
                continue
            eid = self.index.entity_ids[pos]
            results.append({
                "entity_id": eid,
                "canonical_name": self.index.entity_names[pos],
                "score": float(score),
                "source": "vector",
                **self.index.entity_map.get(eid, {}),
            })
        return results

    # ── Graph retrieval ───────────────────────────────────────────────

    def graph_retrieve(self, seed_entity_ids: list[str],
                       hops: int | None = None) -> list[dict]:
        """BFS expansion from seed entities up to max_hops."""
        hops = hops or self.max_hops
        # Convert to undirected so we can traverse backwards (e.g. Object -> Subject)
        g = self.index.graph.to_undirected() if hasattr(self.index.graph, "to_undirected") else self.index.graph
        visited: dict[str, int] = {}  # entity_id -> hop distance

        for seed in seed_entity_ids:
            if seed not in g:
                continue
            try:
                lengths = dict(
                    zip(*zip(*[
                        (node, dist)
                        for node, dist in
                        __import__('networkx').single_source_shortest_path_length(
                            g, seed, cutoff=hops
                        ).items()
                    ]))
                ) if False else \
                    __import__('networkx').single_source_shortest_path_length(
                        g, seed, cutoff=hops
                    )
                for node, dist in lengths.items():
                    if node not in visited or dist < visited[node]:
                        visited[node] = dist
            except Exception:
                visited[seed] = 0

        results = []
        for eid, dist in sorted(visited.items(), key=lambda x: x[1]):
            score = 1.0 / (1.0 + dist)
            results.append({
                "entity_id": eid,
                "canonical_name": self.index.entity_map.get(eid, {}).get(
                    "canonical_name", eid),
                "score": score,
                "hop_distance": dist,
                "source": "graph",
                **self.index.entity_map.get(eid, {}),
            })
        return results[:self.top_k * 2]

    # ── RRF fusion ────────────────────────────────────────────────────

    def rrf_fuse(self, *ranked_lists: list[dict],
                 k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion — combines multiple ranked lists
        without any learned parameters.
        Score for entity e = sum over lists L of 1 / (k + rank_in_L(e))
        """
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}

        for ranked in ranked_lists:
            for rank, item in enumerate(ranked):
                eid = item["entity_id"]
                scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank + 1)
                if eid not in meta:
                    meta[eid] = item

        fused = sorted(scores.items(), key=lambda x: -x[1])
        return [
            {**meta[eid], "rrf_score": score, "source": "hybrid"}
            for eid, score in fused[:self.top_k]
        ]

    # ── Full hybrid retrieve ──────────────────────────────────────────

    def retrieve(self, query: str) -> dict:
        """
        Run all three modes and fuse with RRF.
        Returns a dict with all intermediate results for analysis.
        """
        # Step 1: vector retrieval
        vector_hits = self.vector_retrieve(query)

        # Step 2: graph retrieval seeded from top-3 vector hits
        seed_ids = [h["entity_id"] for h in vector_hits[:3]]
        graph_hits = self.graph_retrieve(seed_ids)

        # Step 3: RRF fusion
        fused = self.rrf_fuse(vector_hits, graph_hits)

        return {
            "query": query,
            "vector_hits": vector_hits,
            "graph_hits": graph_hits[:self.top_k],
            "fused": fused,
            "top_entity": fused[0]["canonical_name"] if fused else None,
        }

    # ── Evidence quality check ────────────────────────────────────────

    def evidence_hit(self, query: str, gold_answer: str) -> bool:
        """
        Check if gold_answer appears in the top-k retrieved entity names.
        This is the metric the RL reward is based on.
        """
        result = self.retrieve(query)
        gold_lower = gold_answer.lower()
        return any(
            gold_lower in item["canonical_name"].lower()
            or item["canonical_name"].lower() in gold_lower
            for item in result["fused"]
        )
