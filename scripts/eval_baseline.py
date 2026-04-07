#!/usr/bin/env python3
"""
Evaluate the fixed hybrid retrieval baseline.
Creates 20 test queries from your existing documents and measures
evidence hit rate — the number that the RL agent must beat.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mmkg.retrieval.index import KGIndex
from mmkg.retrieval.baseline import HybridRetriever

# Load the index
index = KGIndex()
index.load("data/processed/kg_index.faiss")
retriever = HybridRetriever(index, top_k=10, max_hops=2)

# Hand-crafted test queries matched to your documents
# query -> expected gold entity name
TEST_QUERIES = [
    ("Who developed the theory of relativity?", "albert einstein"),
    ("Where was Einstein born?", "ulm"),
    ("Which university did Einstein work at?", "princeton university"),
    ("Who discovered polonium?", "marie curie"),
    ("What did Newton invent?", "calculus"),
    ("Who proposed evolution by natural selection?", "charles darwin"),
    ("Which scientist invented alternating current?", "nikola tesla"),
    ("Where did Stephen Hawking work?", "cambridge university"),
    ("Who formulated the uncertainty principle?", "werner heisenberg"),
    ("What is the Schrodinger equation related to?", "quantum mechanics"),
    ("Which physicist worked at Caltech?", "richard feynman"),
    ("Who originated quantum theory?", "max planck"),
    ("Where was Niels Bohr from?", "copenhagen"),
    ("What did Darwin publish in 1859?", "on the origin of species"),
    ("Which element did Curie discover?", "polonium"),
    ("Who held the Lucasian professorship?", "stephen hawking"),
    ("Where did Tesla conduct experiments?", "wardenclyffe"),
    ("Who received the Nobel Prize in 1932?", "werner heisenberg"),
    ("What did Feynman contribute to?", "quantum electrodynamics"),
    ("Where was Newton born?", "woolsthorpe"),
]

print(f"Running {len(TEST_QUERIES)} test queries...\n")
hits = 0
vector_only_hits = 0
graph_only_hits = 0

for query, gold in TEST_QUERIES:
    result = retriever.retrieve(query)

    # Check hit in fused (hybrid) results
    hybrid_hit = retriever.evidence_hit(query, gold)

    # Check hit in vector-only results
    v_hit = any(
        gold in r["canonical_name"].lower() or r["canonical_name"].lower() in gold
        for r in result["vector_hits"]
    )

    # Check hit in graph-only results
    g_hit = any(
        gold in r["canonical_name"].lower() or r["canonical_name"].lower() in gold
        for r in result["graph_hits"]
    )

    if hybrid_hit:
        hits += 1
    if v_hit:
        vector_only_hits += 1
    if g_hit:
        graph_only_hits += 1

    status = "HIT" if hybrid_hit else "MISS"
    top = result["top_entity"] or "none"
    print(f"[{status}] {query[:50]:<50} | gold={gold:<25} | top={top}")

print(f"\n=== Baseline Results ===")
print(f"Vector-only hit rate:  {vector_only_hits}/{len(TEST_QUERIES)} "
      f"= {vector_only_hits/len(TEST_QUERIES)*100:.1f}%")
print(f"Graph-only hit rate:   {graph_only_hits}/{len(TEST_QUERIES)} "
      f"= {graph_only_hits/len(TEST_QUERIES)*100:.1f}%")
print(f"Hybrid (RRF) hit rate: {hits}/{len(TEST_QUERIES)} "
      f"= {hits/len(TEST_QUERIES)*100:.1f}%")
print()
print("This is the number your RL agent must beat.")
print("Target: RL agent achieves same hit rate in fewer average steps.")
