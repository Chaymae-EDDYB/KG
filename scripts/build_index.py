#!/usr/bin/env python3
"""Build the FAISS index over all processed documents."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mmkg.retrieval.index import KGIndex

index = KGIndex(encoder_model="all-MiniLM-L6-v2")
index.build(processed_dir="data/processed")
index.save("data/processed/kg_index.faiss")

print(f"\nIndex built:")
print(f"  Nodes in graph: {index.graph.number_of_nodes()}")
print(f"  Edges in graph: {index.graph.number_of_edges()}")
print(f"  Vectors in FAISS: {index.index.ntotal}")
