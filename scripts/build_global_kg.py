#!/usr/bin/env python3
"""
Build a single unified global KG by merging all per-document GraphDocuments.
This creates cross-document edges that make graph retrieval meaningful.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

processed = Path("data/processed")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Load all documents
all_docs = []
for f in sorted(processed.glob("*.json")):
    if "index" in f.name or "global" in f.name:
        continue
    try:
        all_docs.append(json.loads(f.read_text()))
    except:
        pass

print(f"Loaded {len(all_docs)} documents")

# Collect all entities with embeddings
all_entities = []
for doc in all_docs:
    for ent in doc["entities"]:
        all_entities.append({**ent, "doc_id": doc["doc_id"]})

# Deduplicate across documents using embedding similarity
# Two entities from different documents are "the same" if cosine sim > 0.85
names = [e["canonical_name"] for e in all_entities]
print(f"Encoding {len(names)} entity names for cross-document deduplication...")
embeddings = encoder.encode(names, normalize_embeddings=True, show_progress_bar=True)

MERGE_THRESHOLD = 0.85
# Build canonical id map: entity_id -> canonical_id (the surviving node)
canonical_map = {}  # original_id -> canonical_id
canonical_entities = {}  # canonical_id -> entity dict

for i, ent in enumerate(all_entities):
    eid = ent["id"]
    if eid in canonical_map:
        continue  # already merged

    # Find all entities that should merge with this one
    sims = embeddings @ embeddings[i]  # cosine similarity to all others
    merge_candidates = [
        j for j, s in enumerate(sims)
        if s >= MERGE_THRESHOLD and j != i
        and all_entities[j]["id"] not in canonical_map
        and all_entities[j]["doc_id"] != ent["doc_id"]  # only cross-document merges
    ]

    canonical_map[eid] = eid
    canonical_entities[eid] = ent

    for j in merge_candidates:
        other_id = all_entities[j]["id"]
        canonical_map[other_id] = eid  # redirect to this canonical node
        # Add the other doc_id as evidence of cross-document grounding in metadata
        if "cross_doc_ids" not in canonical_entities[eid]["metadata"]:
            canonical_entities[eid]["metadata"]["cross_doc_ids"] = []
        canonical_entities[eid]["metadata"]["cross_doc_ids"].append(all_entities[j]["doc_id"])

print(f"Before merge: {len(all_entities)} entities")
print(f"After merge:  {len(canonical_entities)} unique entities")
print(f"Cross-document merges: {sum(1 for e in canonical_entities.values() if 'cross_doc_ids' in e.get('metadata', {}))}")

# Build global relation list with canonical ids
all_relations = []
for doc in all_docs:
    for rel in doc["relations"]:
        canonical_subj = canonical_map.get(rel["subject_id"], rel["subject_id"])
        canonical_obj  = canonical_map.get(rel["object_id"],  rel["object_id"])
        if canonical_subj in canonical_entities and canonical_obj in canonical_entities:
            rel_copy = dict(rel)
            rel_copy["metadata"] = dict(rel.get("metadata", {}))
            rel_copy["metadata"]["source_doc"] = doc["doc_id"]
            rel_copy["subject_id"] = canonical_subj
            rel_copy["object_id"] = canonical_obj
            all_relations.append(rel_copy)

# Remove duplicate relations (same subject, predicate, object)
seen_rels = set()
deduped_relations = []
for r in all_relations:
    key = (r["subject_id"], r["predicate"], r["object_id"])
    if key not in seen_rels:
        seen_rels.add(key)
        deduped_relations.append(r)

print(f"Total relations (before dedup): {len(all_relations)}")
print(f"Total relations (after dedup):  {len(deduped_relations)}")

# Strip the temporary 'doc_id' field before serializing
final_entities = []
for e in canonical_entities.values():
    e_copy = dict(e)
    e_copy.pop("doc_id", None)
    final_entities.append(e_copy)

# Save the global KG
global_kg = {
    "entities": final_entities,
    "relations": deduped_relations,
    "metadata": {
        "source_documents": len(all_docs),
        "total_entities": len(canonical_entities),
        "total_relations": len(deduped_relations),
    }
}
out_path = processed / "global_kg.json"
out_path.write_text(json.dumps(global_kg, indent=2))
print(f"\nSaved global KG to {out_path}")

# Show some cross-document merges to verify
print("\nSample cross-document entity merges:")
for eid, ent in list(canonical_entities.items())[:5]:
    if "cross_doc_ids" in ent.get("metadata", {}):
        print(f"  '{ent['canonical_name']}' appears in: {ent['doc_id']} + {ent['metadata']['cross_doc_ids']}")
