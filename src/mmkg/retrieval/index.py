"""
KGIndex: loads all processed GraphDocuments and builds a unified
in-memory graph + FAISS index over entity embeddings.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from mmkg.schemas.core import GraphDocument, Entity
from mmkg.utils import get_logger

log = get_logger(__name__)


class KGIndex:
    """
    Unified index over all processed GraphDocuments.
    Builds:
      - a NetworkX DiGraph with all entities and relations
      - a FAISS flat inner-product index over entity name embeddings
      - a mapping from FAISS position -> entity id and canonical name
    """

    def __init__(self, encoder_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(encoder_model)
        self.graph = nx.DiGraph()
        self.entity_ids: list[str] = []      # position -> entity id
        self.entity_names: list[str] = []    # position -> canonical name
        self.entity_map: dict[str, dict] = {}  # entity id -> full entity dict
        self.index: faiss.IndexFlatIP | None = None
        self.is_built = False

    def build(self, processed_dir: str | Path = "data/processed") -> None:
        """Load all GraphDocument JSONs and build the unified index."""
        processed_dir = Path(processed_dir)
        json_files = sorted(processed_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {processed_dir}")

        global_kg_file = processed_dir / "global_kg.json"
        if global_kg_file in json_files:
            log.info("Found global_kg.json! Using it exclusively to prevent individual node duplication.")
            json_files = [global_kg_file]
        else:
            json_files = [f for f in json_files if "kg_index" not in f.name]

        global_kg_file = processed_dir / "global_kg.json"
        if global_kg_file in json_files:
            log.info("Found global_kg.json, using it exclusively.")
            json_files = [global_kg_file]
        else:
            # Avoid loading faiss metadata json
            json_files = [f for f in json_files if "kg_index" not in f.name]

        log.info("Building KG index from {n} documents", n=len(json_files))

        all_entities: list[Entity] = []

        for json_file in json_files:
            try:
                raw = json.loads(json_file.read_text())
                doc = GraphDocument.model_validate(raw)
            except Exception as e:
                log.warning("Skipping {f}: {e}", f=json_file.name, e=e)
                continue

            # Add entities to graph
            for ent in doc.entities:
                if ent.id not in self.graph:
                    self.graph.add_node(
                        ent.id,
                        canonical_name=ent.canonical_name,
                        entity_type=ent.entity_type.value,
                        modality=ent.modality.value,
                        source_system=ent.source_system,
                        doc_id=doc.doc_id,
                    )
                    self.entity_map[ent.id] = {
                        "id": ent.id,
                        "canonical_name": ent.canonical_name,
                        "entity_type": ent.entity_type.value,
                        "modality": ent.modality.value,
                        "doc_id": doc.doc_id,
                    }
                    all_entities.append(ent)

            # Add relations as edges
            for rel in doc.relations:
                if rel.subject_id in self.graph and rel.object_id in self.graph:
                    self.graph.add_edge(
                        rel.subject_id,
                        rel.object_id,
                        predicate=rel.predicate,
                        confidence=rel.confidence,
                        source_system=rel.source_system,
                    )

        log.info("Graph: {n} nodes, {e} edges", n=self.graph.number_of_nodes(),
                 e=self.graph.number_of_edges())

        # Build FAISS index
        if not all_entities:
            raise ValueError("No entities found across all documents")

        names = [e.canonical_name for e in all_entities]
        self.entity_ids = [e.id for e in all_entities]
        self.entity_names = names

        log.info("Encoding {n} entity names...", n=len(names))
        embeddings = self.encoder.encode(
            names, normalize_embeddings=True, show_progress_bar=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.is_built = True

        log.info("FAISS index built: {n} vectors, dim={d}", n=len(names), d=dim)

    def save(self, path: str | Path = "data/processed/kg_index.faiss") -> None:
        """Save the FAISS index to disk."""
        path = Path(path)
        faiss.write_index(self.index, str(path))
        # Save the entity id mapping and graph alongside
        meta_path = path.with_suffix(".json")
        import json
        meta_path.write_text(json.dumps({
            "entity_ids": self.entity_ids,
            "entity_names": self.entity_names,
            "entity_map": self.entity_map,
            "graph_data": nx.node_link_data(self.graph)
        }, indent=2))
        log.info("Saved index to {p}", p=path)

    def load(self, path: str | Path = "data/processed/kg_index.faiss") -> None:
        """Load a previously saved FAISS index."""
        path = Path(path)
        self.index = faiss.read_index(str(path))
        meta_path = path.with_suffix(".json")
        import json
        meta = json.loads(meta_path.read_text())
        self.entity_ids = meta["entity_ids"]
        self.entity_names = meta["entity_names"]
        self.entity_map = meta["entity_map"]
        if "graph_data" in meta:
            self.graph = nx.node_link_graph(meta["graph_data"])
        self.is_built = True
        log.info("Loaded index: {n} vectors, {g} graph nodes", n=len(self.entity_ids), g=self.graph.number_of_nodes())
