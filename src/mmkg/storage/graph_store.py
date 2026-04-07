from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
import networkx as nx
from mmkg.schemas.core import GraphDocument
from mmkg.utils import get_logger

log = get_logger(__name__)


class GraphStore:
    def __init__(self, output_dir: Union[str, Path] = "data/processed", fmt: str = "both"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fmt = fmt

    def save(self, doc: GraphDocument) -> None:
        if self.fmt in ("json", "both"):
            path = self.output_dir / f"{doc.doc_id}.json"
            path.write_text(doc.model_dump_json(indent=2))
            log.info("Saved JSON: {p}", p=path)
        if self.fmt in ("graphml", "both"):
            g = self._to_networkx(doc)
            path = self.output_dir / f"{doc.doc_id}.graphml"
            nx.write_graphml(g, str(path))
            log.info("Saved GraphML: {p}", p=path)

    def load(self, doc_id: str) -> Optional[GraphDocument]:
        path = self.output_dir / f"{doc_id}.json"
        if not path.exists():
            return None
        return GraphDocument.model_validate(json.loads(path.read_text()))

    def _to_networkx(self, doc: GraphDocument) -> nx.DiGraph:
        g = nx.DiGraph()
        g.graph["doc_id"] = doc.doc_id
        for e in doc.entities:
            g.add_node(e.id, canonical_name=e.canonical_name,
                       entity_type=e.entity_type.value,
                       modality=e.modality.value,
                       source_system=e.source_system)
        for r in doc.relations:
            g.add_edge(r.subject_id, r.object_id,
                       key=r.id, predicate=r.predicate,
                       confidence=r.confidence,
                       source_system=r.source_system)
        return g
