from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from mmkg.adapters.base import BaseAdapter
from mmkg.schemas.core import (
    GraphDocument, Entity, Relation, Evidence,
    EntityType, Modality, SourceType
)
from mmkg.utils import get_logger

log = get_logger(__name__)


class StubAdapter(BaseAdapter):
    """Returns synthetic data. Used for skeleton testing only."""

    def extract(
        self,
        doc_id: str,
        text: str,
        image_paths: Optional[List[Path]] = None,
    ) -> GraphDocument:
        log.info("StubAdapter.extract called for doc_id={d}", d=doc_id)

        ev_text = Evidence(
            source_doc_id=doc_id,
            source_type=SourceType.TEXT_SPAN,
            content=text[:100],
            model_name="stub",
        )
        e1 = Entity(
            canonical_name="Stub Entity A",
            entity_type=EntityType.PERSON,
            modality=Modality.TEXT,
            source_system="stub",
            evidence_ids=[ev_text.id],
        )
        e2 = Entity(
            canonical_name="Stub Entity B",
            entity_type=EntityType.ORGANIZATION,
            modality=Modality.TEXT,
            source_system="stub",
            evidence_ids=[ev_text.id],
        )
        rel = Relation(
            subject_id=e1.id,
            predicate="affiliated_with",
            object_id=e2.id,
            confidence=0.9,
            source_system="stub",
            evidence_ids=[ev_text.id],
        )

        evidences = [ev_text]
        entities = [e1, e2]
        relations = [rel]

        if image_paths:
            for img_path in image_paths:
                ev_img = Evidence(
                    source_doc_id=doc_id,
                    source_type=SourceType.IMAGE_REGION,
                    content="stub_visual_object",
                    image_path=str(img_path),
                    bbox=[10.0, 10.0, 100.0, 100.0],
                    model_name="stub",
                    confidence=0.8,
                )
                e_img = Entity(
                    canonical_name="stub_visual_object",
                    entity_type=EntityType.OBJECT,
                    modality=Modality.IMAGE,
                    source_system="stub",
                    evidence_ids=[ev_img.id],
                )
                rel_cross = Relation(
                    subject_id=e1.id,
                    predicate="depicted_as",
                    object_id=e_img.id,
                    confidence=0.75,
                    source_system="stub-fusion",
                    evidence_ids=[ev_img.id],
                )
                evidences.append(ev_img)
                entities.append(e_img)
                relations.append(rel_cross)

        return GraphDocument(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            evidences=evidences,
            pipeline_stage="stub",
        )
