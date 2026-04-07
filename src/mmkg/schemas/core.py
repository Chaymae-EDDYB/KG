from __future__ import annotations
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


class SourceType(str, Enum):
    TEXT_SPAN = "text_span"
    IMAGE_REGION = "image_region"
    SCENE_GRAPH = "scene_graph"
    MODEL_OUTPUT = "model_output"


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    OBJECT = "OBJECT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    ATTRIBUTE = "ATTRIBUTE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:8]}")
    source_doc_id: str
    source_type: SourceType
    content: str = ""
    bbox: Optional[List[float]] = None
    page: Optional[int] = None
    image_path: Optional[str] = None
    model_name: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}


class Entity(BaseModel):
    id: str = Field(default_factory=lambda: f"ent-{uuid.uuid4().hex[:8]}")
    canonical_name: str
    entity_type: EntityType = EntityType.UNKNOWN
    modality: Modality = Modality.TEXT
    aliases: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    source_system: str = "unknown"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def name_not_empty(self) -> "Entity":
        if not self.canonical_name.strip():
            raise ValueError("canonical_name must not be empty")
        return self


class Relation(BaseModel):
    id: str = Field(default_factory=lambda: f"rel-{uuid.uuid4().hex[:8]}")
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source_system: str = "unknown"
    evidence_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def confidence_in_range(self) -> "Relation":
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}")
        return self


class GraphDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: f"doc-{uuid.uuid4().hex[:8]}")
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    evidences: List[Evidence] = Field(default_factory=list)
    pipeline_stage: str = "unknown"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    model_config = {"extra": "forbid"}

    def entity_by_id(self, eid: str) -> Optional[Entity]:
        return next((e for e in self.entities if e.id == eid), None)

    def relations_for_entity(self, eid: str) -> List[Relation]:
        return [r for r in self.relations
                if r.subject_id == eid or r.object_id == eid]

    def summary(self) -> str:
        return (f"GraphDocument(doc_id={self.doc_id!r}, "
                f"entities={len(self.entities)}, "
                f"relations={len(self.relations)}, "
                f"stage={self.pipeline_stage!r})")
