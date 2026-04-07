import pytest
from mmkg.schemas.core import (
    Entity, EntityType, Evidence, GraphDocument,
    Modality, Relation, SourceType
)

class TestEvidence:
    def test_minimal(self):
        ev = Evidence(source_doc_id="d1", source_type=SourceType.TEXT_SPAN)
        assert ev.id.startswith("ev-")

    def test_image_region(self):
        ev = Evidence(source_doc_id="d1", source_type=SourceType.IMAGE_REGION,
                      bbox=[0.0, 0.0, 100.0, 100.0], confidence=0.9)
        assert ev.bbox == [0.0, 0.0, 100.0, 100.0]

class TestEntity:
    def test_minimal(self):
        e = Entity(canonical_name="Paris")
        assert e.modality == Modality.TEXT

    def test_empty_name_raises(self):
        with pytest.raises(Exception):
            Entity(canonical_name="  ")

    def test_image_entity(self):
        e = Entity(canonical_name="dog", modality=Modality.IMAGE,
                   entity_type=EntityType.OBJECT)
        assert e.modality == Modality.IMAGE

class TestRelation:
    def test_confidence_out_of_range(self):
        with pytest.raises(Exception):
            Relation(subject_id="a", predicate="p", object_id="b", confidence=2.0)

    def test_cross_modal_predicate(self):
        r = Relation(subject_id="e1", predicate="depicted_as",
                     object_id="e2", source_system="fusion")
        assert r.predicate == "depicted_as"

class TestGraphDocument:
    def _make(self):
        ev = Evidence(source_doc_id="d1", source_type=SourceType.TEXT_SPAN)
        e1 = Entity(canonical_name="London", entity_type=EntityType.LOCATION,
                    evidence_ids=[ev.id])
        e2 = Entity(canonical_name="UK", entity_type=EntityType.LOCATION,
                    evidence_ids=[ev.id])
        r = Relation(subject_id=e1.id, predicate="located_in", object_id=e2.id)
        return GraphDocument(doc_id="d1", entities=[e1, e2],
                             relations=[r], evidences=[ev], pipeline_stage="test")

    def test_entity_by_id(self):
        gd = self._make()
        assert gd.entity_by_id(gd.entities[0].id) is not None

    def test_missing_entity_returns_none(self):
        gd = self._make()
        assert gd.entity_by_id("nonexistent") is None

    def test_json_roundtrip(self):
        gd = self._make()
        gd2 = GraphDocument.model_validate_json(gd.model_dump_json())
        assert gd2.doc_id == gd.doc_id
        assert len(gd2.entities) == 2
