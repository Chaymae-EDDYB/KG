import json
from pathlib import Path
import pytest
from mmkg.pipeline.ingest import IngestPipeline
from mmkg.schemas.core import GraphDocument

def make_cfg(tmp_path):
    return {
        "extraction": {"primary": "stub", "relation_confidence_threshold": 0.5},
        "fusion": {"clip_similarity_threshold": 0.80, "merge_above_threshold": False},
        "storage": {"output_dir": str(tmp_path), "format": "json"},
        "mkgformer": {"enabled": False},
        "itext2kg": {},
    }

def test_ingest_text_only(tmp_path):
    pipeline = IngestPipeline(make_cfg(tmp_path))
    result = pipeline.run(doc_id="test-001",
                          text="Einstein was born in Ulm, Germany.")
    assert result.doc_id == "test-001"
    assert len(result.entities) > 0
    saved = tmp_path / "test-001.json"
    assert saved.exists()
    loaded = GraphDocument.model_validate(json.loads(saved.read_text()))
    assert loaded.doc_id == "test-001"

def test_ingest_with_image(tmp_path):
    pipeline = IngestPipeline(make_cfg(tmp_path))
    result = pipeline.run(doc_id="test-002",
                          text="Einstein at the blackboard.",
                          image_paths=[Path("data/samples/test.jpg")])
    image_ents = [e for e in result.entities if e.modality.value == "image"]
    assert len(image_ents) >= 1
    cross_modal = [r for r in result.relations if r.predicate == "depicted_as"]
    assert len(cross_modal) >= 1

def test_pipeline_stage_is_fusion(tmp_path):
    pipeline = IngestPipeline(make_cfg(tmp_path))
    result = pipeline.run(doc_id="test-003",
                          text="Marie Curie discovered radium.",
                          image_paths=[Path("data/samples/test.jpg")])
    assert result.pipeline_stage == "fusion"
