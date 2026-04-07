from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
from mmkg.adapters.base import BaseAdapter
from mmkg.schemas.core import (
    GraphDocument, Entity, Relation, Evidence,
    EntityType, Modality, SourceType
)
from mmkg.utils import get_logger

log = get_logger(__name__)

_LABEL_TO_TYPE = {
    "PER": EntityType.PERSON,
    "ORG": EntityType.ORGANIZATION,
    "LOC": EntityType.LOCATION,
    "MISC": EntityType.OTHER,
    "O": EntityType.UNKNOWN,
}


class MKGFormerAdapter(BaseAdapter):
    """Calls MKGformer via subprocess in its isolated conda environment."""

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("mkgformer", {})
        self.conda_env = self.cfg.get("conda_env", "mkgformer")
        self.runner_script = Path(self.cfg.get(
            "runner_script", "scripts/mkgformer_runner/run_mner.py"
        ))
        self.bert_model = self.cfg.get("bert_model", "bert-base-uncased")
        self.vit_model = self.cfg.get("vit_model", "openai/clip-vit-base-patch32")
        self.enabled = self.cfg.get("enabled", True)
        self.tmp_dir = Path("data/tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        doc_id: str,
        text: str,
        image_paths: Optional[List[Path]] = None,
    ) -> GraphDocument:
        if not self.enabled:
            log.warning("MKGformer disabled — returning empty GraphDocument")
            return GraphDocument(doc_id=doc_id, pipeline_stage="mkgformer-disabled")

        image_path = str(image_paths[0]) if image_paths else None
        input_file = self.tmp_dir / f"mkgformer_input_{doc_id}.json"
        output_file = self.tmp_dir / f"mkgformer_output_{doc_id}.json"

        input_file.write_text(json.dumps({
            "doc_id": doc_id,
            "text": text,
            "image_path": image_path,
        }))

        cmd = [
            "conda", "run", "--no-capture-output",
            "-n", self.conda_env,
            "python",
            str(self.runner_script),
            "--input", str(input_file),
            "--output", str(output_file),
            "--bert-model", self.bert_model,
            "--vit-model", self.vit_model,
        ]

        log.info("Calling MKGformer runner in env={e}", e=self.conda_env)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            log.error("MKGformer runner failed:\n{err}", err=result.stderr)
            return GraphDocument(doc_id=doc_id, pipeline_stage="mkgformer-error",
                                 metadata={"error": result.stderr})

        if not output_file.exists():
            log.error("MKGformer output file not created: {f}", f=output_file)
            return GraphDocument(doc_id=doc_id, pipeline_stage="mkgformer-missing-output")

        output = json.loads(output_file.read_text())

        if output.get("error"):
            log.error("MKGformer error: {e}", e=output["error"])
            return GraphDocument(doc_id=doc_id, pipeline_stage="mkgformer-error",
                                 metadata={"error": output["error"]})

        return self._to_graph_document(doc_id, output, image_path)

    def _to_graph_document(self, doc_id: str, output: dict,
                           image_path: Optional[str]) -> GraphDocument:
        entities: list[Entity] = []
        evidences: list[Evidence] = []

        for raw_ent in output.get("entities", []):
            ev = Evidence(
                source_doc_id=doc_id,
                source_type=(SourceType.MODEL_OUTPUT if raw_ent.get("image_grounded")
                             else SourceType.TEXT_SPAN),
                content=raw_ent["name"],
                image_path=image_path,
                model_name="mkgformer-mner",
                confidence=raw_ent.get("confidence", 1.0),
            )
            ent = Entity(
                canonical_name=raw_ent["name"],
                entity_type=_LABEL_TO_TYPE.get(raw_ent.get("label", "MISC"),
                                                EntityType.OTHER),
                modality=(Modality.MULTIMODAL if raw_ent.get("image_grounded")
                          else Modality.TEXT),
                source_system="mkgformer",
                evidence_ids=[ev.id],
            )
            evidences.append(ev)
            entities.append(ent)

        return GraphDocument(
            doc_id=doc_id,
            entities=entities,
            relations=[],
            evidences=evidences,
            pipeline_stage="mkgformer-mner",
        )
