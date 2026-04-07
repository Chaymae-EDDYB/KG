from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from mmkg.adapters.stub import StubAdapter
from mmkg.fusion.cross_modal import CrossModalFuser
from mmkg.schemas.core import GraphDocument
from mmkg.storage.graph_store import GraphStore
from mmkg.utils import get_logger

log = get_logger(__name__)


class IngestPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        primary = cfg.get("extraction", {}).get("primary", "stub")

        if primary == "stub":
            self.text_adapter = StubAdapter()
            self.mm_adapter = StubAdapter()
        elif primary == "itext2kg":
            from mmkg.adapters.itext2kg_adapter import IText2KGAdapter
            from mmkg.adapters.mkgformer_adapter import MKGFormerAdapter
            self.text_adapter = IText2KGAdapter(cfg)
            self.mm_adapter = MKGFormerAdapter(cfg)
        else:
            raise ValueError(f"Unknown extractor: {primary}")

        fusion_cfg = cfg.get("fusion", {})
        self.fuser = CrossModalFuser(
            similarity_threshold=fusion_cfg.get("clip_similarity_threshold", 0.80),
            merge_entities=fusion_cfg.get("merge_above_threshold", False),
        )
        storage_cfg = cfg.get("storage", {})
        self.store = GraphStore(
            output_dir=storage_cfg.get("output_dir", "data/processed"),
            fmt=storage_cfg.get("format", "both"),
        )

    def run(
        self,
        doc_id: str,
        text: str,
        image_paths: Optional[List[Path]] = None,
    ) -> GraphDocument:
        log.info("IngestPipeline.run: doc_id={d}", d=doc_id)

        text_doc = self.text_adapter.extract(doc_id, text, image_paths)
        log.info("Text extraction: {s}", s=text_doc.summary())

        mm_doc = self.mm_adapter.extract(doc_id, text, image_paths)
        log.info("Multimodal extraction: {s}", s=mm_doc.summary())

        fused = self.fuser.fuse(text_doc, mm_doc)
        log.info("After fusion: {s}", s=fused.summary())

        self.store.save(fused)
        return fused
