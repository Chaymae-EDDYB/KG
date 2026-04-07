from __future__ import annotations
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from mmkg.adapters.base import BaseAdapter
from mmkg.schemas.core import (
    GraphDocument, Entity, Relation, Evidence,
    EntityType, Modality, SourceType
)
from mmkg.utils import get_logger

load_dotenv()
log = get_logger(__name__)

_ENTITY_TYPE_MAP: Dict[str, EntityType] = {
    "person": EntityType.PERSON,
    "per": EntityType.PERSON,
    "organization": EntityType.ORGANIZATION,
    "org": EntityType.ORGANIZATION,
    "location": EntityType.LOCATION,
    "loc": EntityType.LOCATION,
    "gpe": EntityType.LOCATION,
    "event": EntityType.EVENT,
    "concept": EntityType.CONCEPT,
}


def _map_entity_type(label: str) -> EntityType:
    return _ENTITY_TYPE_MAP.get(str(label).lower().strip(), EntityType.OTHER)


def _make_llm(model_name: str, api_key: str):
    """
    Build a LangChain ChatGroq object.
    Use llama-3.3-70b-versatile for best structured-output support.
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model_name,
        groq_api_key=api_key,
        temperature=0,
    )


def _make_embeddings(model_name: str):
    """
    Build a local HuggingFace embeddings model (free, no API needed).
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _sanitize_props(props) -> dict:
    """
    Convert iText2KG property objects and numpy arrays to plain
    JSON-serializable Python dicts/lists.
    """
    # If props is a Pydantic model (e.g., EntityProperties), convert it
    if hasattr(props, "model_dump"):
        props = props.model_dump()
    elif hasattr(props, "dict"):
        props = props.dict()
    elif not isinstance(props, dict):
        props = {}

    cleaned: Dict[str, object] = {}
    for k, v in props.items():
        if isinstance(v, np.ndarray):
            # Drop large embedding arrays to keep output clean
            continue
        elif isinstance(v, (np.floating, np.integer)):
            cleaned[k] = v.item()
        else:
            cleaned[k] = v
    return cleaned


class IText2KGAdapter(BaseAdapter):
    """
    Wraps iText2KG using Groq LLM + local HuggingFace embeddings.
    build_graph is async so we wrap it with asyncio.run().
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg.get("itext2kg", {})
        # Use 70b model — 8b models fail on structured output / tool calls
        self.llm_model_name = self.cfg.get("llm_model", "llama-3.3-70b-versatile")
        self.embeddings_model_name = self.cfg.get("embeddings_model", "all-MiniLM-L6-v2")
        self.ent_threshold = self.cfg.get("entity_merge_threshold", 0.85)
        self.rel_threshold = self.cfg.get("relation_merge_threshold", 0.80)
        self.api_key = os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at https://console.groq.com"
            )

    def extract(
        self,
        doc_id: str,
        text: str,
        image_paths: Optional[List[Path]] = None,
    ) -> GraphDocument:
        log.info("IText2KGAdapter.extract: doc_id={d}", d=doc_id)

        llm = _make_llm(self.llm_model_name, self.api_key)
        embeddings = _make_embeddings(self.embeddings_model_name)

        try:
            return self._run_itext2kg(doc_id, text, llm, embeddings)
        except Exception as e1:
            log.warning("iText2KG path failed: {e} — trying Atom", e=str(e1)[:200])
            try:
                return self._run_atom(doc_id, text, llm, embeddings)
            except Exception as e2:
                raise RuntimeError(
                    f"Both iText2KG and Atom failed.\n"
                    f"iText2KG: {e1}\nAtom: {e2}"
                )

    # ── iText2KG path ─────────────────────────────────────────────────

    def _run_itext2kg(self, doc_id: str, text: str, llm, embeddings) -> GraphDocument:
        from itext2kg import iText2KG

        builder = iText2KG(
            llm_model=llm,
            embeddings_model=embeddings,
            sleep_time=1,
        )

        sections = [s.strip() for s in text.split("\n\n") if s.strip()]
        if not sections:
            sections = [text]

        async def _run():
            return await builder.build_graph(
                sections=sections,
                ent_threshold=self.ent_threshold,
                rel_threshold=self.rel_threshold,
            )

        kg = asyncio.run(_run())
        log.info("iText2KG.build_graph returned: {t}, "
                 "entities={e}, relationships={r}",
                 t=type(kg).__name__,
                 e=len(getattr(kg, "entities", [])),
                 r=len(getattr(kg, "relationships", [])))

        return self._kg_to_graph_document(doc_id, kg)

    # ── Atom path (fallback) ──────────────────────────────────────────

    def _run_atom(self, doc_id: str, text: str, llm, embeddings) -> GraphDocument:
        from itext2kg import Atom

        atom = Atom(llm_model=llm, embeddings_model=embeddings)
        obs_timestamp = datetime.now().strftime("%Y-%m-%d")

        lines = [s.strip() for s in text.split("\n") if s.strip()]

        result = atom.build_graph(
            atomic_facts=lines,
            obs_timestamp=obs_timestamp,
            ent_threshold=self.ent_threshold,
            rel_threshold=self.rel_threshold,
        )

        if asyncio.iscoroutine(result):
            async def _run():
                return await result
            kg = asyncio.run(_run())
        else:
            kg = result

        log.info("Atom.build_graph returned: {t}", t=type(kg).__name__)
        return self._kg_to_graph_document(doc_id, kg)

    # ── KG → GraphDocument ────────────────────────────────────────────

    def _kg_to_graph_document(self, doc_id: str, kg) -> GraphDocument:
        entities: List[Entity] = []
        relations: List[Relation] = []
        evidences: List[Evidence] = []
        name_to_entity: Dict[str, Entity] = {}

        # ── entities ──────────────────────────────────────────────────
        raw_entities = getattr(kg, "entities", None)
        if raw_entities is None and isinstance(kg, dict):
            raw_entities = kg.get("entities", kg.get("nodes", []))
        if raw_entities is None:
            raw_entities = []

        for raw_ent in raw_entities:
            name, label, props = self._parse_entity(raw_ent)
            if not name:
                continue
            ev = Evidence(
                source_doc_id=doc_id,
                source_type=SourceType.TEXT_SPAN,
                content=name,
                model_name=f"itext2kg/{self.llm_model_name}",
            )
            ent = Entity(
                canonical_name=name,
                entity_type=_map_entity_type(label),
                modality=Modality.TEXT,
                source_system="itext2kg",
                evidence_ids=[ev.id],
                metadata=props,
            )
            evidences.append(ev)
            entities.append(ent)
            name_to_entity[name] = ent

        # ── relationships ─────────────────────────────────────────────
        # iText2KG KnowledgeGraph uses .relationships (list of Relationship)
        raw_rels = getattr(kg, "relationships", None)
        if raw_rels is None and isinstance(kg, dict):
            for key in ["relationships", "relations", "edges", "triples"]:
                if key in kg and kg[key]:
                    raw_rels = kg[key]
                    break
        if raw_rels is None:
            raw_rels = []

        for raw_rel in raw_rels:
            subj_name, obj_name, predicate, confidence, props = self._parse_relation(raw_rel)
            if not subj_name or not obj_name:
                continue
            subj = name_to_entity.get(subj_name)
            obj = name_to_entity.get(obj_name)
            if subj is None or obj is None:
                log.warning("Dangling relation: {s} -{p}-> {o}",
                            s=subj_name, p=predicate, o=obj_name)
                continue
            rel = Relation(
                subject_id=subj.id,
                predicate=predicate.lower().replace(" ", "_"),
                object_id=obj.id,
                confidence=min(1.0, max(0.0, confidence)),
                source_system="itext2kg",
                metadata=props,
            )
            relations.append(rel)

        log.info("Converted KG: {e} entities, {r} relations",
                 e=len(entities), r=len(relations))

        return GraphDocument(
            doc_id=doc_id,
            entities=entities,
            relations=relations,
            evidences=evidences,
            pipeline_stage="itext2kg",
        )

    # ── Static helpers ────────────────────────────────────────────────

    @staticmethod
    def _parse_entity(raw) -> Tuple[str, str, dict]:
        if isinstance(raw, dict):
            name = str(raw.get("name", raw.get("label", raw.get("id", ""))))
            label = str(raw.get("label", raw.get("type", "unknown")))
            props = raw.get("properties", {}) or {}
        else:
            name = str(getattr(raw, "name", getattr(raw, "label",
                       getattr(raw, "id", ""))))
            label = str(getattr(raw, "label", getattr(raw, "type", "unknown")))
            props = getattr(raw, "properties", {}) or {}
        return name.strip(), label.strip(), _sanitize_props(props)

    @staticmethod
    def _parse_relation(raw) -> Tuple[str, str, str, float, dict]:
        """
        Parse a Relationship from iText2KG.
        iText2KG Relationship has: startEntity (Entity), endEntity (Entity),
        name (str), properties (RelationshipProperties)
        """
        if isinstance(raw, dict):
            subj = str(raw.get("startNode", raw.get("start",
                       raw.get("source", raw.get("subject", "")))))
            obj = str(raw.get("endNode", raw.get("end",
                      raw.get("target", raw.get("object", "")))))
            pred = str(raw.get("name", raw.get("type",
                       raw.get("relation", raw.get("predicate", "related_to")))))
            conf = float(raw.get("confidence", 1.0))
            props = raw.get("properties", {}) or {}
        else:
            # iText2KG Relationship object: .startEntity.name, .endEntity.name
            start_ent = getattr(raw, "startEntity", None)
            end_ent = getattr(raw, "endEntity", None)
            subj = str(getattr(start_ent, "name", "")) if start_ent else str(
                getattr(raw, "startNode", getattr(raw, "start",
                getattr(raw, "source", getattr(raw, "subject", "")))))
            obj = str(getattr(end_ent, "name", "")) if end_ent else str(
                getattr(raw, "endNode", getattr(raw, "end",
                getattr(raw, "target", getattr(raw, "object", "")))))
            pred = str(getattr(raw, "name", getattr(raw, "type",
                       getattr(raw, "relation",
                       getattr(raw, "predicate", "related_to")))))
            conf = float(getattr(raw, "confidence", 1.0))
            props = getattr(raw, "properties", {}) or {}
        return subj.strip(), obj.strip(), pred.strip(), conf, _sanitize_props(props)
