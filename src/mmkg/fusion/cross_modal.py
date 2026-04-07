from __future__ import annotations
import numpy as np
from mmkg.schemas.core import (
    GraphDocument, Entity, Relation, Evidence,
    Modality, SourceType
)
from mmkg.utils import get_logger

log = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


class CrossModalFuser:
    def __init__(
        self,
        similarity_threshold: float = 0.80,
        merge_entities: bool = False,
        encoder=None,
    ):
        self.threshold = similarity_threshold
        self.merge_entities = merge_entities
        self._encoder = encoder

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Loaded SentenceTransformer encoder")
        return self._encoder

    def _embed_entities(self, entities: list[Entity]) -> dict[str, list[float]]:
        encoder = self._get_encoder()
        names = [e.canonical_name for e in entities]
        embeddings = encoder.encode(names, normalize_embeddings=True).tolist()
        return {e.id: emb for e, emb in zip(entities, embeddings)}

    def fuse(
        self,
        text_doc: GraphDocument,
        multimodal_doc: GraphDocument,
    ) -> GraphDocument:
        log.info(
            "Fusing: text({t} ents) + multimodal({m} ents)",
            t=len(text_doc.entities), m=len(multimodal_doc.entities)
        )

        all_entities = list(text_doc.entities) + list(multimodal_doc.entities)
        all_relations = list(text_doc.relations) + list(multimodal_doc.relations)
        all_evidences = list(text_doc.evidences) + list(multimodal_doc.evidences)

        text_ents = [e for e in all_entities if e.modality == Modality.TEXT]
        mm_ents = [e for e in all_entities
                   if e.modality in (Modality.MULTIMODAL, Modality.IMAGE)]

        if not text_ents or not mm_ents:
            log.info("No cross-modal pairs to align — returning merged document")
            return GraphDocument(
                doc_id=text_doc.doc_id,
                entities=all_entities,
                relations=all_relations,
                evidences=all_evidences,
                pipeline_stage="fusion",
                metadata={**text_doc.metadata, **multimodal_doc.metadata},
            )

        # Step 1: exact name matching
        linked_pairs: list[tuple[Entity, Entity]] = []
        text_names = {e.canonical_name.lower(): e for e in text_ents}
        for mm_ent in mm_ents:
            key = mm_ent.canonical_name.lower()
            if key in text_names:
                linked_pairs.append((text_names[key], mm_ent))

        # Step 2: embedding similarity for unmatched
        matched_text_ids = {p[0].id for p in linked_pairs}
        matched_mm_ids = {p[1].id for p in linked_pairs}
        unmatched_text = [e for e in text_ents if e.id not in matched_text_ids]
        unmatched_mm = [e for e in mm_ents if e.id not in matched_mm_ids]

        if unmatched_text and unmatched_mm:
            try:
                text_embs = self._embed_entities(unmatched_text)
                mm_embs = self._embed_entities(unmatched_mm)
                for t_ent in unmatched_text:
                    best_sim, best_mm = 0.0, None
                    for mm_ent in unmatched_mm:
                        sim = _cosine_similarity(text_embs[t_ent.id], mm_embs[mm_ent.id])
                        if sim > best_sim:
                            best_sim, best_mm = sim, mm_ent
                    if best_mm is not None and best_sim >= self.threshold:
                        linked_pairs.append((t_ent, best_mm))
                        log.debug("Embedding match (sim={s:.2f}): {a} <-> {b}",
                                  s=best_sim, a=t_ent.canonical_name,
                                  b=best_mm.canonical_name)
            except Exception as e:
                log.warning("Embedding alignment failed: {e}", e=e)

        # Step 3: create edges or merge
        new_relations: list[Relation] = []
        new_evidences: list[Evidence] = []
        ids_to_remove: set[str] = set()

        for text_ent, mm_ent in linked_pairs:
            ev = Evidence(
                source_doc_id=text_doc.doc_id,
                source_type=SourceType.MODEL_OUTPUT,
                content=f"cross-modal: {text_ent.canonical_name} <-> {mm_ent.canonical_name}",
                model_name="cross-modal-fuser",
            )
            new_evidences.append(ev)

            if self.merge_entities:
                text_ent.modality = Modality.MULTIMODAL
                text_ent.aliases.append(mm_ent.canonical_name)
                text_ent.evidence_ids.extend(mm_ent.evidence_ids)
                ids_to_remove.add(mm_ent.id)
                for rel in all_relations:
                    if rel.subject_id == mm_ent.id:
                        rel.subject_id = text_ent.id
                    if rel.object_id == mm_ent.id:
                        rel.object_id = text_ent.id
            else:
                new_relations.append(Relation(
                    subject_id=text_ent.id,
                    predicate="depicted_as",
                    object_id=mm_ent.id,
                    confidence=1.0,
                    source_system="fusion",
                    evidence_ids=[ev.id],
                ))

        final_entities = [e for e in all_entities if e.id not in ids_to_remove]
        final_relations = all_relations + new_relations
        final_evidences = all_evidences + new_evidences

        log.info("Fusion done: {e} entities, {r} relations ({c} cross-modal)",
                 e=len(final_entities), r=len(final_relations), c=len(new_relations))

        return GraphDocument(
            doc_id=text_doc.doc_id,
            entities=final_entities,
            relations=final_relations,
            evidences=final_evidences,
            pipeline_stage="fusion",
            metadata={**text_doc.metadata, **multimodal_doc.metadata},
        )
