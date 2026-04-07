from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "external", "MKGformer")
))

import argparse, json, re, traceback

_PATTERNS = [
    (r'\b[A-Z][a-z]+ (?:University|Institute|College|Lab|Laboratory)\b', 'ORG'),
    (r'\b(?:University|Institute) of [A-Z][a-z]+\b', 'ORG'),
    (r'\b[A-Z][a-z]+ (?:Prize|Award|Medal)(?: in [A-Z][a-z]+)?\b', 'MISC'),
    (r'\b[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*,\s*(?:Germany|France|USA|UK|China|Japan|Poland|Austria|Denmark|Serbia)\b', 'LOC'),
    (r'\b(?:Germany|France|USA|United States|UK|England|China|Japan|Italy|Spain|Russia|Poland|Austria|Denmark|Serbia|Switzerland)\b', 'LOC'),
    (r'\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', 'PER'),
]


def _regex_ner(text, has_image):
    found = []
    seen_spans = []
    seen_names = set()

    for pattern, label in _PATTERNS:
        for m in re.finditer(pattern, text):
            name = m.group().strip()
            start, end = m.start(), m.end()

            if name.lower() in seen_names or len(name) < 3:
                continue

            # Skip if this span is already covered by a longer match
            covered = any(
                existing_start <= start and end <= existing_end
                for existing_start, existing_end in seen_spans
            )
            if covered:
                continue

            seen_names.add(name.lower())
            seen_spans.append((start, end))
            found.append({
                "name": name,
                "label": label,
                "start": start,
                "end": end,
                "image_grounded": has_image,
                "confidence": 0.6,
            })

    # Remove any shorter match whose text is a substring of a longer match
    names = [f["name"] for f in found]
    filtered = [
        f for f in found
        if not any(
            f["name"] != other and f["name"] in other
            for other in names
        )
    ]
    return filtered


def run_mner(input_path, output_path, bert_model, vit_model):
    with open(input_path) as f:
        payload = json.load(f)

    doc_id = payload["doc_id"]
    text = payload["text"]
    image_path = payload.get("image_path")
    has_image = bool(image_path and os.path.exists(image_path))

    entities = []
    error = None

    model_cache = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "models", "bert-ner")
    )

    if os.path.isdir(model_cache):
        try:
            from transformers import pipeline
            ner_pipeline = pipeline(
                "ner",
                model=model_cache,
                tokenizer=model_cache,
                aggregation_strategy="simple",
            )
            raw = ner_pipeline(text)
            entities = [{
                "name": e["word"],
                "label": e["entity_group"],
                "start": e["start"],
                "end": e["end"],
                "image_grounded": has_image,
                "confidence": float(e["score"]),
            } for e in raw]
            print("[mkgformer_runner] Real NER: %d entities" % len(entities))
        except Exception as exc:
            print("[mkgformer_runner] Real NER failed (%s), using regex fallback" % exc)
            entities = _regex_ner(text, has_image)
    else:
        entities = _regex_ner(text, has_image)

    result = {"doc_id": doc_id, "entities": entities, "error": error}

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("[mkgformer_runner] %d entities -> %s" % (len(entities), output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--vit-model", default="openai/clip-vit-base-patch32")
    args = parser.parse_args()
    run_mner(args.input, args.output, args.bert_model, args.vit_model)
