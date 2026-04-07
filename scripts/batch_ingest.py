"""Batch ingest all .txt files in data/raw/ through the MMKG pipeline."""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from mmkg.utils.config import load_config
from mmkg.utils.logging_utils import configure_logging
from mmkg.pipeline.ingest import IngestPipeline


def main():
    configure_logging("INFO")
    cfg = load_config(None)
    pipeline = IngestPipeline(cfg)

    raw_dir = Path("data/raw")
    txt_files = sorted(raw_dir.glob("*.txt"))

    if not txt_files:
        print("No .txt files found in %s" % raw_dir)
        sys.exit(1)

    image_path = raw_dir / "test_image.jpg"
    images = [image_path] if image_path.exists() else None

    print("=" * 60)
    print("Batch Ingestion: %d documents" % len(txt_files))
    print("Images: %s" % ("yes" if images else "no"))
    print("=" * 60)

    successes = 0
    failures = 0
    t0 = time.time()

    for i, txt_file in enumerate(txt_files, 1):
        doc_id = txt_file.stem
        text = txt_file.read_text()

        print("\n[%d/%d] Ingesting doc_id=%s ..." % (i, len(txt_files), doc_id))
        try:
            result = pipeline.run(doc_id=doc_id, text=text, image_paths=images)
            print("  -> %s" % result.summary())
            successes += 1
        except Exception as exc:
            print("  -> FAILED: %s" % str(exc)[:200])
            failures += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("Batch complete in %.1fs" % elapsed)
    print("  Successes: %d" % successes)
    print("  Failures:  %d" % failures)
    print("=" * 60)


if __name__ == "__main__":
    main()
