"""Integration test for the real Docling adapter.

Opt-in: OMOZO_INTEGRATION=1 (downloads the bge-m3 tokenizer from HF on
first run; conversion itself is CPU-only and fast for markdown).
"""

import os

import pytest

from omozo.docling_adapter import convert_and_chunk

pytestmark = pytest.mark.skipif(
    not os.environ.get("OMOZO_INTEGRATION"),
    reason="set OMOZO_INTEGRATION=1 to run Docling integration tests",
)


def test_convert_and_chunk_markdown(tmp_path):
    doc = tmp_path / "sample.md"
    doc.write_text(
        "# The Harbour\n\nMorning fog rolls over the quay.\n\n"
        "## Diesel Engines\n\nThe engine room hums low all night.\n",
        encoding="utf-8",
    )
    chunks = convert_and_chunk(doc)
    assert chunks, "expected at least one chunk"
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "harbour" in joined.lower() or "fog" in joined.lower()
    assert all(chunk.title == "sample" for chunk in chunks)


def test_full_pipeline_with_real_converter(conn, embedder, tmp_path):
    from omozo import documents, ingest

    root = tmp_path / "docs-1"
    root.mkdir()
    doc = root / "sample.md"
    doc.write_text(
        "# The Harbour\n\nMorning fog rolls over the quay. "
        "Gulls circle above the masts while the tide comes in.\n",
        encoding="utf-8",
    )
    report = ingest.ingest(conn, embedder, [doc], [root])
    assert report.added == 1 and report.chunks >= 1
    out = documents.search(
        conn, embedder, "fog over the quay", k=5, source=None, preview_chars=600
    )
    assert "[c" in out
    assert "docs-1 » sample.md" in out


def test_ocr_scanned_pdf_downloads_models_to_writable_dir(tmp_path, monkeypatch):
    """RapidOCR models must download into OMOZO_MODELS_DIR, not site-packages."""
    from PIL import Image, ImageDraw

    import omozo.docling_adapter as adapter

    models_dir = tmp_path / "ocr-models"
    monkeypatch.setenv("OMOZO_MODELS_DIR", str(models_dir))
    # The converter caches pinned OCR paths built from the env at first use;
    # reset so this test builds its own with the monkeypatched dir.
    adapter._converter = None
    adapter._chunker = None

    # Image-only PDF: text exists only as pixels, so OCR must run.
    img = Image.new("RGB", (600, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), "The harbour sleeps under morning fog.", fill="black")
    pdf = tmp_path / "scan.pdf"
    img.save(pdf, "PDF")

    from omozo.docling_adapter import convert_and_chunk

    chunks = convert_and_chunk(pdf)
    assert chunks, "expected at least one chunk"
    joined = " ".join(chunk.text.lower() for chunk in chunks)
    assert "harbour" in joined or "fog" in joined
    assert (models_dir / "PP-OCRv6_det_small.onnx").exists()
