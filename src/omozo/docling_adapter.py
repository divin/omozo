"""Docling adapter: convert a file and chunk it for the embedding model.

Imported lazily so the MCP server path never loads Docling/torch.

OCR note: RapidOCR downloads its detection/recognition models on first use.
By default it writes them into its own site-packages directory, which is
read-only for the non-root container user — so we pin explicit model paths
under OMOZO_MODELS_DIR (default: ${DATA_DIR}/models/rapidocr, i.e. inside
the persisted /data volume). First use downloads there once; later runs and
container recreations reuse the files.
"""

from __future__ import annotations

import os
from pathlib import Path

from omozo.ingest import Chunk

_CHUNK_TOKENS = 512
_EMBED_MODEL = "BAAI/bge-m3"
_converter = None  # cache DocumentConverter across files
_chunker = None


def _ocr_models_dir() -> Path:
    configured = os.environ.get("OMOZO_MODELS_DIR", "").strip()
    if configured:
        return Path(configured)
    data_dir = os.environ.get("DATA_DIR", "./data").strip() or "./data"
    return Path(data_dir) / "models" / "rapidocr"


# Model set for the onnxruntime backend: det/rec from PP-OCRv6, cls from
# PP-OCRv4 (the angle classifier rapidocr's cls step expects; the newer
# textline-orientation models have incompatible input shapes). The rec
# character dict is embedded in the ONNX model, so no keys file is needed.
_OCR_MODELS = {
    "det": "PP-OCRv6_det_small.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    "rec": "PP-OCRv6_rec_small.onnx",
}


def _ensure_ocr_models(models_dir: Path) -> None:
    """Download RapidOCR models into models_dir (persisted, writable).

    RapidOCR's own auto-download targets its site-packages directory, which
    is read-only for the container user — so we download explicitly via
    rapidocr's own downloader and pin the paths in RapidOcrOptions.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    targets = {name: models_dir / fname for name, fname in _OCR_MODELS.items()}
    if all(p.exists() for p in targets.values()):
        return

    from rapidocr.inference_engine.base import FileInfo, InferSession
    from rapidocr.utils.download_models import download_task
    from rapidocr.utils.typings import EngineType, ModelType, OCRVersion, TaskType

    file_infos = {
        "det": FileInfo(
            EngineType.ONNXRUNTIME,
            OCRVersion.PPOCRV6,
            TaskType.DET,
            "ch",
            ModelType.SMALL,
        ),
        "cls": FileInfo(
            EngineType.ONNXRUNTIME,
            OCRVersion.PPOCRV4,
            TaskType.CLS,
            "ch",
            ModelType.MOBILE,
        ),
        "rec": FileInfo(
            EngineType.ONNXRUNTIME,
            OCRVersion.PPOCRV6,
            TaskType.REC,
            "ch",
            ModelType.SMALL,
        ),
    }
    for name, file_info in file_infos.items():
        target = targets[name]
        if target.exists():
            continue
        download_task(models_dir, file_info)
        downloaded = (
            models_dir / Path(InferSession.get_model_url(file_info)["model_dir"]).name
        )
        if not downloaded.exists():
            raise RuntimeError(f"RapidOCR model download failed for {name}")
        if downloaded != target:
            downloaded.rename(target)
    print(f"omozo: RapidOCR models ready in {models_dir}")


def _make_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    models_dir = _ocr_models_dir()
    ocr_options = RapidOcrOptions(
        det_model_path=str(models_dir / _OCR_MODELS["det"]),
        cls_model_path=str(models_dir / _OCR_MODELS["cls"]),
        rec_model_path=str(models_dir / _OCR_MODELS["rec"]),
    )
    pipeline_options = PdfPipelineOptions(ocr_options=ocr_options)

    tokenizer = HuggingFaceTokenizer.from_pretrained(
        _EMBED_MODEL, max_tokens=_CHUNK_TOKENS
    )
    chunker = HybridChunker(tokenizer=tokenizer)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter, chunker


_IMAGE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def convert_and_chunk(path: Path) -> list[Chunk]:
    """Convert one document with Docling and produce contextualized chunks."""
    global _converter, _chunker
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        # OCR models must exist before docling builds the pdf/image pipeline;
        # defer the download until a scanned document actually needs them.
        _ensure_ocr_models(_ocr_models_dir())
    if _converter is None or _chunker is None:
        _converter, _chunker = _make_converter()
    assert _chunker is not None

    result = _converter.convert(str(path))
    doc = result.document
    title = doc.name or path.stem

    chunks: list[Chunk] = []
    for chunk in _chunker.chunk(doc):
        text = _chunker.contextualize(chunk)
        headings = getattr(chunk.meta, "headings", None) or []
        section = " » ".join(headings)
        chunks.append(Chunk(text=text, title=title, section=section))
    if not chunks:
        # Degenerate documents (e.g. a single OCR'd line the chunker skips)
        # still deserve an entry: fall back to the exported text.
        text = doc.export_to_text().strip()
        if text:
            chunks.append(Chunk(text=text, title=title, section=""))
    return chunks
