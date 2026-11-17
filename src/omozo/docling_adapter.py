"""Docling adapter: convert a file and chunk it for the embedding model.

Imported lazily so the MCP server path never loads Docling/torch.
"""

from __future__ import annotations

from pathlib import Path

from omozo.ingest import Chunk

_CHUNK_TOKENS = 512
_EMBED_MODEL = "BAAI/bge-m3"
_converter = None  # cache DocumentConverter across files


def convert_and_chunk(path: Path) -> list[Chunk]:
    """Convert one document with Docling and produce contextualized chunks."""
    global _converter
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )

    if _converter is None:
        _converter = DocumentConverter()
    tokenizer = HuggingFaceTokenizer.from_pretrained(
        _EMBED_MODEL, max_tokens=_CHUNK_TOKENS
    )
    chunker = HybridChunker(tokenizer=tokenizer)

    result = _converter.convert(str(path))
    doc = result.document
    title = doc.name or path.stem

    chunks: list[Chunk] = []
    for chunk in chunker.chunk(doc):
        text = chunker.contextualize(chunk)
        headings = getattr(chunk.meta, "headings", None) or []
        section = " » ".join(headings)
        chunks.append(Chunk(text=text, title=title, section=section))
    return chunks
