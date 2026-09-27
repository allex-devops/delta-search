"""Reading and chunking documents."""
import re
from pathlib import Path

import fitz  # pymupdf

TEXT_SUFFIXES = {".txt", ".md"}


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into chunks of at most `size` characters, preferring to break at sentence ends."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # only look for a break in the back half, otherwise a stray early period makes tiny chunks
            floor = start + size // 2
            cut = max(text.rfind(". ", floor, end), text.rfind("\n", floor, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)  # the max guarantees progress
    return chunks


def read_pages(path: Path) -> list[tuple[int, str]]:
    """(page number, text) pairs. Plain text and markdown files count as a single page."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as doc:
            return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    if suffix in TEXT_SUFFIXES:
        return [(1, path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"unsupported file type: {path.suffix or path.name}")
