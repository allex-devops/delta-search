import re

from shared.docs import chunk_text

MIN_CHUNK = 200  # a short paragraph (a heading, a one-liner) is glued to the one after it


def chunk_paragraphs(text: str, size: int = 800) -> list[str]:
    """One chunk per paragraph, so an edit only changes the chunks around it.

    A sliding window of fixed size shifts every boundary after an edit, which would force the whole
    document to be re-embedded. Paragraph boundaries stay put.
    """
    out: list[str] = []
    carry = ""
    for raw in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", raw).strip()
        if not para:
            continue
        para = f"{carry} {para}".strip()
        carry = ""
        if len(para) < MIN_CHUNK:
            carry = para
        elif len(para) > size:
            out += chunk_text(para, size, overlap=size // 8)  # a very long paragraph is split on its own
        else:
            out.append(para)
    if carry:
        out.append(carry)  # a short trailing paragraph still gets indexed
    return out
