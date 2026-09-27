import re
import zlib

import numpy as np
import pytest

from searchsvc.index import SearchIndex

STOP = set("a an and are as at be by do does for how in is it of on or the to what when who why with that this from about".split())


class CountingEmbed:
    """Hashed bag of words that also remembers how many texts it was asked to embed."""

    def __init__(self):
        self.texts_embedded = 0
        self.calls = 0

    def __call__(self, texts, kind="document"):
        self.calls += 1
        self.texts_embedded += len(texts)
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                if word not in STOP:
                    out[row, zlib.crc32(word.encode()) % 256] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


def paragraph(topic: str) -> str:
    """A paragraph long enough to stand as its own chunk. Its text depends only on the topic, never on where
    it sits in a document, because moving a paragraph must not look like editing it."""
    return " ".join(f"The {topic} appears in this sentence and the {topic} matters a great deal here." for _ in range(4))


def document(*topics: str) -> str:
    return "\n\n".join(paragraph(t) for t in topics)


@pytest.fixture
def embed():
    return CountingEmbed()


@pytest.fixture
def index(tmp_path, embed):
    return SearchIndex(tmp_path / "idx", embed, "fake-model")
