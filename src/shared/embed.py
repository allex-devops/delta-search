import numpy as np

from .cache import DiskCache, make_key
from .llm import LLM

# nomic-embed-text was trained with these task prefixes and retrieves noticeably worse without them
NOMIC_PREFIX = {"document": "search_document: ", "query": "search_query: "}


def embed_texts(
    texts: list[str],
    *,
    kind: str = "document",
    llm: LLM | None = None,
    model: str | None = None,
    batch_size: int = 32,
    cache: DiskCache | None = None,
) -> np.ndarray:
    """Embed texts, hitting the model only for ones we haven't seen. Returns (len(texts), dim)."""
    llm = llm or LLM()
    model = model or llm.embed_model
    cache = cache or DiskCache("embed-" + model.replace(":", "_").replace("/", "_"))
    prefix = NOMIC_PREFIX.get(kind, "") if "nomic" in model else ""

    out: list[np.ndarray | None] = [None] * len(texts)
    todo: list[tuple[int, str, str]] = []
    for i, text in enumerate(texts):
        full = prefix + text
        key = make_key(model, full)
        cached = cache.get_array(key)
        if cached is None:
            todo.append((i, key, full))
        else:
            out[i] = cached

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors = llm.embed([full for _, _, full in batch], model=model)
        for (i, key, _), vec in zip(batch, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            cache.set_array(key, arr)
            out[i] = arr

    if not out:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(out)
