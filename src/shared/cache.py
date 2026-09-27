import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .paths import CACHE_DIR


def make_key(*parts) -> str:
    # sort_keys so the same dict in a different order still hashes the same
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, namespace: str, root: Path | None = None):
        self.dir = Path(root or CACHE_DIR) / namespace

    def _path(self, key: str, ext: str) -> Path:
        return self.dir / key[:2] / f"{key}.{ext}"

    def _write(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # write to a temp file and rename, so a job killed mid-write never leaves a half file
        fd, tmp = tempfile.mkstemp(dir=path.parent)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

    def get_json(self, key: str):
        path = self._path(key, "json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            return None

    def set_json(self, key: str, value) -> None:
        self._write(self._path(key, "json"), json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def get_array(self, key: str) -> np.ndarray | None:
        path = self._path(key, "npy")
        try:
            return np.load(io.BytesIO(path.read_bytes()))
        except FileNotFoundError:
            return None
        except (ValueError, EOFError):
            path.unlink(missing_ok=True)
            return None

    def set_array(self, key: str, arr: np.ndarray) -> None:
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr, dtype=np.float32))
        self._write(self._path(key, "npy"), buf.getvalue())
