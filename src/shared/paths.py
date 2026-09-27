import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("SHARED_CACHE_DIR", ROOT / ".cache"))
