import sys
from pathlib import Path


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "bench":
        from .bench import main as bench

        bench(sys.argv[2:])
    elif cmd == "serve-bench":  # started by the benchmark itself, not by people
        from .bench import serve

        serve(Path(sys.argv[2]), int(sys.argv[3]))
    else:
        print("usage: python -m searchsvc bench [--sizes N ...]\n"
              "       uv run uvicorn searchsvc.api:app_from_env --factory --port 8002")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
