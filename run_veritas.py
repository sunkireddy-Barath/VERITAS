"""Launch the VERITAS API + frontend.

    python run_veritas.py          then open http://localhost:8000

Checks prerequisites first and says exactly what is missing, because the most
common failure is running this before the notebooks have produced checkpoints.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    problems = []
    if not (ROOT / "checkpoints" / "tokenizer.json").exists():
        problems.append("checkpoints/tokenizer.json missing -> run notebook 01")
    if not any((ROOT / "checkpoints").glob("*.pt")):
        problems.append("no model checkpoint -> run notebook 03 (and 04)")
    if not (ROOT / "data" / "real").exists():
        problems.append("data/real missing -> python scripts/fetch_real_data.py")
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        problems.append("pip install fastapi 'uvicorn[standard]' pydantic")

    if problems:
        print("Cannot start:")
        for p in problems:
            print("  -", p)
        return 1

    print("VERITAS starting on http://localhost:8000")
    print("(first request loads the model and real data; give it a few seconds)\n")
    return subprocess.call([sys.executable, "-m", "uvicorn", "api.main:app",
                            "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
