"""Execute a notebook's code cells in order, in one namespace.

    python scripts/run_notebook.py notebooks/01_tokenizer.ipynb [--quiet]

nbconvert is not required. This is what CI runs: it proves every cell executes
top to bottom from a clean interpreter, which is the property that actually
breaks when a notebook rots.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: plt.show() must not block

# Windows consoles default to cp1252, which cannot encode the arrows and box
# characters the notebooks print. Without this the RUNNER crashes and reports a
# perfectly healthy notebook as failed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def run(path: Path, quiet: bool = False) -> bool:
    cells = [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__main__"}
    ok = True
    for i, cell in enumerate(cells, 1):
        src = "".join(cell["source"])
        buf = io.StringIO()
        t0 = time.time()
        try:
            with redirect_stdout(buf):
                exec(compile(src, f"{path.name}:cell{i}", "exec"), ns)
            dt = time.time() - t0
            print(f"  cell {i:2d}/{len(cells)} OK   {dt:6.2f}s")
            if not quiet and buf.getvalue().strip():
                for line in buf.getvalue().strip().split("\n")[:6]:
                    print(f"       | {line[:200]}")
        except Exception:
            print(f"  cell {i:2d}/{len(cells)} FAIL")
            print("       " + traceback.format_exc().replace("\n", "\n       ")[:1800])
            ok = False
            break
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = []
    for nbp in args.notebooks:
        p = Path(nbp)
        print(f"\n=== {p.name} ===")
        import os
        os.chdir(root / "notebooks")
        if not run(root / nbp if not p.is_absolute() else p, args.quiet):
            failures.append(p.name)
    print("\n" + ("ALL NOTEBOOKS RAN" if not failures else f"FAILED: {failures}"))
    sys.exit(1 if failures else 0)
