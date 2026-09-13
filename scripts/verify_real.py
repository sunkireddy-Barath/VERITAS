"""End-to-end verification against REAL data. No mock fixtures.

    python scripts/verify_real.py

Asserts the behaviours VERITAS claims, on real SEC filings:
  current / historical / stale-qualified / abstention / real restatement.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veritas.agents.orchestrator import VeritasConfig
from veritas.ingest.real_sources import build_real_system
from veritas.model.transformer import VeritasLM
from veritas.pipeline import VeritasSystemBuilder
from veritas.tokenizer.bpe import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def build():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(ROOT / "checkpoints" / "tokenizer.json")
    ck = ROOT / "checkpoints" / "sft.pt"
    if not ck.exists():
        ck = ROOT / "checkpoints" / "best.pt"
    model = VeritasLM.load(str(ck), device).eval()
    b = VeritasSystemBuilder(model, tok, device=device, domain="corporate")
    print(f"device={device} checkpoint={ck.name}")
    # The same loader the API uses, so these checks see exactly what users see.
    build_real_system(b, ROOT / "data" / "real", verbose=True)
    return b, b.build(config=VeritasConfig(verbose=False, domain="corporate"))


CASES = [
    # (question, kind, expected). "contains" checks the VALUE, not merely that
    # some answer came back -- "answers" is how a wrong fiscal year passed.
    ("What was Apple Inc revenue in 2016?", "contains", "215.64"),
    ("What was Apple Inc net income in 2020?", "contains", "57.41"),
    # Microsoft's FY2021 ends 2021-06-30: 168.09B. 198.27B is FY2022.
    ("What was Microsoft revenue in 2021?", "contains", "168.09"),
    ("What was Apple Inc revenue in 2023?", "contains", "383.29"),
    # three 10-Ks restating the same figure are corroboration, not a dispute
    ("What was Apple Inc revenue in 2023?", "excludes", "disagree"),
    ("What is Apple Inc revenue?", "contains", "416.16"),
    ("What is Nvidia's revenue?", "excludes", "{{"),
    ("Who is the current CEO of Apple Inc.?", "contains", "John Ternus"),
    ("Who was the CEO of Microsoft in 2020?", "contains", "Satya Nadella"),
    ("Who was the CEO of Apple Inc in 2005?", "contains", "Steve Jobs"),
    ("What is Apple Inc headcount?", "abstains", None),
    ("What is Apple Inc number of unicorns?", "abstains", None),
    ("What was Apple Inc net income in 2008?", "conflict", None),
]


def main() -> int:
    b, veritas = build()
    print(f"\nentities={len(b.store.entities())} chunks={len(b.corpus)} "
          f"versions={sum(len(v) for v in b.store._index.values())}\n")
    passed = 0
    for q, kind, expect in CASES:
        a = veritas.answer(q)
        low = (a.answer or "").lower()
        if kind == "abstains":
            ok = a.abstained
        elif kind == "conflict":
            ok = "disagree" in low or a.support_level == "CONFLICTED"
        elif kind == "answers":
            ok = not a.abstained and len(a.answer) > 40
        elif kind == "excludes":
            ok = not a.abstained and expect not in (a.answer or "")
        else:
            ok = expect in (a.answer or "")
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} [{kind:9s}] {q}")
        print(f"        {(a.answer or '')[:150]}")
        print(f"        support={a.support_level} abstained={a.abstained} "
              f"coverage={a.coverage:.2f} citations={len(a.citations)}\n")
    print(f"{passed}/{len(CASES)} real-data checks passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
