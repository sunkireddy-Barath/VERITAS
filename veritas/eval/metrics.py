"""Evaluation metrics (spec section 26).

Retrieval metrics
-----------------
* **Recall@k**  -- fraction of gold documents in the top k. The ceiling on
  everything downstream: evidence not retrieved cannot be verified.
* **Precision@k** -- fraction of the top k that is gold. Matters because every
  irrelevant chunk is context the verifier must reject.
* **MRR**       -- 1/rank of the first gold hit. Rewards getting one right
  answer to the top, which is what a single-fact question needs.
* **nDCG@k**    -- DCG / ideal DCG with a log2(1+rank) discount. The only one
  of the four that handles graded relevance and position together, so it is the
  headline retrieval number.

Generation / system metrics
---------------------------
* **evidence coverage** -- supported claims / checkable claims. The metric this
  project is organised around.
* **citation accuracy** -- of the citations emitted, how many actually support
  the claim they are attached to. Catches citation-shaped hallucination: the
  right-looking marker on an unrelated sentence.
* **temporal accuracy** -- did the answer use evidence valid at the asked-about
  time. Scored separately per question category, because a system can score
  well overall while failing every HISTORICAL question.
* **abstention quality** -- balanced accuracy over (should abstain, did
  abstain). Reported as a pair (correct abstention rate, over-abstention rate),
  never as a single number, because trading one for the other is the whole
  design question.
* **freshness lag** -- wall-clock time between a new source appearing and the
  answer changing. This is the metric a batch-rebuild system cannot win.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set


# --------------------------------------------------------------- retrieval
def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(retrieved[:k]) & set(gold)) / len(set(gold))


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not retrieved[:k]:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(retrieved[:k])


def mrr(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    g = set(gold)
    for i, d in enumerate(retrieved, 1):
        if d in g:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevance: Dict[str, float], k: int) -> float:
    dcg = sum(relevance.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(retrieved[:k]))
    ideal = sum(r / math.log2(i + 2)
                for i, r in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    return dcg / ideal if ideal > 0 else 0.0


# -------------------------------------------------------------- generation
def normalize_answer(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def answer_match(predicted: str, gold: Sequence[str]) -> bool:
    """Containment, not exact match: VERITAS answers are full sentences with
    citations, so exact match would score a correct answer as wrong."""
    p = normalize_answer(predicted)
    return any(normalize_answer(g) in p for g in gold if g)


def token_f1(predicted: str, gold: str) -> float:
    p, g = normalize_answer(predicted).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = set(p) & set(g)
    if not common:
        return 0.0
    n = sum(min(p.count(t), g.count(t)) for t in common)
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def citation_accuracy(answer, verifier=None) -> float:
    """Of the claims that carry citations, how many are actually SUPPORTED."""
    cited = [c for c in answer.claims if c.get("citations")]
    if not cited:
        return 0.0
    return sum(c["verdict"] == "SUPPORTED" for c in cited) / len(cited)


def unsupported_claim_rate(answer) -> float:
    if not answer.claims:
        return 0.0
    bad = sum(c["verdict"] in ("INSUFFICIENT", "REFUTED") for c in answer.claims)
    return bad / len(answer.claims)


# ---------------------------------------------------------------- temporal
def temporal_accuracy(answer, expected_value: str, expected_outdated: Sequence[str] = ()) -> float:
    """1.0 only if the answer states the right value AND does not present a
    superseded value as current. Half credit for right value, stale framing."""
    text = normalize_answer(answer.answer)
    got_right = normalize_answer(expected_value) in text
    quoted_stale = any(
        normalize_answer(o) in text and "until" not in text and "previous" not in text
        and "was" not in text
        for o in expected_outdated
    )
    if got_right and not quoted_stale:
        return 1.0
    if got_right:
        return 0.5
    return 0.0


def contradiction_detection(answer, has_conflict: bool) -> Optional[bool]:
    """True positive / true negative on conflict surfacing."""
    surfaced = bool(answer.conflicts) or answer.support_level == "CONFLICTED"
    return surfaced == has_conflict


# -------------------------------------------------------------- abstention
@dataclass
class AbstentionScore:
    correct_abstentions: int = 0
    missed_abstentions: int = 0     # should have abstained, answered anyway
    over_abstentions: int = 0       # had evidence, refused anyway
    correct_answers: int = 0

    @property
    def abstention_precision(self) -> float:
        d = self.correct_abstentions + self.over_abstentions
        return self.correct_abstentions / d if d else 1.0

    @property
    def abstention_recall(self) -> float:
        d = self.correct_abstentions + self.missed_abstentions
        return self.correct_abstentions / d if d else 1.0

    @property
    def balanced(self) -> float:
        p, r = self.abstention_precision, self.abstention_recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def update(self, should_abstain: bool, did_abstain: bool) -> None:
        if should_abstain and did_abstain:
            self.correct_abstentions += 1
        elif should_abstain and not did_abstain:
            self.missed_abstentions += 1
        elif not should_abstain and did_abstain:
            self.over_abstentions += 1
        else:
            self.correct_answers += 1


# ------------------------------------------------------------- aggregation
@dataclass
class RunMetrics:
    name: str
    n: int = 0
    accuracy: float = 0.0
    f1: float = 0.0
    coverage: float = 0.0
    citation_accuracy: float = 0.0
    unsupported_rate: float = 0.0
    temporal_accuracy: float = 0.0
    conflict_detection: float = 0.0
    recall_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mean_latency_s: float = 0.0
    abstention: AbstentionScore = field(default_factory=AbstentionScore)
    by_category: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        return {
            "system": self.name, "n": self.n,
            "accuracy": round(self.accuracy, 3), "f1": round(self.f1, 3),
            "coverage": round(self.coverage, 3),
            "cite_acc": round(self.citation_accuracy, 3),
            "unsupported": round(self.unsupported_rate, 3),
            "temporal": round(self.temporal_accuracy, 3),
            "conflict": round(self.conflict_detection, 3),
            "recall@5": round(self.recall_at_5, 3),
            "ndcg@10": round(self.ndcg_at_10, 3),
            "abstain_f1": round(self.abstention.balanced, 3),
            "latency_s": round(self.mean_latency_s, 3),
        }


def format_table(rows: Sequence[Dict[str, object]]) -> str:
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), max(len(str(r[c])) for r in rows)) for c in cols}
    head = " | ".join(str(c).ljust(widths[c]) for c in cols)
    sep = "-|-".join("-" * widths[c] for c in cols)
    body = "\n".join(" | ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows)
    return f"{head}\n{sep}\n{body}"
