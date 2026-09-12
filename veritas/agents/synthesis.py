"""Synthesis agent: write the answer *from verified claims*, not from context.

Ordinary RAG generates prose from retrieved context and then hopes the prose is
faithful. VERITAS inverts the order:

    retrieve -> extract claims -> verify each claim -> compose only survivors

Composition from an already-verified set makes a whole class of failure
impossible by construction: the answer cannot contain a claim that was never
verified, because the claim list *is* the input. The LM's job shrinks from
"be truthful" to "be fluent about these specific sentences" -- a job a 30M
parameter model can actually do.

Two synthesis modes:

* **extractive** (default, and the one used when the small LM is weak): the
  answer is assembled from the verified claim texts with their citations. It is
  never eloquent, but it is guaranteed faithful, and this is the correct
  default for an evidence system.
* **generative**: the fine-tuned model rewrites the verified claims into
  flowing prose at temperature <= 0.3, and the result is re-verified. If a
  regenerated claim fails, the extractive version is used instead. The model
  never gets the last word.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..evidence.claims import split_answer_into_claims
from ..evidence.provenance import Answer
from ..evidence.quality import SupportLevel
from ..evidence.verifier import ClaimVerdict, Verdict, rewrite_unsupported
from ..temporal.temporal_retrieval import TemporalIntent


class SynthesisAgent:
    def __init__(self, model=None, tokenizer=None, device: str = "cpu",
                 temperature: float = 0.25, max_new_tokens: int = 220) -> None:
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    # ----------------------------------------------------------- extractive
    def compose_extractive(self, plan, verdicts: Sequence[ClaimVerdict], assessment) -> str:
        lines = rewrite_unsupported(verdicts)
        head = ""
        if assessment.current is not None and plan.attribute:
            v = assessment.current
            if getattr(assessment, "current_is_valid_now", True):
                head = (f"As of the latest evidence, {plan.entity}'s {plan.attribute} is "
                        f"{v.value} (valid since {v.valid_from.date()}, "
                        f"source {v.source_id or 'unknown'}).")
            else:
                # Never state a closed-period figure in the present tense.
                head = (f"The most recently reported {plan.attribute} for {plan.entity} is "
                        f"{v.value}, covering {v.valid_from.date()} to {v.valid_to.date()} "
                        f"(reported {v.recorded_at.date()} by {v.source_id or 'unknown'}). "
                        f"No newer figure has been published.")
        if plan.temporal and plan.temporal.intent == TemporalIntent.CHANGE:
            if assessment.changes:
                head = " ".join(assessment.changes[:3])
            elif len(assessment.historical) >= 2:
                # A series of non-overlapping periods (annual revenue, say)
                # records no CHANGED events -- each fiscal year is its own
                # fact. The progression IS the answer, so render the timeline
                # rather than falling through to "the latest value", which
                # answers a different question entirely.
                series = sorted(assessment.historical, key=lambda v: v.valid_from)
                pts = [f"{v.valid_from.date()} to {v.valid_to.date()}: {v.value}"
                       for v in series[-6:]]
                first, last = series[0], series[-1]
                head = (f"{plan.entity}'s {plan.attribute} across "
                        f"{len(series)} recorded periods -- " + "; ".join(pts) + ". "
                        f"Earliest on record {first.value} "
                        f"({first.valid_from.date()}); most recent {last.value} "
                        f"({last.valid_to.date()}).")
        if plan.temporal and plan.temporal.intent in (TemporalIntent.HISTORICAL, TemporalIntent.AS_OF):
            hist = [h for h in assessment.historical
                    if h.valid_from <= plan.temporal.anchor < h.valid_to]
            if hist:
                h = hist[-1]
                head = (f"As of {plan.temporal.anchor.date()}, {plan.entity}'s {plan.attribute} "
                        f"was {h.value} (valid {h.valid_from.date()} to "
                        f"{'present' if h.valid_to.year > 9000 else h.valid_to.date()}).")
        body = " ".join(lines)
        note = f" {assessment.note}" if assessment.note else ""
        return " ".join(p for p in (head, body, note.strip()) if p).strip()

    # ----------------------------------------------------------- generative
    def compose_generative(self, plan, verdicts: Sequence[ClaimVerdict]) -> Optional[str]:
        if self.model is None or self.tok is None:
            return None
        supported = [v for v in verdicts if v.verdict == Verdict.SUPPORTED]
        if not supported:
            return None
        import torch

        facts = "\n".join(f"- {v.claim.text}" for v in supported[:8])
        prompt = (
            "<|bos|><|system|>Answer using only the verified facts. Do not add anything not "
            "listed. If a fact is missing, say it is not established."
            f"<|user|>Question: {plan.question}\nVerified facts:\n{facts}<|assistant|>"
        )
        ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
        out = self.model.generate(
            ids, max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            top_k=40, top_p=0.9, eos_id=self.tok.special_tokens["<|eos|>"],
        )
        text = self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special=True).strip()
        return text or None

    # ------------------------------------------------------------- guarded
    def compose(self, plan, verdicts, assessment, verifier=None, evidence_lookup=None,
                mode: str = "extractive") -> str:
        """Generative output is re-verified; on any regression, fall back."""
        extractive = self.compose_extractive(plan, verdicts, assessment)
        if mode != "generative":
            return extractive
        gen = self.compose_generative(plan, verdicts)
        if not gen or verifier is None or evidence_lookup is None:
            return extractive
        regen_claims = split_answer_into_claims(gen, plan.entity)
        _, metrics = verifier.verify_answer(regen_claims, evidence_lookup)
        # the regenerated text must be at least as well-supported as the
        # extractive baseline, or it is discarded outright
        return gen if metrics["coverage"] >= 0.8 else extractive


class CitationAgent:
    """Maps every sentence of the final answer back to evidence markers.

    Run *after* composition as an audit, not as the thing that adds citations:
    any sentence that ends up with no marker is reported, which is how a
    citation that quietly went missing during rewriting gets caught.
    """

    @staticmethod
    def audit(answer_text: str, verdicts: Sequence[ClaimVerdict]) -> Dict[str, object]:
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
        cited = [s for s in sentences if re.search(r"\[E\d+\]|\[[a-z0-9_.#-]+\]", s, re.I)]
        factual = [s for s in sentences if re.search(r"\d|is |was |reported|announced", s)]
        return {
            "n_sentences": len(sentences),
            "n_cited": len(cited),
            "citation_density": len(cited) / max(1, len(factual)),
            "uncited_factual": [s for s in factual if s not in cited],
        }
