"""The VERITAS agentic loop: plan -> search -> assess -> refine -> verify -> answer.

    QUESTION
      -> PLAN
      -> SEARCH  <-------------------+
      -> RETRIEVE                    |
      -> TEMPORAL ASSESSMENT         | refine query against the specific gap
      -> EVIDENCE SUFFICIENT? --no --+   (bounded: max_iterations)
           |yes
      -> VERIFY CLAIMS
      -> CONTRADICTION ANALYSIS
      -> ABSTAIN? -- yes -> "insufficient evidence" + what is missing
           |no
      -> SYNTHESISE -> RE-VERIFY -> ANSWER + EVIDENCE + TIMELINE + CONFIDENCE

Two properties make this different from "retrieve once, then answer":

* **The loop condition is evidence sufficiency, not a step count.** It stops
  when the planner's required evidence is satisfied, and it refines against the
  *named gap* ("prior state missing") rather than rephrasing the question.
* **It is bounded.** `max_iterations` caps retrieval, and a no-progress check
  breaks early when an iteration adds no new documents. Unbounded agent loops
  are how a research agent turns one question into 200 retrievals.

Every decision is appended to `trace`, so the answer ships with the reasoning
path that produced it.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..evidence.claims import extract_claims, split_answer_into_claims
from ..evidence.contradiction import ContradictionDetector
from ..evidence.provenance import Answer, ProvenanceBuilder
from ..evidence.quality import SourcePolicy, SupportLevel
from ..evidence.verifier import ClaimVerifier, EvidenceItem, Verdict
from .planner import QueryPlanner
from .researcher import AbstentionAgent, EvidenceAgent, RetrievalAgent, TemporalAgent
from .synthesis import CitationAgent, SynthesisAgent


@dataclass
class VeritasConfig:
    k: int = 8
    candidate_k: int = 50
    max_iterations: int = 3
    synthesis_mode: str = "extractive"
    domain: str = "corporate"
    verbose: bool = True


class Veritas:
    """End-to-end pipeline. Everything is injected, so each stage is swappable
    (and each baseline in `eval/` is this pipeline with stages removed)."""

    def __init__(
        self,
        retriever,
        corpus: Dict[str, str],
        metadata: Dict[str, dict],
        store,
        graph=None,
        model=None,
        tokenizer=None,
        reranker=None,
        nli=None,
        config: Optional[VeritasConfig] = None,
    ) -> None:
        self.cfg = config or VeritasConfig()
        self.corpus = corpus
        self.metadata = metadata
        self.store = store
        self.graph = graph
        self.planner = QueryPlanner(model, tokenizer, self.cfg.domain)
        self.retrieval = RetrievalAgent(retriever, corpus, metadata, reranker)
        self.temporal = TemporalAgent(store)
        self.evidence = EvidenceAgent(SourcePolicy())
        self.verifier = ClaimVerifier(nli_model=nli)
        self.contradiction = ContradictionDetector(SourcePolicy())
        self.synthesis = SynthesisAgent(model, tokenizer)
        self.abstention = AbstentionAgent()
        self.provenance = ProvenanceBuilder(graph, store)

    # ------------------------------------------------------------------ run
    def answer(self, question: str, history: Optional[Sequence[str]] = None) -> Answer:
        t0 = time.time()
        trace: List[str] = []

        plan = self.planner.plan(question, history)
        trace.append(f"PLAN {plan.describe()}")

        # Refuse question SHAPES the evidence store cannot settle, before
        # spending a retrieval round. A forecast has no dated evidence because
        # the future has not been filed; a causal question needs an explanatory
        # source, and a table of figures is not one. Answering either with a
        # historical number looks authoritative and answers something else.
        from .planner import UNANSWERABLE_TYPES

        if plan.question_type in UNANSWERABLE_TYPES:
            why = ("This is a forecast. The knowledge base records only what has "
                   "been reported and when; it holds no projections, so there is "
                   "no evidence that could support a figure for a future period."
                   if plan.question_type == "FORECAST" else
                   "This asks for a cause. The knowledge base records what the "
                   "reported values were and when they were filed, not why they "
                   "changed; no explanatory source is on record.")
            trace.append(f"ABSTAIN unanswerable-type={plan.question_type}")
            ans = Answer(question=question,
                         answer=f"I cannot establish this from the available evidence. {why}")
            ans.abstained = True
            ans.support_level = SupportLevel.INSUFFICIENT
            ans.unknown = [why]
            ans.iterations = 0
            ans.trace = trace
            if plan.entity and plan.attribute:
                cur, valid_now = self.store.latest(plan.entity, plan.attribute)
                if cur is not None:
                    ans.current_status = (
                        f"For reference, the most recent recorded {plan.attribute} for "
                        f"{plan.entity} is {cur.value} covering "
                        f"{cur.valid_from.date()} to {cur.valid_to.date()} "
                        f"(source {cur.source_id}). That is a record, not an answer to "
                        f"the question asked.")
            return ans
        if self.cfg.verbose:
            print(f"[plan] {plan.describe()}")

        queries = list(plan.sub_queries)
        seen: Dict[str, EvidenceItem] = {}
        suff = None
        assessment = None
        iterations = 0

        for it in range(min(self.cfg.max_iterations, plan.max_iterations)):
            iterations = it + 1
            hits = self.retrieval.search(queries, plan, self.cfg.k, self.cfg.candidate_k)
            new = [h for h in hits if h.doc_id not in seen]
            for h in hits:
                seen[h.doc_id] = h
            trace.append(f"SEARCH#{iterations} queries={len(queries)} hits={len(hits)} new={len(new)}")

            evidence = list(seen.values())
            assessment = self.temporal.assess(plan, evidence)
            agreement = self._agreement(evidence, plan)
            suff = self.evidence.assess(plan, evidence, assessment, agreement)
            trace.append(f"SUFFICIENCY score={suff.score:.2f} label={suff.label} gaps={suff.gaps}")
            if self.cfg.verbose:
                print(f"[iter {iterations}] {len(evidence)} docs | score {suff.score:.2f} "
                      f"| {suff.label} | gaps: {suff.gaps or 'none'}")

            if suff.sufficient:
                trace.append("STOP evidence sufficient")
                break
            if not new and it > 0:
                trace.append("STOP no new evidence (refinement exhausted)")
                break
            queries = [self.planner.refine(plan, g) for g in suff.gaps] or queries

        evidence = list(seen.values())

        # ---- claim extraction + verification --------------------------------
        candidate_claims = []
        for e in evidence[: self.cfg.k]:
            # Only supply the entity as the subject hint when the document
            # ACTUALLY MENTIONS it. Forcing the hint unconditionally rewrites
            # every retrieved document's claims as if they were about the asked
            # entity -- which turned a NASA article about "aerospace workforce"
            # into a confident answer about Apple's headcount. The hint is a
            # coreference aid, not a licence to reattribute.
            hint = plan.entity if _mentions(e.text, plan.entity) else ""
            candidate_claims.extend(
                extract_claims(e.text, e.doc_id, e.doc_id, hint)
            )
        # Keep only claims about the asked-about entity AND attribute.
        #
        # There is deliberately NO fallback to "some arbitrary claims" when
        # nothing matches. An empty focused set means the corpus has documents
        # near the question but nothing that speaks to it -- which is exactly
        # the INSUFFICIENT case, and must reach the abstention gate rather than
        # be papered over with whatever ranked highest.
        focused = candidate_claims
        if plan.attribute:
            focused = [c for c in focused if c.attribute == plan.attribute]
        if plan.entity:
            # A claim with NO subject must be DROPPED, not kept. Keeping it let
            # an unrelated news item answer "What is Apple's headcount?" with
            # full confidence -- the precise failure this system exists to
            # prevent. If we cannot tell who a claim is about, it cannot be
            # evidence about a specific entity.
            ent = plan.entity.lower()
            focused = [c for c in focused
                       if c.subject and _entity_overlap(c.subject.lower(), ent)]
        if not plan.attribute:
            # The question asks about a property we could not map to a known
            # attribute ("number of unicorns"). Requiring a shared content word
            # with the question stops the system answering it with whatever
            # else it holds about the entity -- which is how "unicorns" got
            # answered with net income.
            qwords = {w for w in re.findall(r"[a-z]{4,}", plan.question.lower())
                      if w not in _QUESTION_STOP
                      and w not in plan.entity.lower()}
            if qwords:
                focused = [c for c in focused
                           if qwords & set(re.findall(r"[a-z]{4,}", c.text.lower()))]
        if not plan.attribute and not plan.entity:
            focused = candidate_claims[:6]

        # For a point-in-time question, drop claims sourced from documents whose
        # validity does not cover the anchor: an FY2021 filing cannot answer
        # "in 2020", and letting it through produced false CONFLICTED verdicts.
        if plan.temporal and plan.temporal.intent in ("HISTORICAL", "AS_OF")                 and not plan.needs_both_states:
            anchor = plan.temporal.anchor.isoformat()
            in_period = [c for c in focused
                         if not (self._doc_span(c.chunk_id) or self._doc_span(c.doc_id))
                         or _spans_overlap(self._doc_span(c.chunk_id)
                                           or self._doc_span(c.doc_id), (anchor, anchor))]
            focused = in_period or focused
        focused = focused[:12]

        anchor = plan.temporal.anchor.isoformat() if plan.temporal else None
        lookup = lambda c: self._evidence_for_claim(c, evidence)
        verdicts, metrics = self.verifier.verify_answer(focused, lookup, anchor)
        trace.append(f"VERIFY claims={metrics['n_checkable']} coverage={metrics['coverage']:.2f}")

        # ---- contradictions ---------------------------------------------------
        conflicts = []
        if plan.entity and plan.attribute:
            for a, b in self.store.conflicts(plan.entity, plan.attribute):
                conflicts.append(self.contradiction.compare_versions(a, b, plan.domain))
        for i, va in enumerate(verdicts):
            for vb in verdicts[i + 1:]:
                # Successive states are not a disagreement. Two claims whose
                # source documents cover non-overlapping validity intervals
                # describe different points on the timeline, and flagging them
                # as a conflict is the false positive that makes a
                # contradiction detector useless -- it would fire on every
                # entity that ever changed.
                if not self._intervals_overlap(va.claim, vb.claim):
                    continue
                c = self.contradiction.compare_claims(va.claim, vb.claim)
                if c.severity >= 0.5:
                    conflicts.append(c)
        trace.append(f"CONTRADICTIONS {len(conflicts)}")

        # ---- abstention -------------------------------------------------------
        n_supported = sum(v.verdict == Verdict.SUPPORTED for v in verdicts)
        n_stale = sum(v.verdict == Verdict.TEMPORAL_MISMATCH for v in verdicts)
        has_store_value = assessment is not None and assessment.current is not None
        abstain, reason = self.abstention.should_abstain(
            suff, metrics["coverage"], n_supported, n_stale, has_store_value)

        has_conflict = any(c.severity >= 0.7 for c in conflicts)
        label = SupportLevel.CONFLICTED if has_conflict else (suff.label if suff else SupportLevel.INSUFFICIENT)

        if abstain:
            answer_text = f"I cannot establish this from the available evidence. {reason}"
            trace.append(f"ABSTAIN {reason}")
            # The sufficiency score describes the RETRIEVED SET, which can look
            # healthy while none of it settles the question. Reporting
            # "HIGH SUPPORT" beside a refusal is self-contradictory and is
            # exactly the kind of confident-looking label this system exists to
            # avoid emitting.
            label = SupportLevel.INSUFFICIENT
        else:
            answer_text = self.synthesis.compose(
                plan, verdicts, assessment, self.verifier, lookup, self.cfg.synthesis_mode
            )
            trace.append(f"SYNTHESIS mode={self.cfg.synthesis_mode} chars={len(answer_text)}")

        ans = self.provenance.build(
            question=question, verdicts=verdicts, metrics=metrics,
            entity=plan.entity, attribute=plan.attribute, conflicts=conflicts,
            evidence_score=suff.score if suff else 0.0,
            factors=suff.factors if suff else None,
            support_level=label, answer_text=answer_text, trace=trace,
        )
        ans.abstained = abstain
        ans.iterations = iterations
        if abstain and reason:
            ans.unknown.insert(0, reason)
        audit = CitationAgent.audit(answer_text, verdicts)
        ans.trace.append(f"CITATION_AUDIT density={audit['citation_density']:.2f} "
                         f"uncited={len(audit['uncited_factual'])}")
        ans.trace.append(f"ELAPSED {time.time() - t0:.2f}s")
        return ans

    # -------------------------------------------------------------- helpers
    def _evidence_for_claim(self, claim, evidence: Sequence[EvidenceItem]) -> List[EvidenceItem]:
        """Route each claim to the evidence that can actually settle it.

        Two filters, in order:

        1. **Temporal scoping.** Evidence describing a DIFFERENT valid interval
           is neither support nor refutation -- it is about a different fact.
           Without this, "Apple revenue was 215.64B (FY2016)" is checked against
           "Apple revenue was 229.23B (FY2017)", the numbers differ, and every
           claim comes back CONFLICTED. That is the temporal-vs-factual
           confusion this whole system exists to prevent, so the verifier must
           not be handed cross-period evidence in the first place.
        2. **Lexical pre-filter.** Entailment is the expensive step, so it is
           not run on every (claim, doc) pair.
        """
        from ..evidence.verifier import lexical_entailment

        from ..evidence.claims import ATTRIBUTE_LEXICON

        claim_span = self._doc_span(claim.chunk_id) or self._doc_span(claim.doc_id)
        # An evidence passage about a DIFFERENT attribute cannot settle this
        # claim. Real SEC filings state revenue and net income for the same
        # fiscal period in the same sentence template, so without this filter a
        # revenue document "refutes" a net-income claim purely because the
        # numbers differ -- a false CONFLICTED on every financial question.
        words = ATTRIBUTE_LEXICON.get(claim.attribute, ()) if claim.attribute else ()
        pool: List[EvidenceItem] = []
        for e in evidence:
            if claim_span and (e.valid_from or e.valid_to):
                if not _spans_overlap(claim_span, (e.valid_from, e.valid_to)):
                    continue
            if words and not any(w in e.text.lower() for w in words):
                continue
            pool.append(e)
        pool = pool or list(evidence)

        scored = [(lexical_entailment(claim.text, e.text), e) for e in pool]
        scored.sort(key=lambda t: -t[0])
        return [e for sc, e in scored[:5] if sc > 0.15] or [e for _sc, e in scored[:2]]

    def _doc_span(self, doc_id: str):
        md = self.metadata.get(doc_id)
        if not md:
            return None
        vf, vt = md.get("valid_from"), md.get("valid_to")
        return (vf, vt) if (vf or vt) else None

    def _intervals_overlap(self, a, b) -> bool:
        """Do the two claims' source documents cover overlapping valid time?"""
        sa = self._doc_span(a.chunk_id) or self._doc_span(a.doc_id) or (a.valid_from, a.valid_to)
        sb = self._doc_span(b.chunk_id) or self._doc_span(b.doc_id) or (b.valid_from, b.valid_to)
        return _spans_overlap(sa, sb)

    def _agreement(self, evidence: Sequence[EvidenceItem], plan) -> float:
        """Fraction of evidence agreeing on the modal value, WITHIN a period.

        Scoped by valid interval for the same reason evidence routing is: two
        revenue figures for different fiscal years are not a disagreement, and
        counting them as one drove every answer to CONFLICTED on real SEC data.
        Only evidence overlapping the asked-about time is compared.
        """
        if not plan.attribute or len(evidence) < 2:
            return 1.0
        from collections import Counter

        anchor = plan.temporal.anchor if plan.temporal else None
        vals: Counter = Counter()
        for e in evidence:
            if anchor is not None and (e.valid_from or e.valid_to):
                if not _spans_overlap((e.valid_from, e.valid_to),
                                      (anchor.isoformat(), anchor.isoformat())):
                    continue
            hint = plan.entity if _mentions(e.text, plan.entity) else ""
            for c in extract_claims(e.text, e.doc_id, e.doc_id, hint):
                if c.attribute == plan.attribute and c.value:
                    vals[c.value.lower()] += 1
        if not vals:
            return 1.0
        return vals.most_common(1)[0][1] / sum(vals.values())


#: Interrogative and filler words that carry no topical content.
_QUESTION_STOP = frozenset(
    "what when where which whose there does have with from that this been were "
    "about into over under many much number count current currently latest "
    "please tell know".split()
)


def _spans_overlap(a, b) -> bool:
    """Do two (valid_from, valid_to) pairs overlap? Unknown bounds are open."""
    from ..temporal.versioning import BEGINNING, FOREVER, _dt

    af = _dt(a[0]) if a[0] else BEGINNING
    at = _dt(a[1]) if a[1] else FOREVER
    bf = _dt(b[0]) if b[0] else BEGINNING
    bt = _dt(b[1]) if b[1] else FOREVER
    return af < bt and bf < at


def _mentions(text: str, entity: str) -> bool:
    """Does this text actually refer to the entity?

    Requires every distinctive token of the entity name to appear (corporate
    suffixes ignored), so "Apple Inc" matches a document saying "Apple" but a
    document mentioning neither matches nothing.
    """
    if not entity:
        return False
    import re as _re

    suffixes = {"inc", "incorporated", "corp", "corporation", "co", "company",
                "ltd", "limited", "plc", "llc", "holdings", "group", "the"}
    low = text.lower()
    toks = [t for t in _re.findall(r"[a-z0-9]+", entity.lower())
            if t not in suffixes and len(t) > 2]
    return bool(toks) and all(t in low for t in toks)


def _entity_overlap(subject: str, entity: str) -> bool:
    """Token-overlap entity match: "Acme" matches "Acme Industries", but
    "Helios Energy" does not match "Nova Logistics". Substring matching alone
    would let a one-character company name match everything."""
    a = {w for w in subject.replace(".", " ").split() if len(w) > 2}
    b = {w for w in entity.replace(".", " ").split() if len(w) > 2}
    return bool(a & b) if (a and b) else subject in entity or entity in subject
