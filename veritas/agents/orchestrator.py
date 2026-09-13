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
    def answer(self, question: str, history: Optional[Sequence[str]] = None,
               mode: Optional[str] = None) -> Answer:
        """Answer one question. `mode` ("extractive" | "generative") applies to
        this call only; the configured default is used when it is omitted."""
        t0 = time.time()
        trace: List[str] = []
        mode = mode or self.cfg.synthesis_mode

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

        # A comparison is several point questions plus an honest join: each side
        # is answered and verified on its own timeline, then placed beside the
        # others with its own period. A side the store has no record of is named
        # as unestablished rather than silently dropped.
        if plan.question_type == "COMPARISON" and plan.attribute:
            entities = self._comparison_entities(question, plan.attribute)
            unmatched = [n for n in self.planner.candidates(question)
                         if not self._comparison_entities(n, plan.attribute)
                         and not self._is_attribute_phrase(n)]
            if len(entities) + len(unmatched) >= 2:
                trace.append(f"COMPARISON entities={entities} unrecognised={unmatched}")
                return self._compare(question, plan, entities, unmatched, mode, trace, t0)
        if self.cfg.verbose:
            print(f"[plan] {plan.describe()}")

        # "Revenue in 2021" names a FISCAL year, and a fiscal year is labelled
        # by the calendar year it ENDS in: Microsoft's FY2021 runs July 2020 to
        # June 2021. A fixed mid-year anchor lands inside the next year's period
        # for every company whose year does not end in December, which answered
        # "Microsoft revenue in 2021" with the FY2022 figure. Re-anchor on the
        # annual period that actually ends in the asked year.
        tq = plan.temporal
        if tq is not None and tq.year_only and plan.entity and plan.attribute:
            year = tq.anchor.year
            annual = [v for v in self.store.history(plan.entity, plan.attribute)
                      if v.valid_to.year == year
                      and 300 <= (v.valid_to - v.valid_from).days <= 380]
            if annual:
                p = max(annual, key=lambda v: v.recorded_at)
                tq.anchor = p.valid_from + (p.valid_to - p.valid_from) / 2
                tq.fiscal_period = (p.valid_from, p.valid_to, year)
                trace.append(f"FISCAL_YEAR {year} -> {p.valid_from.date()}..{p.valid_to.date()}")

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

        # ---- the stored value the answer states -------------------------------
        # The headline comes from the temporal store, so the documents that
        # asserted that value must be in the evidence and cited. Retrieval alone
        # missed them ("CEO" never appears in "chief executive officer"), which
        # left "Apple's CEO is ..." with no citation at all.
        head = self._head_versions(plan, assessment)
        head_docs = {eid for v in head for eid in v.evidence_ids}
        # A document backing only a SUPERSEDED version is the prior record, not
        # a rival source: Apple's original FY2008 10-K (4.83B) must not "refute"
        # its own restatement (6.12B). The original stays in the timeline and in
        # the synthesis note; it just does not vote in verification.
        if plan.entity and plan.attribute:
            superseded_docs = {
                eid for v in self.store.history(plan.entity, plan.attribute,
                                                include_superseded=True)
                if v.superseded_at is not None for eid in v.evidence_ids} - head_docs
            if superseded_docs:
                kept = [e for e in evidence
                        if self.metadata.get(e.doc_id, {}).get("doc_id") not in superseded_docs]
                if len(kept) < len(evidence):
                    trace.append(f"SUPERSEDED excluded={len(evidence) - len(kept)} documents")
                evidence = kept
        head_evidence = [e for e in evidence
                         if self.metadata.get(e.doc_id, {}).get("doc_id") in head_docs]
        have = {e.doc_id for e in evidence}
        for cid, md in self.metadata.items():
            if md.get("doc_id") in head_docs and cid not in have:
                item = EvidenceItem(
                    doc_id=cid, text=self.corpus.get(cid, ""),
                    source=str(md.get("source", cid)), date=md.get("date"),
                    tier=int(md.get("tier", 3)), valid_from=md.get("valid_from"),
                    valid_to=md.get("valid_to"), score=1.0)
                evidence.append(item)
                head_evidence.append(item)
                have.add(cid)
        if head:
            trace.append(f"HEAD versions={len(head)} documents={len(head_evidence)}")

        # ---- claim extraction + verification --------------------------------
        candidate_claims = []
        # Claims are drawn from the documents behind the stated value when there
        # are any; every retrieved document still takes part in verifying them.
        for e in (head_evidence or evidence[: self.cfg.k]):
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
                # Only a dispute about the period the answer states counts
                # against it: Apple's FY2008 restatement is no reason to label
                # its FY2020 net income CONFLICTED.
                if head and not any(a.valid_from < h.valid_to and h.valid_from < a.valid_to
                                    for h in head):
                    continue
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
        # For a point-in-time question the store value only counts when it is
        # the one being stated; the latest value does not answer "in 1990".
        if plan.temporal and plan.temporal.intent in ("CURRENT", "HISTORICAL", "AS_OF"):
            has_store_value = bool(head)
        else:
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
                plan, verdicts, assessment, self.verifier, lookup, mode
            )
            trace.append(f"SYNTHESIS mode={mode} chars={len(answer_text)}")

        ans = self.provenance.build(
            question=question, verdicts=verdicts, metrics=metrics,
            entity=plan.entity, attribute=plan.attribute, conflicts=conflicts,
            evidence_score=suff.score if suff else 0.0,
            factors=suff.factors if suff else None,
            support_level=label, answer_text=answer_text, trace=trace,
        )
        ans.abstained = abstain
        ans.iterations = iterations
        # Not a dataclass field, so never serialised: the stored versions the
        # headline states, for _compare to line up without re-deriving them.
        ans._head = head
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
        # Evidence about a DIFFERENT company can neither support nor refute the
        # claim. Without this, Uber's revenue for an overlapping period
        # "refuted" Apple's, and the fallback below brought it back even when
        # every same-company document had been filtered out.
        same_subject = [e for e in evidence
                        if not re.search(r"[a-z0-9]{3,}", claim.subject.lower())
                        or _mentions(e.text, claim.subject)]
        pool: List[EvidenceItem] = []
        for e in same_subject:
            if claim_span and (e.valid_from or e.valid_to):
                if not _spans_overlap(claim_span, (e.valid_from, e.valid_to)):
                    continue
            if words and not any(w in e.text.lower() for w in words):
                continue
            pool.append(e)
        pool = pool or same_subject

        scored = [(lexical_entailment(claim.text, e.text), e) for e in pool]
        scored.sort(key=lambda t: -t[0])
        return [e for sc, e in scored[:5] if sc > 0.15] or [e for _sc, e in scored[:2]]

    @staticmethod
    def _head_versions(plan, assessment) -> List:
        """The stored versions the answer's headline states (mirrors synthesis).

        Empty for a change question, whose answer is the whole timeline, and
        for a point-in-time question with nothing on record for that time.
        """
        if assessment is None or not plan.temporal or not plan.attribute:
            return []
        intent = plan.temporal.intent
        if intent == "CURRENT":
            return [assessment.current] if assessment.current is not None else []
        if intent in ("HISTORICAL", "AS_OF"):
            anchor = plan.temporal.anchor
            return [h for h in assessment.historical if h.valid_from <= anchor < h.valid_to]
        return []

    # ------------------------------------------------------------ comparison
    #: First tokens too generic to name a company alone: "Meta" names Meta
    #: Platforms, but "General" does not name General Motors.
    _GENERIC_HEADS = frozenset(
        "general international advanced united american national first new global".split())
    #: Weakest first. A comparison is only as well supported as its weakest side.
    _LEVEL_RANK = [SupportLevel.INSUFFICIENT, SupportLevel.CONFLICTED, SupportLevel.LOW,
                   SupportLevel.MODERATE, SupportLevel.HIGH]

    def _comparison_entities(self, text: str, attribute: str) -> List[str]:
        """Entities with a recorded `attribute` that `text` names, in order of mention.

        Matched against the store's own names, not capitalised phrases: only
        timelines the store holds can be compared. A name matches by all of its
        core tokens ("Meta Platforms"), by a distinctive first token ("Meta"),
        or by its initials written in capitals ("IBM", "AMD").
        """
        words = re.findall(r"[a-z0-9]+", text.lower())
        caps = set(re.findall(r"\b[A-Z]{2,5}\b", text))
        found = []
        for ent in self.store.entities():
            if not self.store.history(ent, attribute):
                continue
            core = [t for t in re.findall(r"[a-z0-9]+", ent.lower())
                    if t not in _CORPORATE_SUFFIXES]
            if not core:
                continue
            initials = "".join(t[0] for t in core).upper()
            if all(t in words for t in core) or (
                    len(core[0]) >= 4 and core[0] not in self._GENERIC_HEADS
                    and core[0] in words):
                found.append((words.index(core[0]), ent))
            elif len(core) >= 2 and initials in caps:
                found.append((words.index(initials.lower()), ent))
        return list(dict.fromkeys(ent for _pos, ent in sorted(found)))

    @staticmethod
    def _is_attribute_phrase(phrase: str) -> bool:
        """A capitalised phrase made only of attribute words ("CEO", "Revenue")."""
        from ..evidence.claims import ATTRIBUTE_LEXICON

        vocab = {t for words in ATTRIBUTE_LEXICON.values() for w in words for t in w.split()}
        vocab |= {"fy", "usd", "q1", "q2", "q3", "q4"}
        toks = re.findall(r"[a-z0-9]+", phrase.lower())
        return bool(toks) and all(t in vocab for t in toks)

    def _compare(self, question, plan, entities, unmatched, mode, trace, t0) -> Answer:
        """Answer each side as its own question, then join them without lying.

        Three rules make the join honest: every value keeps its own period (two
        companies' "2023" are different fiscal windows, and the answer says so);
        a computed difference is labelled as derived, never as reported; and the
        comparison refuses unless at least two sides are actually established.
        """
        from ..evidence.claims import PERSON_ATTRIBUTES
        from .synthesis import _label

        tq = plan.temporal
        label = _label(plan.attribute)
        person = plan.attribute in PERSON_ATTRIBUTES
        change = tq is not None and tq.intent == "CHANGE"
        year = (tq.anchor.year if tq is not None and tq.explicit_dates
                and tq.intent in ("HISTORICAL", "AS_OF") else None)

        def sub_question(ent: str) -> str:
            if change:
                return f"How did the {label} of {ent} change over time?"
            if person:
                return (f"Who was the {label} of {ent} in {year}?" if year
                        else f"Who is the current {label} of {ent}?")
            return f"What was {ent} {label} in {year}?" if year else f"What is {ent} {label}?"

        def row(entity, q=None, v=None, level=SupportLevel.INSUFFICIENT, cites=(), text=""):
            ok = v is not None
            return {
                "entity": entity, "question": q, "established": ok,
                "value": v.value if ok else None,
                "valid_from": v.valid_from.date().isoformat() if ok else None,
                "valid_to": (("present" if v.valid_to.year > 9000
                              else v.valid_to.date().isoformat()) if ok else None),
                "recorded_at": v.recorded_at.date().isoformat() if ok else None,
                "source": v.source_id if ok else None,
                "restated": bool(ok and v.change_kind == "CORRECTED"),
                "support_level": level if ok else SupportLevel.INSUFFICIENT,
                "citations": list(cites), "answer": text,
            }

        ans = Answer(question=question)
        rows, done, coverage, scores = [], [], [], []
        for ent in entities:
            q = sub_question(ent)
            sub = self.answer(q, None, mode)
            trace.append(f"COMPARE {ent}: {q!r} -> "
                         f"{'ABSTAIN' if sub.abstained else sub.support_level}")
            trace.extend(f"  [{ent}] {line}" for line in sub.trace)

            # Citation markers restart at [E1] in every sub-answer; renumber so
            # each marker names exactly one document across the whole answer.
            remap: Dict[str, str] = {}
            for c in sub.citations:
                new = f"[E{len(ans.citations) + 1}]"
                remap[c.marker] = new
                c.marker = new
                ans.citations.append(c)

            def relabel(text: str, remap=remap) -> str:
                return re.sub(r"\[E\d+\]", lambda m: remap.get(m.group(0), m.group(0)), text)

            for cl in sub.claims:
                cl["citations"] = [remap.get(m, m) for m in cl.get("citations", [])]
                cl["note"] = relabel(cl.get("note", ""))
                ans.claims.append(cl)
            ans.conflicts += [f"{ent}: {relabel(x)}" for x in sub.conflicts]
            ans.unknown += [f"{ent}: {u}" for u in sub.unknown]
            ans.iterations += sub.iterations

            head = getattr(sub, "_head", [])
            v = head[-1] if head else None
            if v is None and change and not sub.abstained:
                v, _valid_now = self.store.latest(ent, plan.attribute)
            if sub.abstained:
                v = None
            rows.append(row(ent, q, v, sub.support_level, remap.values(), relabel(sub.answer)))
            if v is not None:
                done.append((rows[-1], v))
                coverage.append(sub.coverage)
                scores.append(sub.evidence_score)
                if sub.last_verified and (ans.last_verified is None
                                          or sub.last_verified > ans.last_verified):
                    ans.last_verified = sub.last_verified

        for name in unmatched:
            trace.append(f"COMPARE {name}: no {plan.attribute} on record")
            rows.append(row(name))
            ans.unknown.append(f"{name}: no {label} is on record for this entity.")

        ans.comparison = rows
        missing = [r["entity"] for r in rows if not r["established"]]
        if len(done) < 2:
            why = f"the {label} could not be established for {', '.join(missing)}"
            ans.answer = f"I cannot establish this comparison from the available evidence: {why}."
            ans.abstained = True
            ans.support_level = SupportLevel.INSUFFICIENT
            ans.unknown.insert(0, f"A comparison needs at least two established values; {why}.")
            trace.append(f"ABSTAIN comparison established={len(done)} missing={missing}")
        else:
            if change:
                parts = [f"{r['entity']}: {r['answer']}" for r, _v in done]
            else:
                parts = [f"{r['entity']}: {r['value']} ({r['valid_from']} to {r['valid_to']}, "
                         f"{r['source']}{', restated' if r['restated'] else ''})"
                         + (" " + " ".join(r["citations"]) if r["citations"] else "")
                         for r, _v in done]
            text = (f"Comparing {label}" + (f" for {year}" if year else "") + " -- "
                    + "; ".join(parts) + ".")
            if not person and not change:
                text += self._compare_amounts([r for r, _v in done])
                starts = [v.valid_from for _r, v in done]
                if (max(starts) - min(starts)).days > 31:
                    text += (" The periods are not aligned: each figure covers that company's "
                             "own fiscal year, so this compares fiscal years, not one "
                             "calendar window.")
            if missing:
                text += f" Not established: {', '.join(missing)}."
            ans.answer = text
            ans.support_level = min((r["support_level"] for r, _v in done),
                                    key=lambda lv: self._LEVEL_RANK.index(lv)
                                    if lv in self._LEVEL_RANK else 0)
            ans.coverage = min(coverage)
            ans.evidence_score = min(scores)
            trace.append(f"COMPARISON joined={len(done)} support={ans.support_level}")
        trace.append(f"ELAPSED {time.time() - t0:.2f}s")
        ans.trace = trace
        return ans

    @staticmethod
    def _amount(value: str):
        """(amount, unit) from a stored value like "-1.23 billion USD", else None."""
        m = re.fullmatch(r"\s*(-?\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion)?\s*([A-Za-z]{3})?\s*",
                         value or "")
        if not m:
            return None
        scale = {"thousand": 1e3, "million": 1e6, "billion": 1e9}.get((m.group(2) or "").lower(), 1.0)
        return float(m.group(1).replace(",", "")) * scale, (m.group(3) or "").upper()

    @classmethod
    def _compare_amounts(cls, rows) -> str:
        """The spread between the highest and lowest figure, labelled as derived."""
        amounts = [(cls._amount(r["value"]), r["entity"]) for r in rows]
        if any(a is None for a, _e in amounts) or len({a[1] for a, _e in amounts}) != 1:
            return ""   # a non-numeric value or mixed units: no arithmetic across them
        unit = amounts[0][0][1]
        ranked = sorted(((a[0], e) for a, e in amounts), key=lambda t: -t[0])
        (hi, top), (lo, bottom) = ranked[0], ranked[-1]
        if hi == lo:
            return " The values are equal."
        ratio = f" ({hi / lo:.2f}x)" if lo > 0 else ""
        diff = hi - lo
        for div, suf in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
            if diff >= div:
                spread = f"{diff / div:.2f} {suf} {unit}".strip()
                break
        else:
            spread = f"{diff:,.0f} {unit}".strip()
        return (f" {top} is higher than {bottom} by {spread}{ratio}; the difference is "
                f"computed from the cited figures, not reported by either source.")

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
            # Other companies' values are not votes on this one: AMD's and
            # Apple's CEOs made "Microsoft's CEO in 2020" look disputed.
            if plan.entity and not hint:
                continue
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


_CORPORATE_SUFFIXES = frozenset(
    "inc incorporated corp corporation co company ltd limited plc llc lp "
    "holdings group nv sa ag se the".split()
)


def _entity_overlap(subject: str, entity: str) -> bool:
    """Is a claim whose subject is `subject` about the asked `entity`?

    Every distinctive token of the asked entity must appear in the subject:
    "Microsoft" matches "Microsoft Corporation", "Acme" matches "Acme
    Industries". This is the same containment rule `TemporalStore.canonical`
    uses to resolve names, so the claim filter and the store agree.

    It used to accept ANY shared token. "Quillon Robotics Qwkzmpd" -- a company
    the corpus knows nothing about -- then shared "quillon" and "robotics" with
    another company's filing, and the system answered with that company's CEO
    at HIGH SUPPORT instead of abstaining. The same rule would answer a question
    about "Apple Hospitality" with Apple Inc.'s facts.

    Corporate suffixes carry no identity: "Uber Technologies Inc." shares only
    "Inc" with "Apple Inc", and counting it put Uber's revenue into an answer
    about Apple as a "disagreeing source"."""
    def toks(s: str) -> set:
        return {w for w in re.findall(r"[a-z0-9]+", s.lower())
                if len(w) > 2 and w not in _CORPORATE_SUFFIXES}

    a, b = toks(subject), toks(entity)
    return b <= a if (a and b) else subject in entity or entity in subject
