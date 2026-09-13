"""End-to-end smoke tests. Run: python -m pytest tests/ -q  (or: python tests/test_smoke.py)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from veritas.eval.benchmark import build_seed_benchmark
from veritas.model.transformer import ModelConfig, VeritasLM
from veritas.pipeline import VeritasSystemBuilder, load_benchmark_corpus
from veritas.tokenizer.bpe import BPETokenizer


def build_tiny_system():
    bench = build_seed_benchmark()
    texts = [d.text for i in bench.items for d in i.docs] + [i.question for i in bench.items]
    tok = BPETokenizer.train(texts * 4, vocab_size=1200, verbose=False)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4,
                      n_kv_heads=2, max_seq_len=256)
    model = VeritasLM(cfg).eval()
    b = VeritasSystemBuilder(model, tok, device="cpu")
    load_benchmark_corpus(b, bench)
    return b, b.build(), bench


def test_pretokenizer_is_lossless():
    """Guards the catch-all in SPLIT_PATTERN: without it, unmatched characters
    (CJK, emoji, rare symbols) are silently dropped by findall."""
    from veritas.tokenizer.bpe import assert_lossless

    for s in ["plain ascii", "日本語 and emoji 🚀✓",
              "$1,412,000,000 (≈ €1.31B)", "<|evidence|>[E1]", "", "\n\n  \t"]:
        assert_lossless(s)


def test_tokenizer_roundtrip():
    tok = BPETokenizer.train(["the quick brown fox " * 50, "Acme Industries CEO 2026"],
                             vocab_size=400, verbose=False)
    s = "Acme Industries CEO — 2026 ✓ 日本語 🚀"
    assert tok.decode(tok.encode(s)) == s


def test_model_shapes_and_cache():
    cfg = ModelConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64)
    m = VeritasLM(cfg).eval()
    x = torch.randint(0, 300, (2, 16))
    logits, loss = m(x, x)
    assert logits.shape == (2, 16, 300) and loss.item() > 0
    with torch.inference_mode():
        a = m.generate(x[:, :4], max_new_tokens=6, temperature=0.0, use_cache=True)
        b = m.generate(x[:, :4], max_new_tokens=6, temperature=0.0, use_cache=False)
    assert torch.equal(a, b), "KV cache changed the greedy output"


def test_temporal_store_current_vs_historical():
    from veritas.temporal.versioning import TemporalStore

    s = TemporalStore()
    s.assert_fact("Acme", "ceo", "X", valid_from="2024-01-01", source_id="a")
    s.assert_fact("Acme", "ceo", "Y", valid_from="2026-01-01", source_id="b")
    assert s.current("Acme", "ceo").value == "Y"
    assert s.as_of("Acme", "ceo", "2025-01-01").value == "X"
    assert [v.value for v in s.outdated("Acme", "ceo")] == ["X"]


def test_change_detection_skips_cosmetic_edits():
    from veritas.temporal.change_detection import ChangeDetector

    d = ChangeDetector()
    base = "Acme reported revenue of 1.2 billion for the year. " * 10
    assert d.check("s1", base).changed                      # first sight
    assert not d.check("s1", base).changed                  # identical
    assert not d.check("s1", base + "\nLast updated 12:31:02").changed   # cosmetic
    assert d.check("s1", base.replace("1.2 billion", "1.4 billion")).changed


def test_bm25_ranks_exact_terms():
    from veritas.rag.bm25 import BM25

    idx = BM25()
    idx.add("d1", "Marcus Lund was appointed chief executive of Acme in 2026")
    idx.add("d2", "Acme sells industrial pumps and valves worldwide")
    idx.add("d3", "quarterly results for the logistics sector")
    idx.finalize()
    assert idx.search("who is the chief executive of Acme", k=2)[0][0] == "d1"


def test_verifier_catches_numeric_conflict():
    from veritas.evidence.claims import extract_claims
    from veritas.evidence.verifier import ClaimVerifier, EvidenceItem, Verdict

    claims = extract_claims("Nova Logistics opened 15 offices in India in 2026.",
                            "gen", "gen", "Nova Logistics")
    ev = [EvidenceItem("d1", "Nova Logistics opened 12 offices in India in 2026.",
                       "reuters.com", "2026-08-14", 2)]
    v = ClaimVerifier().verify_claim(claims[0], ev)
    assert v.verdict in (Verdict.REFUTED, Verdict.CONFLICTED), v.verdict


def test_end_to_end_current_question():
    _b, veritas, bench = build_tiny_system()
    item = [i for i in bench.items if i.qid == "q_current_ceo"][0]
    ans = veritas.answer(item.question)
    assert ans.question == item.question
    assert ans.iterations >= 1
    assert ans.timeline, "timeline should be reconstructed from the temporal store"
    assert "## Evidence" in ans.to_markdown()
    assert "100%" not in ans.to_markdown()   # spec section 24


def test_abstains_when_no_evidence():
    _b, veritas, bench = build_tiny_system()
    item = [i for i in bench.items if i.qid == "q_insufficient"][0]
    ans = veritas.answer(item.question)
    assert ans.abstained, f"should abstain, got: {ans.answer[:120]}"


def test_new_source_changes_answer_without_retraining():
    b, veritas, _bench = build_tiny_system()
    q = "Who is the current CEO of Acme Industries?"
    before = veritas.answer(q)
    b.add_document("sec.gov", "acme_2027_8k",
                   "Acme Industries filing: Yuki Tanaka was appointed chief executive "
                   "effective March 2027.", "2027-03-02", "Acme Industries")
    after = veritas.answer(q)
    assert b.store.current("Acme Industries", "ceo") is not None
    assert before.timeline != after.timeline or before.answer != after.answer


def test_bare_year_is_year_only():
    from veritas.temporal.temporal_retrieval import parse_temporal_query

    assert parse_temporal_query("What was Microsoft revenue in 2021?").year_only
    assert not parse_temporal_query("What was Microsoft revenue in March 2021?").year_only


def test_fiscal_year_anchors_on_period_ending_that_year():
    """FY2021 for a July-June company is 2020-07-01..2021-06-30. The old
    mid-year anchor (2021-07-01) fell inside FY2022 and returned its figure."""
    b, veritas, _bench = build_tiny_system()
    for vf, vt, val in (("2020-07-01", "2021-06-30", "168.09 billion USD"),
                        ("2021-07-01", "2022-06-30", "198.27 billion USD")):
        b.store.assert_fact("Contoso Corporation", "revenue", val, valid_from=vf,
                            valid_to=vt, recorded_at="2022-07-28", source_id="sec.gov")
    ans = veritas.answer("What was Contoso revenue in 2021?")
    assert any("FISCAL_YEAR 2021 -> 2020-07-01..2021-06-30" in t for t in ans.trace), ans.trace


def test_entity_overlap_ignores_corporate_suffixes():
    from veritas.agents.orchestrator import _entity_overlap

    assert not _entity_overlap("uber technologies inc.", "apple inc")
    assert _entity_overlap("apple inc", "apple inc.")


def test_person_claims_compare_names_not_years():
    from veritas.evidence.claims import extract_claims
    from veritas.evidence.verifier import ClaimVerifier, EvidenceItem, Verdict

    text = "Apple Inc. named Tim Cook chief executive officer, effective 2011-08-24."
    c = extract_claims(text, "d0", "d0", "Apple Inc.")[0]
    assert c.value == "Tim Cook" and c.numeric is None, (c.value, c.numeric)
    same = EvidenceItem("d1", text, "wikidata.org")
    other = EvidenceItem("d2", "Apple Inc. named Steve Jobs chief executive officer, "
                               "effective 1997-09.", "wikidata.org")
    v = ClaimVerifier()
    assert v.verify_claim(c, [same]).verdict == Verdict.SUPPORTED
    assert v.verify_claim(c, [same, other]).verdict == Verdict.CONFLICTED


def test_wiki_markup_fragments_are_rejected():
    from veritas.ingest.real_sources import _clean_wiki_value

    assert _clean_wiki_value("revenue", "{{US$") == ""
    assert _clean_wiki_value("ceo", "{{Unbulleted list") == ""
    assert _clean_wiki_value("ceo", "Gautam Adani {{small") == "Gautam Adani"


def test_wikitext_cleaner_keeps_list_items_and_amounts():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from fetch_real_data import _infobox_field, clean_wikitext

    box = ("{{Infobox company\n"
           "| key_people = {{ubl|Arthur Levinson {{small|(chairman)}}|Tim Cook {{small|(CEO)}}}}\n"
           "| revenue = {{increase}} {{US$|391.04 billion|link=yes}} (2024)\n"
           "| website = example.com\n}}")
    people = clean_wikitext(_infobox_field(box, "key_people"))
    assert "Tim Cook (CEO)" in people, people
    revenue = clean_wikitext(_infobox_field(box, "revenue"))
    assert "US$ 391.04 billion" in revenue, revenue


def test_streaming_pipeline_suppresses_unchanged():
    """The bus must run identically with or without Kafka, and stage 2 must
    stop unchanged documents before they reach the expensive stages."""
    from veritas.ingest.streaming import (
        LocalBus, StreamingIngest, TOPIC_CHANGED, TOPIC_RAW, make_bus,
    )

    b, _veritas, _bench = build_tiny_system()
    si = StreamingIngest(b.ingest, LocalBus())

    doc = ("Helios Energy filing: revenue for the fiscal year was 4.2 billion euros "
           "and construction started at the Almeria site.")
    si.publish_raw("sec.gov", doc, "helios_1", "2026-03-01", "Helios Energy")
    si.publish_raw("sec.gov", doc, "helios_1", "2026-03-01", "Helios Energy")  # identical
    stats = si.run_once()

    assert stats[TOPIC_RAW] == 2, stats
    assert stats["suppressed"] == 1, f"identical re-poll must be suppressed: {stats}"
    assert stats[TOPIC_CHANGED] == 1, stats

    # No broker configured -> in-process bus, same interface.
    assert isinstance(make_bus(None), LocalBus)


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as e:
                print(f"FAIL {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
