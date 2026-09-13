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
    assert _entity_overlap("microsoft corporation", "microsoft")
    # an unknown company sharing words with a known one is NOT that company
    assert not _entity_overlap("quillon robotics 6a93", "quillon robotics qwkzmpd")
    assert not _entity_overlap("apple inc.", "apple hospitality")


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


def _sec_rows_file(rows):
    """Write SEC-shaped rows to a temp .jsonl, so tests use the real loader."""
    import json
    import tempfile

    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    with f:
        for (entity, cik, attr, value, vf, vt, filed) in rows:
            f.write(json.dumps({
                "entity": entity, "cik": cik, "attribute": attr, "value": value,
                "unit": "USD", "valid_from": vf, "valid_to": vt, "filed": filed,
                "form": "10-K", "accn": f"{cik}-{filed}", "restatement": False}) + "\n")
    return f.name


def _load_sec(b, rows):
    import os

    from veritas.ingest.real_sources import load_sec_facts

    path = _sec_rows_file(rows)
    try:
        load_sec_facts(b, path, verbose=False)
    finally:
        os.unlink(path)


def test_primary_revenue_tag_wins_within_one_filing():
    """Walmart's FY2020 10-K reports Revenues (523.96B) and the narrower
    contract-revenue tag (519.93B) for one period. The total must win, not
    whichever row happened to load last."""
    import json
    import os

    from veritas.ingest.real_sources import load_sec_facts

    b, _veritas, _bench = build_tiny_system()
    path = _sec_rows_file([])
    with open(path, "w", encoding="utf-8") as f:
        for tag, value in (("Revenues", 523964000000),
                           ("RevenueFromContractWithCustomerExcludingAssessedTax", 519926000000)):
            f.write(json.dumps({
                "entity": "Contoso Corporation", "cik": "1", "attribute": "revenue",
                "value": value, "unit": "USD", "valid_from": "2019-02-01",
                "valid_to": "2020-01-31", "filed": "2020-03-20", "form": "10-K",
                "accn": "a1", "tag": tag, "restatement": False}) + "\n")
    try:
        load_sec_facts(b, path, verbose=False)
    finally:
        os.unlink(path)
    hist = b.store.history("Contoso Corporation", "revenue")
    assert [v.value for v in hist] == ["523.96 billion USD"], [v.value for v in hist]
    assert not b.store.conflicts("Contoso Corporation", "revenue")


def test_same_source_restatement_is_a_correction_not_a_conflict():
    from veritas.temporal.versioning import ChangeKind, TemporalStore

    s = TemporalStore()
    period = dict(valid_from="2007-09-30", valid_to="2008-09-27")
    s.assert_fact("Acme", "net_income", "4.83 billion USD", recorded_at="2009-10-27",
                  source_id="sec.gov", **period)
    _v, ev = s.assert_fact("Acme", "net_income", "6.12 billion USD", recorded_at="2010-01-25",
                           source_id="sec.gov", **period)
    assert ev.kind == ChangeKind.CORRECTED, ev.kind
    assert s.as_of("Acme", "net_income", "2008-06-01").value == "6.12 billion USD"
    # transaction-time travel: before the restatement was filed, and before anything was
    assert s.as_of("Acme", "net_income", "2008-06-01",
                   known_at="2009-12-01").value == "4.83 billion USD"
    assert s.as_of("Acme", "net_income", "2008-06-01", known_at="2009-01-01") is None
    assert not s.conflicts("Acme", "net_income")
    assert [v.value for v in s.belief_history("Acme", "net_income", "2008-06-01")] == [
        "4.83 billion USD", "6.12 billion USD"]
    # a DIFFERENT source disputing the same period is still a conflict
    _v, ev = s.assert_fact("Acme", "net_income", "5.00 billion USD", recorded_at="2010-02-01",
                           source_id="reuters.com", **period)
    assert ev.kind == ChangeKind.CONFLICT, ev.kind


def test_comparison_names_each_side_and_flags_unaligned_periods():
    b, veritas, _bench = build_tiny_system()
    _load_sec(b, [
        ("Contoso Corporation", "0000000001", "revenue", 211915000000,
         "2022-07-01", "2023-06-30", "2023-07-27"),
        ("Fabrikam Inc.", "0000000002", "revenue", 383285000000,
         "2022-09-25", "2023-09-30", "2023-11-03"),
    ])
    ans = veritas.answer("Compare Contoso and Fabrikam revenue in 2023")
    assert not ans.abstained, ans.answer
    assert [r["entity"] for r in ans.comparison] == ["Contoso Corporation", "Fabrikam Inc."]
    # 211,915,000,000 renders as 211.91 (the float sits just below .915)
    assert "211.91 billion USD" in ans.answer and "383.29 billion USD" in ans.answer, ans.answer
    assert "Fabrikam Inc. is higher than Contoso Corporation by 171.37" in ans.answer or \
        "Fabrikam Inc. is higher than Contoso Corporation by 171.38" in ans.answer, ans.answer
    assert "not reported by either source" in ans.answer, ans.answer
    assert "not aligned" in ans.answer, ans.answer
    markers = [c.marker for c in ans.citations]
    assert len(markers) == len(set(markers)), markers
    # one side unknown: refuse the comparison instead of answering half of it
    half = veritas.answer("Compare Contoso and Zorblax Corporation revenue in 2023")
    assert half.abstained and "Zorblax" in half.answer, half.answer


def test_api_time_travel_change_feed_and_per_request_mode():
    from fastapi.testclient import TestClient

    import api.main as api

    b, veritas, _bench = build_tiny_system()
    _load_sec(b, [
        ("Contoso Corporation", "0000000001", "net_income", 4834000000,
         "2007-09-30", "2008-09-27", "2009-10-27"),
        ("Contoso Corporation", "0000000001", "net_income", 6119000000,
         "2007-09-30", "2008-09-27", "2010-01-25"),
    ])
    api.STATE.update({"ready": True, "error": None, "builder": b, "veritas": veritas})
    try:
        c = TestClient(api.app)   # no `with`: startup (the real boot) does not run
        cfg = c.get("/config.js")
        assert cfg.status_code == 200 and "window.VERITAS_API" in cfg.text, cfg.text
        assert c.get("/health").json()["bus"]["kind"] in ("starting", "none"), c.get("/health").json()
        path = "/entity/Contoso/as_of"
        then = c.get(path, params={"attribute": "net_income", "valid": "2008-06-01",
                                   "known": "2009-12-01"}).json()
        assert then["value"] == "4.83 billion USD", then
        now = c.get(path, params={"attribute": "net_income", "valid": "2008-06-01"}).json()
        assert now["value"] == "6.12 billion USD" and len(now["belief_history"]) == 2, now
        assert "revised" in now["note"], now
        assert c.get(path, params={"attribute": "net_income", "valid": "someday"}).status_code == 422
        assert c.get("/entity/Nobody/as_of", params={"attribute": "revenue"}).status_code == 404

        feed = c.get("/changes", params={"kind": "CORRECTED"}).json()
        assert any(x["entity"] == "Contoso Corporation" and x["new_value"] == "6.12 billion USD"
                   for x in feed["changes"]), feed
        tl = c.get("/entity/Contoso/timeline", params={"attribute": "net_income"}).json()
        assert any(v["superseded_at"] for v in tl["attributes"]["net_income"]["versions"]), tl

        # the request's mode applies to that request only
        veritas.cfg.synthesis_mode = "generative"
        a = c.post("/ask", json={"question": "What was Contoso net income in 2008?",
                                 "mode": "extractive"}).json()
        assert "6.12 billion USD" in a["answer"], a["answer"]
        assert "originally reported as 4.83 billion USD" in a["answer"], a["answer"]
        assert veritas.cfg.synthesis_mode == "generative"
    finally:
        api.STATE.clear()
        api.STATE.update({"ready": False, "error": None})


def test_bus_consumer_thread_feeds_the_answering_store():
    """The deployed topology: a consumer thread ingests from the bus into the
    SAME store that answers questions, and one bad message cannot kill it."""
    import threading
    import time

    from veritas.ingest.streaming import (TOPIC_CHANGED, TOPIC_CHANGES, TOPIC_RAW,
                                          LocalBus, Message, StreamingIngest)

    b, veritas, _bench = build_tiny_system()
    bus, lock = LocalBus(), threading.RLock()
    # inline, consuming veritas.raw only: exactly how the API runs it
    stream = StreamingIngest(b.ingest, bus, lock=lock, inline=True)
    thread, stop = stream.start_background((TOPIC_RAW,))
    try:
        bus.produce(Message(TOPIC_RAW, key="junk", payload={}))   # malformed
        StreamingIngest(None, bus).publish_raw(
            source_id="sec.gov", entity="Quillon Robotics", tier=1, published="2026-09-01",
            doc_id="quillon-8k",
            text="Quillon Robotics filing: Ada Okafor was appointed chief executive "
                 "officer of Quillon Robotics, effective September 2026.")
        cur, deadline = None, time.time() + 60
        while time.time() < deadline:
            with lock:
                cur = b.store.current("Quillon Robotics", "ceo")
            if cur is not None:
                break
            time.sleep(0.1)
        assert cur is not None and "Ada Okafor" in cur.value, cur
        assert stream.stats["errors"] >= 1, stream.stats
        assert thread.is_alive(), "a malformed message killed the consumer thread"
        with lock:
            ans = veritas.answer("Who is the current CEO of Quillon Robotics?")
        assert "Ada Okafor" in ans.answer, ans.answer
    finally:
        stop.set()
        thread.join(timeout=5)


def test_kafka_security_config_reads_managed_broker_settings():
    import os

    from veritas.ingest.streaming import kafka_security_config

    keys = ("VERITAS_KAFKA_SECURITY_PROTOCOL", "VERITAS_KAFKA_SASL_MECHANISM",
            "VERITAS_KAFKA_USERNAME", "VERITAS_KAFKA_PASSWORD")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert kafka_security_config() == {}          # local PLAINTEXT broker
        os.environ.update(VERITAS_KAFKA_SECURITY_PROTOCOL="sasl_ssl",
                          VERITAS_KAFKA_SASL_MECHANISM="scram-sha-256",
                          VERITAS_KAFKA_USERNAME="u", VERITAS_KAFKA_PASSWORD="p")
        assert kafka_security_config() == {
            "security_protocol": "SASL_SSL", "sasl_mechanism": "SCRAM-SHA-256",
            "sasl_plain_username": "u", "sasl_plain_password": "p"}
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


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
