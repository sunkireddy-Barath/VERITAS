"""Run the baseline ladder against TemporalEvidenceBench.

    python scripts/run_eval.py [--synthetic 3] [--checkpoint checkpoints/best.pt]

With no checkpoint the LM is randomly initialised: retrieval, temporal
reasoning, verification and abstention are all still exercised (they are not
generative), so the table is meaningful for every column except the B1 row,
which measures the weights and therefore needs a trained model to say anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from veritas.eval.baselines import (
    BasicRAG, HybridRAG, LLMOnly, TemporalRAG, VeritasSystem, compare,
)
from veritas.eval.benchmark import build_seed_benchmark, expand_synthetic
from veritas.model.transformer import ModelConfig, VeritasLM
from veritas.pipeline import VeritasSystemBuilder, load_benchmark_corpus
from veritas.tokenizer.bpe import BPETokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=int, default=0, help="synthetic items per template")
    ap.add_argument("--checkpoint", type=str, default="")
    ap.add_argument("--tokenizer", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # With an untrained checkpoint the embedder is random, so dense retrieval
    # varies run to run. Seeding makes the table reproducible; the numbers only
    # become meaningful for B1 once a real checkpoint is passed.
    torch.manual_seed(args.seed)

    bench = build_seed_benchmark()
    if args.synthetic:
        bench = expand_synthetic(bench, args.synthetic)
    print("benchmark:", bench.stats())

    texts = [d.text for i in bench.items for d in i.docs] + [i.question for i in bench.items]
    if args.tokenizer and Path(args.tokenizer).exists():
        tok = BPETokenizer.load(args.tokenizer)
    else:
        tok = BPETokenizer.train(texts * 4, vocab_size=1500, verbose=False)

    if args.checkpoint and Path(args.checkpoint).exists():
        model = VeritasLM.load(args.checkpoint)
    else:
        model = VeritasLM(ModelConfig(vocab_size=tok.vocab_size, d_model=128, n_layers=4,
                                      n_heads=4, n_kv_heads=2, max_seq_len=256))
    model.eval()

    b = VeritasSystemBuilder(model, tok, device="cpu")
    n_docs = load_benchmark_corpus(b, bench)
    print(f"ingested {n_docs} documents -> {len(b.corpus)} chunks")
    print("store:", b.ingest.summary())

    veritas = b.build()
    systems = [
        LLMOnly(model, tok),
        BasicRAG(b.vec, b.embedder, b.corpus, b.metadata),
        HybridRAG(b.retriever(), b.corpus, b.metadata),
        TemporalRAG(b.retriever(), b.corpus, b.metadata),
        VeritasSystem(veritas),
    ]
    print("\n" + compare(systems, bench))


if __name__ == "__main__":
    main()
