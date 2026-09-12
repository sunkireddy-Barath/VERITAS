"""Cross-encoder reranker, from scratch.

Bi-encoder vs cross-encoder
---------------------------
Retrieval embeds query and document *independently*:

    score = enc(q) . enc(d)

That independence is what makes it fast -- documents are embedded once,
offline. It is also its ceiling: the document vector is computed without ever
seeing the query, so the model must compress "everything anyone might ask" into
one vector. Fine-grained relations (negation, which of two dates is being
asked about, which entity a pronoun binds to) do not survive that compression.

A cross-encoder concatenates them:

    score = W . trunk([CLS] q <|evidence|> d)

Now every query token attends to every document token, so the model can check
alignment directly. Cost: O(|candidates|) forward passes per query, and nothing
can be precomputed. That is why it is only affordable on the top ~50 candidates
-- which is exactly the two-stage retrieve-then-rerank design.

Why this matters more for VERITAS than for ordinary RAG: the verifier asks
"does this passage *support this claim*", which is an entailment question, not a
similarity question. Bi-encoders famously score a passage stating the opposite
("X did NOT step down") as highly similar to the claim. The cross-encoder is
the first component that can tell those apart, and the same architecture is
reused as the NLI head in evidence/verifier.py.

Training: pairwise ranking loss on (query, positive, negative) triples

    L = -log sigmoid( s(q, d+) - s(q, d-) )

Optimising the *margin between* a good and a bad passage is the right objective
for ranking. Pointwise regression to a relevance label would waste capacity
calibrating absolute scores that only ever get sorted. Hard negatives matter:
mine them from the top BM25/dense hits that are not the gold passage, because
random negatives are trivially separable and produce no gradient.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEncoder(nn.Module):
    """Scores (query, passage) jointly using the shared LM trunk."""

    def __init__(self, model, tokenizer, max_len: int = 384, device: str = "cpu") -> None:
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self.max_len = max_len
        self.device = device
        self.head = nn.Sequential(
            nn.Linear(model.cfg.d_model, model.cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(model.cfg.d_model // 2, 1),
        ).to(device)
        self.sep = tokenizer.special_tokens["<|evidence|>"]
        self.pad = tokenizer.special_tokens["<|pad|>"]

    def _encode_pair(self, queries: Sequence[str], passages: Sequence[str]):
        seqs = []
        for q, p in zip(queries, passages):
            qi = self.tok.encode(q)[: self.max_len // 3]
            pi = self.tok.encode(p)[: self.max_len - len(qi) - 1]
            seqs.append(qi + [self.sep] + pi)
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.float32)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1.0
        return ids.to(self.device), mask.to(self.device)

    def forward(self, queries: Sequence[str], passages: Sequence[str]) -> torch.Tensor:
        ids, mask = self._encode_pair(queries, passages)
        h = self.model.hidden_states(ids)
        pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled).squeeze(-1)

    @torch.inference_mode()
    def rerank(
        self, query: str, passages: Sequence[str], ids: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None, batch_size: int = 16,
    ) -> List[Tuple[str, float]]:
        if not passages:
            return []
        self.eval()
        scores: List[float] = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            s = self(([query] * len(batch)), batch)
            scores.extend(s.float().cpu().tolist())
        keys = list(ids) if ids is not None else [str(i) for i in range(len(passages))]
        out = sorted(zip(keys, scores), key=lambda kv: -kv[1])
        return out[:top_k] if top_k else out


def pairwise_rank_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """-log sigmoid(s+ - s-): unbounded margin, saturating gradient."""
    return F.softplus(-(pos - neg)).mean()


def train_reranker(
    ce: CrossEncoder,
    triples: Sequence[Tuple[str, str, str]],  # (query, positive, hard negative)
    steps: int = 300,
    batch_size: int = 8,
    lr: float = 1e-4,
    log_every: int = 50,
) -> List[float]:
    params = list(ce.head.parameters()) + list(ce.model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    ce.train()
    hist: List[float] = []
    for step in range(steps):
        idx = np.random.choice(len(triples), size=min(batch_size, len(triples)), replace=False)
        qs = [triples[i][0] for i in idx]
        ps = [triples[i][1] for i in idx]
        ns = [triples[i][2] for i in idx]
        loss = pairwise_rank_loss(ce(qs, ps), ce(qs, ns))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(loss.item())
        if step % log_every == 0:
            print(f"[rerank] step {step:4d} | pairwise loss {loss.item():.4f}")
    ce.eval()
    return hist


def mine_hard_negatives(
    retriever, query: str, gold_ids: Sequence[str], n: int = 4, candidate_k: int = 30
) -> List[str]:
    """Top-ranked non-gold candidates: the passages the model currently
    confuses with the answer, which is where the gradient signal is."""
    gold = set(gold_ids)
    out = []
    for c in retriever.retrieve(query, k=candidate_k, candidate_k=candidate_k):
        if c.doc_id not in gold:
            out.append(c.doc_id)
        if len(out) >= n:
            break
    return out
