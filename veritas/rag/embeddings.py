"""Dense retrieval: an embedding encoder and a vector index, both from scratch.

Encoder
-------
The retrieval encoder reuses the pretrained VeritasLM trunk instead of a second
model. Pretraining already taught it the language; a separate encoder would
double training cost and VRAM for no benefit at this scale.

Two changes turn a causal LM into an embedder:

* **Mean pooling over tokens, not the last hidden state.** In a *causal* model
  only the last position has seen the whole sequence, so last-token pooling is
  defensible -- but it is also a single position carrying the entire meaning,
  and it is dominated by whatever token happens to end the chunk. Mean pooling
  averages evidence from every token and is consistently more robust for
  retrieval at small scale. (Masked pooling: padding must be excluded or short
  chunks get their vectors dragged toward the pad embedding.)
* **L2 normalisation.** After normalising, the inner product equals cosine
  similarity, and ||a-b||^2 = 2 - 2 a.b -- so maximum inner product, cosine and
  Euclidean nearest-neighbour all give the *same* ranking. One normalisation
  makes the index metric-agnostic and keeps scores in [-1, 1], which is what
  makes fusion with BM25 tractable later.

Training objective: InfoNCE with in-batch negatives

    L = -log  exp(s(q, d+) / tau) / sum_j exp(s(q, d_j) / tau)

Every other document in the batch is a negative, so a batch of B gives B-1
negatives per query for free -- the reason contrastive retrieval training wants
large batches. tau (~0.05) sharpens the softmax; too high and the gradient is
flat, too low and it fixates on the single hardest negative.

Index
-----
* Exact: one normalised (N, d) float32 matrix; search is a single GEMM. At
  N <= ~10^5 this is the *fastest* option and it is exact -- no ANN needed.
* IVF: k-means into sqrt(N) cells, probe the nprobe nearest cells. Cuts the
  scan to ~nprobe/ncells of the corpus, trading a small recall loss.
* int8 scalar quantisation: 4x memory reduction with ~1% recall loss, because
  normalised embedding components are tightly bounded in [-1, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ encoder
class Embedder:
    """Wraps a VeritasLM trunk as a sentence encoder."""

    def __init__(self, model, tokenizer, max_len: int = 256, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.max_len = max_len
        self.device = device
        self.dim = model.cfg.d_model

    def _pad(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        pad = self.tok.special_tokens["<|pad|>"]
        seqs = [self.tok.encode(t)[: self.max_len] or [pad] for t in texts]
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.float32)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1.0
        return ids.to(self.device), mask.to(self.device)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            ids, mask = self._pad(texts[i : i + batch_size])
            h = self.model.hidden_states(ids)                     # (B, L, d)
            pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            out.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.dim), np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def info_nce_loss(q: torch.Tensor, d: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """q, d: (B, dim) L2-normalised. Positives are on the diagonal."""
    logits = (q @ d.T) / temperature
    labels = torch.arange(q.size(0), device=q.device)
    # symmetric: query->doc and doc->query, which stabilises the embedding space
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def train_embedder(
    model, tokenizer, pairs: Sequence[Tuple[str, str]], steps: int = 300,
    batch_size: int = 16, lr: float = 1e-4, temperature: float = 0.05,
    max_len: int = 256, device: str = "cpu", log_every: int = 50,
):
    """Contrastive fine-tune of the trunk on (query, positive_passage) pairs."""
    emb = Embedder(model, tokenizer, max_len, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    for step in range(steps):
        idx = np.random.choice(len(pairs), size=min(batch_size, len(pairs)), replace=False)
        qs = [pairs[i][0] for i in idx]
        ds = [pairs[i][1] for i in idx]
        qi, qm = emb._pad(qs)
        di, dm = emb._pad(ds)
        hq = model.hidden_states(qi)
        hd = model.hidden_states(di)
        vq = F.normalize((hq * qm[..., None]).sum(1) / qm.sum(1, keepdim=True).clamp(min=1), dim=-1)
        vd = F.normalize((hd * dm[..., None]).sum(1) / dm.sum(1, keepdim=True).clamp(min=1), dim=-1)
        loss = info_nce_loss(vq, vd, temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            print(f"[embed] step {step:4d} | infoNCE {loss.item():.4f}")
        hist.append(loss.item())
    model.eval()
    return hist


# -------------------------------------------------------------------- index
@dataclass
class SearchHit:
    doc_id: str
    score: float
    rank: int


class VectorIndex:
    """Exact / IVF vector index over L2-normalised embeddings."""

    def __init__(self, dim: int, quantize: bool = False) -> None:
        self.dim = dim
        self.quantize = quantize
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self._q: Optional[np.ndarray] = None
        self._scale: float = 1.0
        self.centroids: Optional[np.ndarray] = None
        self.cells: Optional[List[np.ndarray]] = None

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        self.ids.extend(ids)
        if self.quantize:
            self._scale = 127.0
            self._q = np.clip(np.round(self.vectors * self._scale), -127, 127).astype(np.int8)

    # ------------------------------------------------------------- IVF build
    def build_ivf(self, n_cells: Optional[int] = None, iters: int = 12, seed: int = 0) -> "VectorIndex":
        """k-means (Lloyd) with k-means++ seeding on the unit sphere.

        On normalised vectors, Euclidean k-means is spherical k-means: minimising
        ||x - c||^2 is maximising x.c. So assignment is one GEMM + argmax.
        """
        n = len(self.ids)
        if n < 64:
            return self  # exact search is already faster than probing
        k = n_cells or max(1, int(np.sqrt(n)))
        rng = np.random.default_rng(seed)
        X = self.vectors
        # k-means++ seeding: spread initial centroids by D^2 sampling
        centroids = [X[rng.integers(n)]]
        d2 = np.full(n, np.inf, dtype=np.float32)
        for _ in range(k - 1):
            d2 = np.minimum(d2, 2 - 2 * (X @ centroids[-1]))
            p = np.clip(d2, 0, None)
            s = p.sum()
            centroids.append(X[rng.integers(n) if s <= 0 else rng.choice(n, p=p / s)])
        C = np.asarray(centroids, dtype=np.float32)
        for _ in range(iters):
            assign = (X @ C.T).argmax(1)
            for j in range(k):
                m = assign == j
                if m.any():
                    c = X[m].mean(0)
                    C[j] = c / max(np.linalg.norm(c), 1e-12)
        assign = (X @ C.T).argmax(1)
        self.centroids = C
        self.cells = [np.where(assign == j)[0] for j in range(k)]
        return self

    # ---------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int = 20, nprobe: int = 8) -> List[SearchHit]:
        if self.vectors is None or not self.ids:
            return []
        q = np.asarray(query, dtype=np.float32).ravel()
        q /= max(np.linalg.norm(q), 1e-12)

        if self.centroids is not None and self.cells is not None:
            probe = np.argsort(-(self.centroids @ q))[:nprobe]
            cand = np.concatenate([self.cells[j] for j in probe]) if len(probe) else np.arange(len(self.ids))
            sims = self.vectors[cand] @ q
            idx = cand
        elif self._q is not None:
            sims = (self._q @ np.round(q * self._scale).astype(np.int32)) / (self._scale ** 2)
            idx = np.arange(len(self.ids))
        else:
            sims = self.vectors @ q
            idx = np.arange(len(self.ids))

        k = min(k, sims.size)
        if k == 0:
            return []
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [SearchHit(self.ids[int(idx[t])], float(sims[t]), r) for r, t in enumerate(top)]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(self.ids, dtype=object),
                            vectors=self.vectors, dim=self.dim)

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        z = np.load(path, allow_pickle=True)
        idx = cls(int(z["dim"]))
        idx.ids = list(z["ids"])
        idx.vectors = z["vectors"]
        return idx
