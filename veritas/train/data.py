"""Pretraining data pipeline: clean -> dedup -> tokenize -> memmap shards.

Design decisions and why
------------------------
* **uint16 memmap, not a list of tensors.** With vocab <= 65535 each token is
  2 bytes, so a 200M-token corpus is a 400MB file. `np.memmap` lets the OS page
  it in on demand: RAM usage is O(batch), start-up is instant, and the same
  file is shared across dataloader workers with zero copies.
* **Random-offset sampling, not a shuffled index list.** A shuffled list of
  every window costs O(N) memory and destroys locality. Drawing a uniform
  random start offset per sample is O(1), unbiased, and page-cache friendly.
* **Near-duplicate removal with MinHash + LSH.** Web corpora are 20-50%
  duplicated; duplicates waste compute and cause memorisation. Exact hashing
  misses "same article, different boilerplate". MinHash estimates Jaccard
  similarity in O(k) per document, and LSH banding turns the O(N^2) all-pairs
  comparison into O(N) bucket lookups.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------------ cleaning
_WS = re.compile(r"[ \t\x0b\f\r]+")
_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """NFKC normalise, strip control chars, collapse whitespace.

    NFKC matters: the same visible string can have several byte encodings
    (full-width digits, ligatures, non-breaking spaces). Without normalisation
    the tokenizer learns separate merges for each variant and entity matching
    in the evidence layer silently fails.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub("", text)
    text = _WS.sub(" ", text)
    text = _NL.sub("\n\n", text)
    return text.strip()


def quality_filter(
    text: str, min_chars: int = 200, max_symbol_ratio: float = 0.3, min_mean_word_len: float = 2.5
) -> bool:
    """Cheap heuristics from the C4/Gopher recipes. Each rejects a known
    failure mode: too short (no learnable structure), symbol-heavy (nav bars,
    code dumps, base64), degenerate words (OCR noise)."""
    if len(text) < min_chars:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    if alpha / len(text) < 1 - max_symbol_ratio:
        return False
    words = text.split()
    if not words:
        return False
    if sum(len(w) for w in words) / len(words) < min_mean_word_len:
        return False
    if len(set(words)) / len(words) < 0.25:  # repetitive spam
        return False
    return True


# --------------------------------------------------------- deduplication
def _shingles(text: str, k: int = 5) -> List[int]:
    words = text.lower().split()
    if len(words) < k:
        return [hash(" ".join(words))]
    return [hash(" ".join(words[i : i + k])) for i in range(len(words) - k + 1)]


class MinHashDeduper:
    """MinHash + LSH banding near-duplicate detector.

    MinHash: for a random permutation h, P(min h(A) == min h(B)) = J(A, B).
    Averaging over `num_perm` permutations estimates Jaccard with standard
    error ~1/sqrt(num_perm). We use the standard (a*x + b) mod prime family.

    LSH: split the signature into b bands of r rows. Two docs collide in a band
    with probability J^r, so P(collide at all) = 1 - (1 - J^r)^b -- an S-curve
    with its threshold near (1/b)^(1/r). Choosing b=16, r=8 puts the threshold
    around 0.8 Jaccard.
    """

    _MERSENNE = (1 << 61) - 1

    def __init__(self, num_perm: int = 128, bands: int = 16, seed: int = 0) -> None:
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm, self.bands = num_perm, bands
        self.rows = num_perm // bands
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, self._MERSENNE, num_perm, dtype=np.uint64)
        self.b = rng.integers(0, self._MERSENNE, num_perm, dtype=np.uint64)
        self.buckets: Dict[Tuple[int, bytes], List[int]] = defaultdict(list)
        self.exact: set = set()

    def signature(self, text: str) -> np.ndarray:
        sh = np.array([h & 0xFFFFFFFFFFFFFFF for h in _shingles(text)], dtype=np.uint64)
        if sh.size == 0:
            return np.zeros(self.num_perm, dtype=np.uint64)
        # (num_perm, n_shingles) -> min over shingles
        hashed = (self.a[:, None] * sh[None, :] + self.b[:, None]) % self._MERSENNE
        return hashed.min(axis=1)

    def add(self, doc_id: int, text: str) -> bool:
        """Return True if the doc is new, False if it duplicates a seen doc."""
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        if digest in self.exact:
            return False
        sig = self.signature(text)
        keys = [
            (i, sig[i * self.rows : (i + 1) * self.rows].tobytes()) for i in range(self.bands)
        ]
        if any(self.buckets.get(k) for k in keys):
            return False
        self.exact.add(digest)
        for k in keys:
            self.buckets[k].append(doc_id)
        return True


def dedupe(docs: Iterable[str], num_perm: int = 128, bands: int = 16) -> Iterator[str]:
    d = MinHashDeduper(num_perm, bands)
    for i, doc in enumerate(docs):
        if d.add(i, doc):
            yield doc


# ------------------------------------------------------------ tokenize/shard
def build_shard(
    docs: Iterable[str],
    tokenizer,
    out_path: str | Path,
    eos_token: str = "<|eos|>",
    dtype=np.uint16,
) -> int:
    """Tokenize documents into one flat uint16 .bin file, EOS-separated.

    Documents are concatenated into a single stream rather than padded to a
    fixed length: padding a 512-token window to fit a 40-token document wastes
    >90% of the FLOPs. The EOS token teaches the model where a document ends,
    so cross-document attention within a window is harmless.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.special_tokens[eos_token]
    buf: List[int] = []
    total = 0
    with open(out_path, "wb") as f:
        for doc in docs:
            ids = tokenizer.encode(doc)
            ids.append(eos)
            buf.extend(ids)
            if len(buf) >= 1 << 20:  # flush every ~1M tokens to bound RAM
                np.asarray(buf, dtype=dtype).tofile(f)
                total += len(buf)
                buf.clear()
        if buf:
            np.asarray(buf, dtype=dtype).tofile(f)
            total += len(buf)
    return total


class TokenDataset:
    """Memory-mapped token stream with O(1) random-window sampling."""

    def __init__(self, path: str | Path, seq_len: int, dtype=np.uint16) -> None:
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{path}: {len(self.data)} tokens < seq_len+1")

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def batch(self, batch_size: int, device: str = "cpu", generator=None):
        """Sample `batch_size` random windows. Returns (x, y) with y = x shifted
        by one -- next-token prediction targets."""
        import torch

        ix = torch.randint(len(self), (batch_size,), generator=generator)
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.seq_len].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy(self.data[i + 1 : i + 1 + self.seq_len].astype(np.int64)) for i in ix]
        )
        if device.startswith("cuda"):
            # pin + non_blocking overlaps the H2D copy with the previous step
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        return x, y


def split_shard(src: str | Path, train_out: str | Path, val_out: str | Path, val_frac: float = 0.01):
    """Contiguous tail split (not random): a random split would put windows
    overlapping the same documents in both sets, leaking train into val and
    making validation perplexity optimistic."""
    if Path(src).stat().st_size == 0:
        raise ValueError(
            f"{src} is empty: tokenization produced 0 tokens. Usually the corpus was "
            f"entirely removed by quality_filter or dedupe -- check the survival rates.")
    data = np.memmap(src, dtype=np.uint16, mode="r")
    n_val = max(1024, int(len(data) * val_frac))
    np.asarray(data[:-n_val]).tofile(train_out)
    np.asarray(data[-n_val:]).tofile(val_out)
    return len(data) - n_val, n_val
