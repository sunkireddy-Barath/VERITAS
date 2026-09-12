"""Byte-level BPE tokenizer, implemented from scratch.

Algorithm
---------
Byte-Pair Encoding (Sennrich et al., 2016) applied over *bytes* rather than
unicode characters, so the vocabulary is closed: any byte string is encodable
and ``decode(encode(x)) == x`` exactly -- no <unk>, no unicode holes. That
property matters for VERITAS because evidence text is scraped from the open
web: filings, tables and PDFs contain currency symbols, CJK, emoji and broken
mojibake, and a tokenizer that silently drops bytes would silently corrupt the
evidence we later cite.

Why BPE and not the alternatives
--------------------------------
* word-level      -> unbounded vocab, <unk> on every new entity name. Fatal
                     here: entity names *are* the payload.
* char/byte-level -> no <unk>, but sequences are ~4x longer; attention is
                     O(L^2), so this is the most expensive possible choice.
* WordPiece       -> likelihood-based merges, needs an LM scoring pass; more
                     compute for ~the same compression as BPE.
* Unigram (SPM)   -> better theory (EM over a token lattice), but training is
                     iterative EM and encoding needs Viterbi at inference.
* BPE             -> greedy frequency merges. Training is near-linear with the
                     index below, encoding is a deterministic merge replay,
                     compression is within a few % of Unigram. Best
                     accuracy/compute ratio, which is why GPT-2/3/4 use it.

Training complexity
-------------------
The textbook loop is O(n_merges x corpus). This implementation restructures it:

  1. Pre-tokenize the corpus once with a regex and collapse it to a
     ``{word -> frequency}`` dict. Natural language is Zipfian, so ~10^8
     characters collapse to ~10^5-10^6 distinct words. Every later step is
     paid per *distinct word*, weighted by its frequency.
  2. Maintain ``pair_counts[pair]`` and an inverted index
     ``pair_to_words[pair] -> {word_index}``.
  3. Each merge rewrites ONLY the words in ``pair_to_words[best]`` and applies
     local +/- deltas to ``pair_counts``, pushing updates into a lazy-deletion
     max-heap.

Cost: O(N_pretok) once, then O(sum |affected words|) per merge instead of
O(corpus) per merge. Memory is O(distinct words), not O(corpus).

Encoding replays merges in learned rank order using a doubly linked list plus
a heap: O(L log L) per pre-token, with an LRU cache at the word level so the
Zipfian head of the distribution is effectively free.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:  # `regex` supports \p{L}; fall back to stdlib `re` with an ASCII pattern
    import regex as _re

    SPLIT_PATTERN = (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
    )
except ImportError:  # pragma: no cover - environment dependent
    import re as _re

    # stdlib `re` has no \p{L}, so unicode letters are matched with [^\W\d_]
    # (word chars minus digits and underscore), which IS unicode-aware.
    #
    # The trailing `|\S` catch-all is load-bearing: without it, any character
    # matched by no earlier alternative is silently DROPPED by findall, and the
    # tokenizer stops round-tripping. That is a data-loss bug, not a quality
    # issue -- it would corrupt exactly the CJK/symbol content that appears in
    # scraped evidence. `assert_lossless()` below guards it.
    SPLIT_PATTERN = (
        r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\w]?[^\W\d_]+|\d{1,3}"
        r"| ?[^\s\w]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+|\S"
    )


def assert_lossless(text: str) -> None:
    """Pre-tokenization must partition the input, losing nothing.

    `re.findall` returns only what the pattern matches; anything unmatched
    vanishes. Concatenating the pieces has to reproduce the input exactly.
    """
    pieces = _re.compile(SPLIT_PATTERN).findall(text)
    joined = "".join(pieces)
    if joined != text:
        raise AssertionError(
            f"pre-tokenizer dropped {len(text) - len(joined)} characters; "
            f"SPLIT_PATTERN is missing a catch-all alternative"
        )

Pair = Tuple[int, int]

#: Special tokens. The last four are VERITAS-specific control tokens: the model
#: is trained to emit them, which makes evidence spans, claim spans, temporal
#: qualifiers and abstention *parseable* rather than regex-guessed from prose.
SPECIAL_TOKENS: Tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|evidence|>",  # opens a retrieved evidence block
    "<|claim|>",     # opens an atomic claim being asserted/verified
    "<|time|>",      # opens a temporal qualifier (valid_from / valid_to)
    "<|unknown|>",   # explicit abstention: "insufficient evidence"
)


@dataclass
class BPETokenizer:
    """A trained byte-level BPE tokenizer."""

    merges: Dict[Pair, int] = field(default_factory=dict)  # pair -> rank
    vocab: Dict[int, bytes] = field(default_factory=dict)  # id   -> bytes
    special_tokens: Dict[str, int] = field(default_factory=dict)  # str -> id

    # ---------------------------------------------------------------- build
    def __post_init__(self) -> None:
        if not self.vocab:
            self.vocab = {i: bytes([i]) for i in range(256)}
        self._special_re = None
        self._rebuild()

    def _rebuild(self) -> None:
        """Derive lookup tables. Merged id for rank r is exactly 256 + r,
        because ids are handed out in merge order during training."""
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        self._pat = _re.compile(SPLIT_PATTERN)
        if self.special_tokens:
            alts = "|".join(
                _re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)
            )
            self._special_re = _re.compile("(" + alts + ")")
        self._encode_word_cached = lru_cache(maxsize=262_144)(self._encode_word)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # ------------------------------------------------------------- training
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8192,
        specials: Sequence[str] = SPECIAL_TOKENS,
        min_frequency: int = 1,
        verbose: bool = True,
    ) -> "BPETokenizer":
        n_merges = vocab_size - 256 - len(specials)
        if n_merges <= 0:
            raise ValueError("vocab_size must exceed 256 + len(specials)")

        # --- step 1: corpus -> {word -> frequency} --------------------------
        freqs: Counter = Counter()
        pat = _re.compile(SPLIT_PATTERN)
        for text in texts:
            freqs.update(pat.findall(text))
        if verbose:
            print(f"[bpe] {len(freqs):,} distinct pre-tokens")

        words: List[List[int]] = []
        counts: List[int] = []
        for word, c in freqs.items():
            if c < min_frequency:
                continue
            words.append(list(word.encode("utf-8")))
            counts.append(c)
        del freqs

        # --- step 2: initial pair statistics + inverted index ---------------
        pair_counts: Counter = Counter()
        pair_to_words: Dict[Pair, Set[int]] = {}
        for wi, (w, c) in enumerate(zip(words, counts)):
            for a, b in zip(w, w[1:]):
                pair_counts[(a, b)] += c
                pair_to_words.setdefault((a, b), set()).add(wi)

        # lazy-deletion max-heap: stale entries are validated on pop
        heap: List[Tuple[int, Pair]] = [(-c, p) for p, c in pair_counts.items()]
        heapify(heap)

        merges: Dict[Pair, int] = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        next_id = 256

        # --- step 3: merge loop with local delta updates ---------------------
        for rank in range(n_merges):
            best: Optional[Pair] = None
            best_count = 0
            while heap:
                neg_c, pair = heappop(heap)
                cur = pair_counts.get(pair, 0)
                if cur == -neg_c and cur > 0:
                    best, best_count = pair, cur
                    break
            if best is None:
                if verbose:
                    print(f"[bpe] corpus exhausted after {rank} merges")
                break

            new_id = next_id
            next_id += 1
            merges[best] = rank
            vocab[new_id] = vocab[best[0]] + vocab[best[1]]

            touched = pair_to_words.pop(best, set())
            dirty: Counter = Counter()
            for wi in touched:
                w = words[wi]
                c = counts[wi]
                out: List[int] = []
                i, n = 0, len(w)
                while i < n:
                    if i < n - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                        if out:  # left neighbour pair changes
                            dirty[(out[-1], best[0])] -= c
                            dirty[(out[-1], new_id)] += c
                        if i + 2 < n:  # right neighbour pair changes
                            dirty[(best[1], w[i + 2])] -= c
                            dirty[(new_id, w[i + 2])] += c
                        dirty[best] -= c
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = out
                for a, b in zip(out, out[1:]):
                    if a == new_id or b == new_id:
                        pair_to_words.setdefault((a, b), set()).add(wi)

            for pair, delta in dirty.items():
                if not delta:
                    continue
                new_c = pair_counts.get(pair, 0) + delta
                if new_c <= 0:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)
                else:
                    pair_counts[pair] = new_c
                    heappush(heap, (-new_c, pair))

            if verbose and (rank + 1) % 1000 == 0:
                print(f"[bpe] merge {rank + 1}/{n_merges} {vocab[new_id]!r} freq={best_count}")

        special_ids = {s: next_id + i for i, s in enumerate(specials)}
        return cls(merges=merges, vocab=vocab, special_tokens=special_ids)

    # ------------------------------------------------------------- encoding
    def _encode_word(self, word: bytes) -> Tuple[int, ...]:
        """Replay merges on one pre-token, lowest rank first.

        Linked list (prev/nxt/alive) + heap keyed by merge rank. Popping the
        globally-lowest rank reproduces exactly the order the merges were
        learned in, which is what makes encoding deterministic.
        """
        ids = list(word)
        n = len(ids)
        if n < 2:
            return tuple(ids)

        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        nxt[n - 1] = -1
        alive = [True] * n

        heap: List[Tuple[int, int]] = []
        for i in range(n - 1):
            r = self.merges.get((ids[i], ids[i + 1]))
            if r is not None:
                heap.append((r, i))
        heapify(heap)

        while heap:
            rank, i = heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1 or not alive[j]:
                continue
            if self.merges.get((ids[i], ids[j])) != rank:
                continue  # stale entry: one side was already merged away
            ids[i] = 256 + rank
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
                r = self.merges.get((ids[i], ids[k]))
                if r is not None:
                    heappush(heap, (r, i))
            p = prev[i]
            if p != -1:
                r = self.merges.get((ids[p], ids[i]))
                if r is not None:
                    heappush(heap, (r, p))

        out: List[int] = []
        i = 0
        while i != -1:
            out.append(ids[i])
            i = nxt[i]
        return tuple(out)

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """Encode text -> ids. Literal special-token strings are honoured."""
        if self._special_re is not None and "<|" in text:
            pieces = self._special_re.split(text)
        else:
            pieces = [text]
        out: List[int] = []
        if add_special:
            out.append(self.special_tokens["<|bos|>"])
        for piece in pieces:
            if not piece:
                continue
            if piece in self.special_tokens:
                out.append(self.special_tokens[piece])
                continue
            for word in self._pat.findall(piece):
                out.extend(self._encode_word_cached(word.encode("utf-8")))
        if add_special:
            out.append(self.special_tokens["<|eos|>"])
        return out

    def encode_batch(self, texts: Sequence[str], add_special: bool = False) -> List[List[int]]:
        return [self.encode(t, add_special) for t in texts]

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        parts: List[bytes] = []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                if not skip_special:
                    parts.append(self.id_to_special[i].encode("utf-8"))
            else:
                parts.append(self.vocab.get(i, b""))
        # errors="replace" only fires mid-stream (partial token during
        # streaming generation); a complete sequence always round-trips.
        return b"".join(parts).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "merges": [[a, b, r] for (a, b), r in self.merges.items()],
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "special_tokens": self.special_tokens,
        }
        p.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): r for a, b, r in payload["merges"]}
        vocab = {int(k): bytes.fromhex(v) for k, v in payload["vocab"].items()}
        return cls(merges=merges, vocab=vocab, special_tokens=payload["special_tokens"])

    # -------------------------------------------------------------- metrics
    def compression_ratio(self, text: str) -> float:
        """bytes per token -- the number to report when comparing vocab sizes."""
        ids = self.encode(text)
        return len(text.encode("utf-8")) / max(1, len(ids))
