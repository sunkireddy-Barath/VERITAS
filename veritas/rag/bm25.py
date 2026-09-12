"""BM25 sparse retrieval, from scratch.

The scoring function
--------------------
For query Q and document D:

    score(D, Q) = sum_{t in Q} IDF(t) * ( f(t,D) * (k1 + 1) ) /
                               ( f(t,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(t) = ln( 1 + (N - n_t + 0.5) / (n_t + 0.5) )

Reading the three parts:

* **IDF** -- a term in few documents is discriminative. The +0.5 smoothing and
  the outer 1 + ... keep the value positive even for a term appearing in most
  documents (plain Robertson-Sparck-Jones IDF goes negative there, which lets a
  stopword *subtract* score).
* **Term-frequency saturation** -- f/(f + k1) is concave: the 1st occurrence of
  "revenue" is strong evidence, the 20th adds almost nothing. Raw TF would let
  one keyword-stuffed page dominate. k1 ~ 1.2-1.5 sets how fast it saturates.
* **Length normalisation** -- b*|D|/avgdl penalises long documents, which
  otherwise accumulate matches by sheer size. b=0.75 is the standard
  compromise between b=0 (no normalisation) and b=1 (full).

Why keep BM25 at all when we have embeddings? Because the two fail differently.
Dense retrieval fails on exact strings it never saw: a ticker symbol, a docket
number, a person's surname, a version string. Those are *precisely* the tokens
that identify entities in an evidence system. BM25 has no vocabulary problem
and needs no training. Hybrid retrieval exists to cover both failure modes.

Implementation
--------------
Inverted index term -> (doc_ids[], term_freqs[]) as numpy arrays. Scoring
touches only the postings of query terms and accumulates into a dense score
buffer with `np.add.at`. Cost is O(sum of postings-list lengths for query
terms), not O(N_docs) -- typically 10^3-10^4 of 10^6 docs.
"""
from __future__ import annotations

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.'][a-z0-9]+)*")

# Kept deliberately tiny. Aggressive stopword lists destroy phrase queries like
# "who is the CEO of X" and negations ("not approved"), both of which matter for
# claim-level evidence.
STOPWORDS = frozenset(
    "a an the of and or to in on at for is are was were be been by with as that this it its from".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_ids: List[str] = []
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avgdl: float = 0.0
        self.postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.idf: Dict[str, float] = {}
        self._building: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._lens: List[int] = []

    # ------------------------------------------------------------- indexing
    def add(self, doc_id: str, text: str) -> None:
        idx = len(self.doc_ids)
        self.doc_ids.append(doc_id)
        tf: Dict[str, int] = defaultdict(int)
        n = 0
        for t in tokenize(text):
            tf[t] += 1
            n += 1
        self._lens.append(n)
        for term, f in tf.items():
            self._building[term].append((idx, f))

    def add_many(self, docs: Iterable[Tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def finalize(self) -> "BM25":
        """Merge pending documents into the frozen arrays and recompute IDF.

        Incremental by design: `finalize()` after a later `add()` appends the
        new postings to the existing arrays instead of rebuilding the index.
        That is what lets the continuous ingest pipeline add one document
        without an O(corpus) re-index. IDF *is* recomputed for every term,
        because N changed -- but that is O(vocabulary), not O(corpus), and
        skipping it would leave every term's IDF quietly wrong.
        """
        N = len(self.doc_ids)
        self.doc_len = np.asarray(self._lens, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if N else 0.0

        for term, plist in self._building.items():
            ids = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            fs = np.fromiter((f for _, f in plist), dtype=np.float32, count=len(plist))
            if term in self.postings:
                old_ids, old_fs = self.postings[term]
                ids = np.concatenate([old_ids, ids])
                fs = np.concatenate([old_fs, fs])
            self.postings[term] = (ids, fs)
        self._building = defaultdict(list)

        for term, (ids, _fs) in self.postings.items():
            n_t = len(ids)
            self.idf[term] = float(np.log(1.0 + (N - n_t + 0.5) / (n_t + 0.5)))
        # denominator term that does not depend on the query -> precompute once
        self._len_norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        return self

    # -------------------------------------------------------------- scoring
    def score_all(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.doc_ids), dtype=np.float32)
        qterms = tokenize(query)
        if not qterms:
            return scores
        # repeated query terms are summed once via their multiplicity
        qtf: Dict[str, int] = defaultdict(int)
        for t in qterms:
            qtf[t] += 1
        for term, qf in qtf.items():
            post = self.postings.get(term)
            if post is None:
                continue
            ids, fs = post
            contrib = self.idf[term] * qf * (fs * (self.k1 + 1.0)) / (fs + self._len_norm[ids])
            np.add.at(scores, ids, contrib)
        return scores

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        scores = self.score_all(query)
        if not scores.size:
            return []
        k = min(k, scores.size)
        # argpartition is O(N); a full argsort would be O(N log N)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]

    # ---------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"k1": self.k1, "b": self.b, "doc_ids": self.doc_ids,
                 "doc_len": self.doc_len, "avgdl": self.avgdl,
                 "postings": self.postings, "idf": self.idf}, f, protocol=4)

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        with open(path, "rb") as f:
            d = pickle.load(f)
        o = cls(d["k1"], d["b"])
        o.doc_ids, o.doc_len, o.avgdl = d["doc_ids"], d["doc_len"], d["avgdl"]
        o.postings, o.idf = d["postings"], d["idf"]
        o._len_norm = o.k1 * (1 - o.b + o.b * o.doc_len / max(o.avgdl, 1e-9))
        return o
