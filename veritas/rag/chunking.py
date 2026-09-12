"""Chunking: turn documents into retrievable, citable units.

Chunking is the most under-rated component of a RAG system. It decides what a
citation can even *point at*. VERITAS cites claim -> chunk -> document span, so
a chunk must satisfy three constraints at once:

  1. **Self-contained.** A chunk read alone must still assert what it asserts.
     A sentence like "He stepped down in March" is worthless as evidence.
  2. **Small enough to be precise.** Embedding a 2000-token page averages away
     the one sentence that matters; the dense vector drifts toward the page's
     general topic.
  3. **Offset-preserving.** We keep (start_char, end_char) into the original
     document so the UI can highlight the exact supporting span.

Strategy: recursive structural splitting with a token budget and overlap.
* Split on the strongest available boundary first (headings, then paragraphs,
  then sentences, then whitespace). Structural boundaries correlate with
  semantic boundaries for free -- far cheaper than embedding-based semantic
  chunking, which costs one encoder pass per sentence.
* Overlap of ~15% so a fact that straddles a boundary survives in one piece.
* Prepend the document title and heading path to each chunk ("contextual
  retrieval"): it restores the referent of pronouns and makes "Q3 revenue"
  distinguishable between two companies at near-zero cost.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'\[])")


@dataclass
class Chunk:
    text: str
    doc_id: str
    chunk_id: str
    start_char: int
    end_char: int
    heading_path: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def contextualized(self) -> str:
        """What gets embedded: heading breadcrumb + body."""
        title = str(self.metadata.get("title", ""))
        prefix = " > ".join(p for p in (title, self.heading_path) if p)
        return f"{prefix}\n{self.text}" if prefix else self.text


def _split_keep_offsets(text: str, pattern: re.Pattern, base: int = 0):
    out, last = [], 0
    for m in pattern.finditer(text):
        if m.end() == m.start():
            continue
        piece = text[last : m.start()]
        if piece.strip():
            out.append((piece, base + last))
        last = m.end()
    tail = text[last:]
    if tail.strip():
        out.append((tail, base + last))
    return out or [(text, base)]


def chunk_document(
    text: str,
    doc_id: str,
    metadata: Optional[Dict[str, object]] = None,
    target_tokens: int = 256,
    overlap_tokens: int = 40,
    token_counter: Optional[Callable[[str], int]] = None,
) -> List[Chunk]:
    """Recursive structural chunking with offsets preserved.

    `token_counter` should be the real tokenizer for exact budgets; the default
    (chars/4) is a standard approximation for English and avoids a tokenizer
    pass over the whole corpus during ingestion.
    """
    metadata = dict(metadata or {})
    count = token_counter or (lambda s: max(1, len(s) // 4))

    # --- pass 1: sections delimited by markdown headings -------------------
    sections: List[tuple] = []  # (heading_path, text, offset)
    marks = list(_HEADING.finditer(text))
    if marks:
        stack: List[str] = []
        if marks[0].start() > 0:
            sections.append(("", text[: marks[0].start()], 0))
        for i, m in enumerate(marks):
            level = len(m.group(1))
            stack = stack[: level - 1] + [m.group(2).strip()]
            start = m.end()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            sections.append((" > ".join(stack), text[start:end], start))
    else:
        sections.append(("", text, 0))

    chunks: List[Chunk] = []
    for heading, body, base in sections:
        units = _split_keep_offsets(body, _PARA, base)
        # a paragraph over budget is split again on sentences
        refined = []
        for piece, off in units:
            if count(piece) <= target_tokens:
                refined.append((piece, off))
            else:
                refined.extend(_split_keep_offsets(piece, _SENT, off))

        cur, cur_start, cur_tokens = [], None, 0
        for piece, off in refined:
            n = count(piece)
            if cur and cur_tokens + n > target_tokens:
                body_text = "\n\n".join(cur).strip()
                chunks.append(
                    Chunk(body_text, doc_id, f"{doc_id}#{len(chunks)}", cur_start,
                          cur_start + len(body_text), heading, dict(metadata))
                )
                # carry the tail back as overlap
                keep, kept = [], 0
                for p in reversed(cur):
                    if kept >= overlap_tokens:
                        break
                    keep.insert(0, p)
                    kept += count(p)
                cur, cur_tokens = keep, kept
                cur_start = off - sum(len(p) for p in keep)
            if cur_start is None:
                cur_start = off
            cur.append(piece)
            cur_tokens += n
        if cur:
            body_text = "\n\n".join(cur).strip()
            if body_text:
                chunks.append(
                    Chunk(body_text, doc_id, f"{doc_id}#{len(chunks)}", max(0, cur_start or 0),
                          max(0, cur_start or 0) + len(body_text), heading, dict(metadata))
                )
    return chunks


def chunk_table(rows: Sequence[Sequence[str]], doc_id: str, metadata=None) -> List[Chunk]:
    """Tables are chunked per row, with the header re-attached.

    A table chunked as flat text loses the column->value binding entirely --
    the retrieved text says "2026 1,240" with no way to know 1,240 is revenue.
    Row-level linearisation ("Year: 2026 | Revenue: 1,240") keeps the binding
    and makes cell-level citation possible.
    """
    metadata = dict(metadata or {})
    if not rows:
        return []
    header, *body = rows
    out = []
    for i, row in enumerate(body):
        text = " | ".join(f"{h}: {c}" for h, c in zip(header, row))
        md = dict(metadata)
        md["row"] = i
        out.append(Chunk(text, doc_id, f"{doc_id}#row{i}", 0, len(text), "table", md))
    return out
