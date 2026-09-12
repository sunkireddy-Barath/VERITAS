"""Instruction tuning (SFT) with assistant-only loss masking.

Chat format -- the control tokens are real vocabulary entries, not text:

    <|bos|><|system|>...<|user|>...<|assistant|>...<|eos|>

Why single tokens rather than the literal string "### Assistant:"? A string
costs 4-6 tokens of context per turn and, worse, the model can *generate* a
convincing fake role header mid-answer. A dedicated id cannot be confused with
content, and the boundary is unambiguous when we parse the output.

Why mask the loss to assistant spans? Training on the prompt tokens teaches the
model to *generate questions*, which is wasted capacity and actively harmful --
it raises the probability of the model continuing with a new fabricated user
turn. We zero the loss on system/user tokens and score only what the assistant
should have said.

VERITAS-specific SFT objectives. Beyond "follow instructions", the tuning set
teaches four behaviours the pipeline depends on:
  1. answer strictly from the <|evidence|> block,
  2. emit <|claim|> spans so the verifier can align claims to evidence,
  3. emit <|time|> qualifiers when a fact is time-bounded,
  4. emit <|unknown|> when the evidence does not support an answer.
(4) is the important one: abstention must be a *trained behaviour*, not a
post-hoc filter, otherwise the model always produces a fluent guess that the
verifier then has to delete.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class Turn:
    role: str  # system | user | assistant
    content: str


def render(turns: Sequence[Turn]) -> str:
    parts = ["<|bos|>"]
    for t in turns:
        parts.append(f"<|{t.role}|>{t.content}")
    parts.append("<|eos|>")
    return "".join(parts)


def build_example(
    tokenizer, turns: Sequence[Turn], max_len: int
) -> Optional[Dict[str, List[int]]]:
    """Encode one conversation into ids + a 0/1 loss mask over assistant spans."""
    ids: List[int] = [tokenizer.special_tokens["<|bos|>"]]
    mask: List[int] = [0]
    for t in turns:
        role_id = tokenizer.special_tokens[f"<|{t.role}|>"]
        body = tokenizer.encode(t.content)
        ids.append(role_id)
        mask.append(0)  # never score the role marker itself
        ids.extend(body)
        mask.extend([1 if t.role == "assistant" else 0] * len(body))
    ids.append(tokenizer.special_tokens["<|eos|>"])
    mask.append(1)  # DO score EOS: the model must learn when to stop
    if len(ids) > max_len:
        return None  # drop rather than truncate: a cut answer teaches truncation
    return {"ids": ids, "mask": mask}


class SFTDataset:
    """Padded batches with a loss mask.

    Padding (not packing) is used here on purpose: SFT sets are small, and
    packing several conversations into one window lets an answer attend to an
    unrelated earlier conversation, which measurably increases hallucination in
    grounded-answer settings.
    """

    def __init__(self, examples: List[Dict[str, List[int]]], pad_id: int, max_len: int) -> None:
        self.pad_id = pad_id
        self.max_len = max_len
        self.ids = np.full((len(examples), max_len), pad_id, dtype=np.int64)
        self.mask = np.zeros((len(examples), max_len), dtype=np.int64)
        for i, ex in enumerate(examples):
            n = len(ex["ids"])
            self.ids[i, :n] = ex["ids"]
            self.mask[i, :n] = ex["mask"]

    def __len__(self) -> int:
        return len(self.ids)

    def batch(self, batch_size: int, device: str = "cpu", generator=None):
        idx = torch.randint(len(self), (batch_size,), generator=generator)
        seq = torch.from_numpy(self.ids[idx.numpy()])
        msk = torch.from_numpy(self.mask[idx.numpy()])
        x, y = seq[:, :-1], seq[:, 1:].clone()
        m = msk[:, 1:]                 # mask aligns with the *target* position
        y[m == 0] = -100               # ignore_index -> no loss, no gradient
        return x.to(device), y.to(device), m.to(device)


def load_jsonl(path: str | Path) -> Iterable[List[Turn]]:
    """Read {"messages": [{"role": ..., "content": ...}, ...]} per line."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield [Turn(m["role"], m["content"]) for m in rec["messages"]]


def evidence_prompt(question: str, evidence: Sequence[Dict[str, str]]) -> str:
    """Render a grounded prompt. Every evidence block carries its id and date so
    the model can cite [E1] and reason about recency inside the context."""
    lines = []
    for i, e in enumerate(evidence, 1):
        lines.append(
            f"<|evidence|>[E{i}] source={e.get('source', '?')} "
            f"date={e.get('date', 'unknown')} tier={e.get('tier', '?')}\n{e['text']}"
        )
    lines.append(f"\nQuestion: {question}")
    lines.append(
        "Answer using only the evidence above. Cite [E#] after each claim. "
        "If the evidence is insufficient or conflicting, say so explicitly."
    )
    return "\n".join(lines)
