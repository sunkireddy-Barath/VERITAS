"""Attention, from scratch.

The mathematics
---------------
For one head with queries Q in R^{L x d}, keys K in R^{L x d}, values V:

    A = softmax( (Q K^T) / sqrt(d) + M )        # M is the causal mask
    O = A V

*Why the 1/sqrt(d)?* If q, k have i.i.d. entries with variance 1, then
q . k = sum_{i=1..d} q_i k_i has variance d. Without the scale, the logits grow
like sqrt(d), softmax saturates, and the gradient through softmax
(diag(a) - a a^T) vanishes. Dividing by sqrt(d) fixes the logit variance at ~1
regardless of head size.

*Why the causal mask?* A language model factorises
p(x_1..x_L) = prod_t p(x_t | x_<t). Setting M_ij = -inf for j > i makes
softmax assign exactly zero weight to the future, so all L positions can be
trained in parallel from one forward pass while each still only sees its past.

*Why multi-head?* One softmax produces one convex combination of value vectors
-- one "lookup". h heads of size d/h cost the same FLOPs but give h independent
lookups (a copy head, a positional head, a syntax head...). Representational
width at constant compute.

Three engineering choices this file makes, and why
--------------------------------------------------
1. **RoPE instead of learned/sinusoidal absolute positions.**
   Rotate (q, k) by angle m*theta at position m. Because a rotation is
   orthogonal, <R_m q, R_n k> = <q, R_{n-m} k>: the attention logit depends
   only on the *relative* distance n - m. Consequences that matter for VERITAS:
   evidence chunks can be concatenated in any order without absolute-position
   artifacts, contexts extrapolate beyond the trained length, and no position
   parameters are learned.

2. **Grouped-Query Attention (GQA).** n_kv_heads <= n_heads; query heads share
   K/V. The KV cache is the memory bottleneck at inference:
   2 * n_layers * n_kv * d_head * L * 2 bytes. With n_heads=8, n_kv=2 that
   cache is 4x smaller, which is what lets a long evidence context fit on a
   consumer GPU. Quality loss is small because K/V are far more redundant
   across heads than Q.

3. **Fused SDPA.** `F.scaled_dot_product_attention` dispatches to FlashAttention
   when available: the L x L matrix is never materialised (tiled online
   softmax), so memory goes O(L^2) -> O(L) and it is 2-4x faster. The explicit
   loop is kept below as `naive_attention` for teaching and as a numerical
   reference in the tests.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- RoPE
def build_rope_cache(
    seq_len: int, head_dim: int, theta: float = 10_000.0, device=None, dtype=torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (seq_len, head_dim/2).

    Frequencies are a geometric series theta^(-2i/d): low dimensions rotate
    fast (local/positional detail), high dimensions rotate slowly (long-range
    identity). Computed once at startup, then it is a lookup -- O(1) per step.
    """
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (L, d/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels. x: (B, H, L, D) -> same shape.

    Treat channels (2i, 2i+1) as a complex number and multiply by e^{i m theta}:
        x'_even = x_even * cos - x_odd  * sin
        x'_odd  = x_even * sin + x_odd  * cos
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


# ------------------------------------------------------- reference (teaching)
def naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True
) -> torch.Tensor:
    """The equation, written out. O(L^2) memory -- reference only."""
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)  # (B,H,Lq,Lk)
    if causal:
        lq, lk = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=lk - lq)
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1).to(q.dtype) @ v


# ------------------------------------------------------------------ KV cache
class KVCache:
    """Pre-allocated per-layer key/value cache for incremental decoding.

    Without it, generating T tokens costs O(T^2) *per step* re-encoding the
    whole prefix -> O(T^3) total. With it, each step is O(T) and generation is
    O(T^2) overall. Pre-allocating (rather than torch.cat-ing) avoids a
    realloc + copy of the entire cache on every single token.
    """

    def __init__(self, batch, n_kv_heads, max_len, head_dim, n_layers, device, dtype):
        shape = (n_layers, batch, n_kv_heads, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0
        self.max_len = max_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        t = k.size(2)
        self.k[layer, :, :, self.pos : self.pos + t] = k
        self.v[layer, :, :, self.pos : self.pos + t] = v
        return self.k[layer, :, :, : self.pos + t], self.v[layer, :, :, : self.pos + t]

    def advance(self, t: int) -> None:
        self.pos += t


# ------------------------------------------------------------------ Attention
class GroupedQueryAttention(nn.Module):
    """Causal multi-head attention with GQA + RoPE + optional KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        # bias=False: RMSNorm already removes the mean, so the bias is
        # redundant and costs params + a kernel launch.
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=bias)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        if self.n_rep > 1:
            # expand+reshape is a view-broadcast, not a copy of the cache
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # is_causal must be False when decoding with a cache (Lq=1, Lk=pos+1):
        # every cached key is in the past, so no mask is needed at all.
        is_causal = cache is None or L > 1
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.resid_drop(self.wo(out))
