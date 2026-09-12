"""The VERITAS language model: a decoder-only Transformer, built from scratch.

Block layout (pre-norm, the post-2020 standard):

    h = x + Attn(RMSNorm(x))
    y = h + SwiGLU(RMSNorm(h))

*Why pre-norm and not the original post-norm?* In post-norm
(x + Sublayer(x) then LayerNorm) the residual path is re-normalised at every
layer, so the gradient is repeatedly rescaled and deep stacks need a learning
rate warmup just to avoid diverging. Pre-norm leaves an unnormalised identity
path from the loss to the embedding, so gradients reach layer 0 intact and
training is stable without tricks.

*Why RMSNorm and not LayerNorm?*
    LayerNorm: (x - mu) / sqrt(var + eps) * g + b
    RMSNorm:   x / sqrt(mean(x^2) + eps) * g
RMSNorm drops the mean subtraction and the bias: ~2x fewer reduction passes
over the feature axis, no b parameters, and empirically equal quality. The
mean-centering turns out to do no work once the residual stream is well
conditioned.

*Why SwiGLU and not GELU/ReLU?*
    FFN_swiglu(x) = W2( silu(W1 x) * W3 x )
A gate (W3 x) multiplies the activation, giving a data-dependent,
multiplicative interaction that a single-matrix FFN cannot express. It uses 3
matrices instead of 2, so hidden width is set to (8/3)*d_model rounded to a
multiple of 64 -- this keeps the parameter count equal to a 4*d_model
GELU FFN while consistently beating it in loss per parameter.

*Why tie the embedding and the output head?* The input embedding maps
token -> vector; the LM head maps vector -> token logits. They are inverse maps
over the same vocabulary, so sharing the matrix removes V*d parameters -- at
vocab 8192, d 384 that is 3.1M parameters, a large fraction of a small model --
and acts as a regulariser. Standard since Press & Wolf (2017).

*Initialisation.* Residual-path output projections (wo, w2) are scaled by
1/sqrt(2*n_layers). Each layer adds its output into the residual stream, so
without this the stream's variance grows linearly with depth and the final
logits explode. This is the GPT-2 initialisation rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention, KVCache, build_rope_cache


@dataclass
class ModelConfig:
    """Defaults sized for a single consumer GPU (~30M params, ~8GB budget)."""

    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2           # GQA: 4x smaller KV cache
    max_seq_len: int = 512
    dropout: float = 0.0          # 0.0 while data > params (no overfit risk)
    rope_theta: float = 10_000.0
    tie_weights: bool = True
    ffn_multiple_of: int = 64     # keep GEMM shapes tensor-core friendly

    @property
    def d_ff(self) -> int:
        hidden = int(8 * self.d_model / 3)
        m = self.ffn_multiple_of
        return m * ((hidden + m - 1) // m)

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # float32 for the reduction even under bf16 autocast: mean(x^2) is the
        # one place where low precision actually bites (variance underflow).
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # value branch
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # gate branch
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # projection back
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(
            cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.dropout
        )
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x, cos, sin, cache: Optional[KVCache] = None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, self.layer_idx)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VeritasLM(nn.Module):
    """Decoder-only LM. Also exposes `hidden_states` so the same trunk can be
    reused as the retrieval encoder (see rag/embeddings.py)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.d_model // cfg.n_heads, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "ffn.w2.weight")):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ util
    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Final-layer states, no LM head. Used by the retrieval encoder."""
        # Truncate rather than raise: an over-long chunk should still produce a
        # usable embedding, and the encoder is called on arbitrary corpus text.
        idx = idx[:, : self.cfg.max_seq_len]
        B, L = idx.shape
        cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin, None)
        return self.norm(x)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cache: Optional[KVCache] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L = idx.shape
        start = cache.pos if cache is not None else 0
        if start + L > self.cfg.max_seq_len:
            # Without this the RoPE slice comes back short (or empty) and the
            # failure surfaces as an unrelated broadcast error deep inside
            # attention. Fail where the mistake actually is.
            raise ValueError(
                f"sequence position {start + L} exceeds max_seq_len "
                f"{self.cfg.max_seq_len}: truncate the prompt or raise max_seq_len"
            )
        cos = self.rope_cos[start : start + L].to(idx.device)
        sin = self.rope_sin[start : start + L].to(idx.device)

        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, cos, sin, cache)
        x = self.norm(x)
        if cache is not None:
            cache.advance(L)

        if targets is None:
            # Inference: only the last position is needed. Computing the head
            # for all L positions would be the single largest wasted GEMM.
            return self.lm_head(x[:, -1:, :]), None

        logits = self.lm_head(x)
        if loss_mask is None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        else:
            # Instruction tuning: only score assistant tokens.
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            m = loss_mask.reshape(-1).float()
            loss = (per_tok * m).sum() / m.sum().clamp(min=1.0)
        return logits, loss

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        eos_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Sampling with a KV cache.

        temperature scales logits (low = argmax-like = better for factual
        answers); top-k truncates to the k best; top-p (nucleus) truncates to
        the smallest set whose mass exceeds p, which adapts to how peaked the
        distribution already is. For VERITAS answer synthesis use
        temperature <= 0.3: we want the evidence to decide the wording, not RNG.
        """
        self.eval()
        B = idx.size(0)
        # Budget the context: prompt + new tokens must fit max_seq_len. The
        # prompt keeps its tail (the most recent context), and generation stops
        # at the boundary rather than running off the end of the RoPE table.
        idx = idx[:, -self.cfg.max_seq_len :]
        budget = self.cfg.max_seq_len - idx.size(1)
        if budget <= 0:
            return idx
        max_new_tokens = min(max_new_tokens, budget)
        cache = None
        if use_cache:
            cache = KVCache(
                B, self.cfg.n_kv_heads, self.cfg.max_seq_len,
                self.cfg.d_model // self.cfg.n_heads, self.cfg.n_layers,
                idx.device, self.tok_emb.weight.dtype,
            )
            logits, _ = self(idx, cache=cache)
        else:
            logits, _ = self(idx)

        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                next_tok = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p:
                    s, i = torch.sort(logits, descending=True, dim=-1)
                    cum = torch.softmax(s, -1).cumsum(-1)
                    drop = cum - torch.softmax(s, -1) > top_p
                    s = s.masked_fill(drop, float("-inf"))
                    logits = torch.full_like(logits, float("-inf")).scatter(-1, i, s)
                next_tok = torch.multinomial(torch.softmax(logits, -1), 1)

            if eos_id is not None:
                next_tok = torch.where(finished[:, None], torch.full_like(next_tok, eos_id), next_tok)
                finished |= next_tok.squeeze(-1) == eos_id
            idx = torch.cat([idx, next_tok], dim=1)
            if finished.all():
                break
            if idx.size(1) >= self.cfg.max_seq_len:
                break
            if cache is not None:
                logits, _ = self(next_tok, cache=cache)
            else:
                logits, _ = self(idx)
        return idx

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str) -> None:
        torch.save({"config": self.cfg.to_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "VeritasLM":
        ck = torch.load(path, map_location=device, weights_only=False)
        model = cls(ModelConfig(**ck["config"]))
        model.load_state_dict(ck["state_dict"])
        return model.to(device)
