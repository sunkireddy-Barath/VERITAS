"""Training loop: AdamW + cosine schedule + grad accumulation + bf16 + ckpt.

Why each piece exists
---------------------
* **AdamW, decoupled weight decay.** Adam's update is
  m_hat / (sqrt(v_hat) + eps). Classic L2 folds decay into the gradient, so it
  gets divided by sqrt(v) too -- parameters with large gradient variance end up
  barely regularised. AdamW applies `p -= lr * wd * p` separately, making decay
  uniform. Decay is applied only to matrices: biases, RMSNorm gains and
  embeddings are 1-D and shrinking them just damages the scale of the residual
  stream.
* **Linear warmup then cosine decay.** At step 0 Adam's second moment estimate
  v is near zero, so the effective step is enormous and one bad batch can
  destroy the init. Warmup ramps lr linearly for a few hundred steps. Cosine
  decay then anneals smoothly to ~lr/10, which empirically beats step decay at
  a fixed budget because the late small steps do the fine-grained fitting.
* **Gradient accumulation.** Large batches stabilise the gradient estimate
  (noise ~ 1/sqrt(B)) but batch is capped by VRAM. Accumulating k micro-batches
  before stepping gives the statistics of batch k*B at the memory of B. Loss is
  divided by k so the gradient magnitude matches a true large batch.
* **bf16 autocast.** bf16 has fp32's exponent range with fewer mantissa bits:
  ~2x throughput and ~2x memory on tensor cores, and unlike fp16 it cannot
  overflow, so no GradScaler is needed. Master weights stay fp32.
* **Gradient clipping at global norm 1.0.** Language data is heavy-tailed; one
  batch of unusual text can produce a gradient 100x the norm of the rest and
  undo hours of training. Clipping bounds the damage of any single step.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch


@dataclass
class TrainConfig:
    max_steps: int = 3000
    batch_size: int = 16
    grad_accum: int = 4
    seq_len: int = 512
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)      # beta2=0.95 (not 0.999) -> faster adaptation
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_iters: int = 40
    log_every: int = 25
    ckpt_dir: str = "checkpoints"
    ckpt_every: int = 500
    compile: bool = False           # torch.compile: ~1.3-2x, slow first step
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    seed: int = 1337


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.lr * cfg.min_lr_ratio
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * coeff)


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # fused AdamW does the whole update in one CUDA kernel (fewer launches)
    fused = cfg.device.startswith("cuda")
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, eps=1e-8, fused=fused)


@torch.no_grad()
def estimate_loss(model, dataset, cfg: TrainConfig, iters: int) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = dataset.batch(cfg.batch_size, cfg.device)
        with torch.autocast(device_type=cfg.device.split(":")[0], dtype=getattr(torch, cfg.dtype),
                            enabled=cfg.device.startswith("cuda")):
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


@dataclass
class TrainState:
    step: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


def train(
    model,
    train_ds,
    val_ds=None,
    cfg: Optional[TrainConfig] = None,
    on_log: Optional[Callable[[Dict[str, float]], None]] = None,
) -> TrainState:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True   # 8x faster fp32 matmul
        torch.backends.cudnn.allow_tf32 = True
    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)

    model = model.to(cfg.device)
    raw_model = model
    if cfg.compile:
        model = torch.compile(model)

    opt = make_optimizer(raw_model, cfg)
    state = TrainState()
    amp = cfg.device.startswith("cuda")
    dev_type = cfg.device.split(":")[0]
    best_val = float("inf")
    t0 = time.time()
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.seq_len

    model.train()
    for step in range(cfg.max_steps):
        lr = cosine_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)  # set_to_none frees the grad buffers
        total = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_ds.batch(cfg.batch_size, cfg.device)
            with torch.autocast(device_type=dev_type, dtype=getattr(torch, cfg.dtype), enabled=amp):
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum
            loss.backward()
            total += loss.item()

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        opt.step()
        state.step = step + 1

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            dt = time.time() - t0
            rec = {
                "step": step,
                "loss": total,
                "ppl": math.exp(min(20, total)),
                "lr": lr,
                "grad_norm": float(norm),
                "tok_per_s": tokens_per_step * (step + 1) / max(dt, 1e-6),
            }
            state.history.append(rec)
            msg = (f"step {step:5d} | loss {total:.4f} | ppl {rec['ppl']:8.2f} | "
                   f"lr {lr:.2e} | gn {float(norm):.2f} | {rec['tok_per_s']:,.0f} tok/s")
            print(msg)
            if on_log:
                on_log(rec)

        if val_ds is not None and (step + 1) % cfg.eval_every == 0:
            vl = estimate_loss(model, val_ds, cfg, cfg.eval_iters)
            print(f"  >> val loss {vl:.4f} | val ppl {math.exp(min(20, vl)):.2f}")
            state.history.append({"step": step, "val_loss": vl, "val_ppl": math.exp(min(20, vl))})
            if vl < best_val:
                best_val = vl
                raw_model.save(str(Path(cfg.ckpt_dir) / "best.pt"))

        if (step + 1) % cfg.ckpt_every == 0:
            torch.save(
                {"config": raw_model.cfg.to_dict(), "state_dict": raw_model.state_dict(),
                 "optimizer": opt.state_dict(), "step": step + 1},
                Path(cfg.ckpt_dir) / "last.pt",
            )
    raw_model.save(str(Path(cfg.ckpt_dir) / "final.pt"))
    return state
