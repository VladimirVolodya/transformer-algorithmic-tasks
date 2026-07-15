"""Positional-encoding helpers for the swappable-PE transformer.

Only RoPE needs helpers here; ``absolute`` uses a plain ``nn.Embedding`` and
``nope`` uses nothing (causal masking only).
"""

from __future__ import annotations

import torch


def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """Precompute ``cos``/``sin`` tables of shape ``[seq_len, head_dim]``.

    Uses the standard "rotate-half" RoPE layout (GPT-NeoX style). ``head_dim``
    must be even.
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to ``x`` of shape ``[B, n_heads, T, head_dim]``.

    ``cos``/``sin`` have shape ``[T, head_dim]``.
    """
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin
