"""Fixed ~800k-parameter architecture grid for the Phase-2 experiment.

All keep ``n_layers=4``, ``pe_variant="nope"``, and ``max_len=128``.
``head_dim = d_model // n_heads``.

NoPE relies on the causal mask for order; this removes absolute/RoPE length
binding that collapsed OOD on far lengths.

``attn_frac`` below is the share of Attention vs MLP block params
(``4 L d^2 / (4 L d^2 + 2 L d d_ff)``), the natural x-axis for plots.

Param counts with vocab_size=19 (approximate):

* H 7-head (attn-max):   ~0.78M  attn_frac≈0.75
* A Attention-Heavy:     ~0.90M  attn_frac≈0.67
* E Mid Attention:       ~0.83M  attn_frac≈0.50
* B Baseline:            ~0.80M  attn_frac≈0.33
* F Mild Bottleneck:     ~0.81M  attn_frac≈0.25
* I 3-head mid:          ~0.75M  attn_frac≈0.20
* C Bottleneck:          ~0.81M  attn_frac≈0.13
* G Strong Bottleneck:   ~0.80M  attn_frac≈0.08
* D Extreme Bottleneck:  ~0.84M  attn_frac≈0.05
"""

from __future__ import annotations

from typing import Any

# Name -> kwargs for DecoderTransformer (minus vocab_size / pad_id).
ARCHITECTURES: dict[str, dict[str, Any]] = {
    "H": {
        "d_model": 189,
        "n_heads": 7,
        "head_dim": 27,  # 189 // 7
        "d_ff": 126,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "A": {
        "d_model": 192,
        "n_heads": 6,
        "head_dim": 32,  # 192 // 6
        "d_ff": 192,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "E": {
        "d_model": 160,
        "n_heads": 5,
        "head_dim": 32,  # 160 // 5
        "d_ff": 320,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "B": {
        "d_model": 128,
        "n_heads": 4,
        "head_dim": 32,  # 128 // 4
        "d_ff": 512,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "F": {
        "d_model": 112,
        "n_heads": 4,
        "head_dim": 28,  # 112 // 4
        "d_ff": 672,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "I": {
        "d_model": 96,
        "n_heads": 3,
        "head_dim": 32,  # 96 // 3
        "d_ff": 768,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "C": {
        "d_model": 80,
        "n_heads": 2,
        "head_dim": 40,  # 80 // 2
        "d_ff": 1088,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "G": {
        "d_model": 64,
        "n_heads": 2,
        "head_dim": 32,  # 64 // 2
        "d_ff": 1408,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
    "D": {
        "d_model": 48,
        "n_heads": 1,
        "head_dim": 48,  # 48 // 1
        "d_ff": 2048,
        "n_layers": 4,
        "max_len": 128,
        "pe_variant": "nope",
        "dropout": 0.0,
    },
}

# Ordered by decreasing attention share.
ARCH_NAMES = ("H", "A", "E", "B", "F", "I", "C", "G", "D")
