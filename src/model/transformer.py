"""Minimal decoder-only transformer with a swappable positional-encoding flag.

The ``pe_variant`` flag selects the only thing that differs across the ablation:

* ``"absolute"`` -- learned absolute position embeddings added to token
  embeddings. Positions beyond the training range are untrained -> expected to
  **fail** length-generalization (the point of the experiment).
* ``"nope"``     -- no positional encoding; relies on causal masking only.
  Expected to generalize best.
* ``"rope"``     -- rotary embeddings applied to Q/K inside attention. Expected
  to generalize reasonably.

Everything else is identical across variants, so any OOD difference is
attributable to the PE alone. The Small config (defaults below) is ~0.8M params.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.model.positional import apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, pe_variant, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner = n_heads * head_dim
        self.pe_variant = pe_variant
        self.q_proj = nn.Linear(d_model, self.inner)
        self.k_proj = nn.Linear(d_model, self.inner)
        self.v_proj = nn.Linear(d_model, self.inner)
        self.out_proj = nn.Linear(self.inner, d_model)
        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cache=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pe_variant == "rope":
            cos, sin = rope_cache
            q = apply_rope(q, cos[:T], sin[:T])
            k = apply_rope(k, cos[:T], sin[:T])

        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, self.inner)
        return self.resid_dropout(self.out_proj(attn))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, d_ff, pe_variant, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, head_dim, pe_variant, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope_cache=None):
        x = x + self.attn(self.ln1(x), rope_cache)  # pre-norm
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 2,
        head_dim: int = 64,
        d_ff: int = 512,
        max_len: int = 128,
        pe_variant: str = "absolute",
        dropout: float = 0.0,
        pad_id: int = 13,
    ):
        super().__init__()
        assert pe_variant in {"absolute", "nope", "rope"}, pe_variant
        self.pe_variant = pe_variant
        self.max_len = max_len
        self.head_dim = head_dim
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model) if pe_variant == "absolute" else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, head_dim, d_ff, pe_variant, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # untied

        if pe_variant == "rope":
            cos, sin = build_rope_cache(max_len, head_dim, device="cpu")
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _rope_cache(self, seq_len, device):
        # Rebuild if a sequence longer than the cached table appears (rare; our
        # OOD lengths stay under max_len). Keeps RoPE robust to any length.
        if self.rope_cos.size(0) < seq_len:
            cos, sin = build_rope_cache(seq_len, self.head_dim, device=device)
            self.rope_cos, self.rope_sin = cos, sin
        return self.rope_cos.to(device), self.rope_sin.to(device)

    def forward(self, input_ids, **batch):
        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        if self.pe_variant == "absolute":
            assert T <= self.max_len, (
                f"seq len {T} exceeds absolute-PE max_len {self.max_len}; "
                "increase model.max_len"
            )
            pos = torch.arange(T, device=input_ids.device)
            x = x + self.pos_emb(pos)[None]

        x = self.drop(x)

        rope_cache = self._rope_cache(T, x.device) if self.pe_variant == "rope" else None
        for block in self.blocks:
            x = block(x, rope_cache)

        x = self.ln_f(x)
        return {"logits": self.lm_head(x)}

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, eos_id):
        """Greedy autoregressive decoding.

        Args:
            prompt_ids (LongTensor): ``[B, P]`` prompts (same length within a
                call -- guaranteed for a fixed-length eval set).
            max_new_tokens (int): decoding cap.
            eos_id (int): stop token.
        Returns:
            LongTensor ``[B, steps]`` of generated answer tokens (prompt
            excluded). Sequences that emitted ``<eos>`` are padded with
            ``eos_id`` afterwards.
        """
        was_training = self.training
        self.eval()
        device = prompt_ids.device
        B = prompt_ids.size(0)
        seq = prompt_ids
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = []
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids=seq)["logits"]
            next_tok = logits[:, -1, :].argmax(dim=-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
            generated.append(next_tok)
            seq = torch.cat([seq, next_tok[:, None]], dim=1)
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break
        if was_training:
            self.train()
        return torch.stack(generated, dim=1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __str__(self):
        all_p = self.num_parameters()
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            super().__str__()
            + f"\nPE variant: {self.pe_variant}"
            + f"\nAll parameters: {all_p:,}"
            + f"\nTrainable parameters: {train_p:,}"
        )
