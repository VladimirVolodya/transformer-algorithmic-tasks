"""Model sanity checks (all on CPU, no GPU required)."""

import torch

from src.model.transformer import DecoderTransformer
from src.tasks.vocab import TaskTokenizer

VARIANTS = ["absolute", "nope", "rope"]


def _make_model(pe_variant):
    tok = TaskTokenizer()
    return DecoderTransformer(
        vocab_size=tok.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=2,
        head_dim=64,
        d_ff=512,
        max_len=64,
        pe_variant=pe_variant,
        pad_id=tok.pad_id,
    )


def test_param_count_is_about_0_8M():
    for variant in VARIANTS:
        n = _make_model(variant).num_parameters()
        assert 0.7e6 <= n <= 1.0e6, f"{variant}: {n:,} params outside ~0.8M"


def test_forward_shape():
    tok = TaskTokenizer()
    for variant in VARIANTS:
        model = _make_model(variant)
        x = torch.randint(0, tok.vocab_size, (4, 20))
        assert model(input_ids=x)["logits"].shape == (4, 20, tok.vocab_size)


def test_generate_shape_and_stops():
    tok = TaskTokenizer()
    model = _make_model("rope")
    prompt = torch.randint(0, 10, (3, 8))
    gen = model.generate(prompt, max_new_tokens=10, eos_id=tok.eos_id)
    assert gen.shape[0] == 3 and gen.shape[1] <= 10


def test_absolute_pe_survives_ood_length():
    tok = TaskTokenizer()
    model = _make_model("absolute")
    # longest OOD sequence (~49 tokens) is below max_len=64 -> must not raise
    x = torch.randint(0, tok.vocab_size, (2, 49))
    assert model(input_ids=x)["logits"].shape == (2, 49, tok.vocab_size)


def test_runs_on_cpu():
    model = _make_model("nope")
    assert next(model.parameters()).device.type == "cpu"
