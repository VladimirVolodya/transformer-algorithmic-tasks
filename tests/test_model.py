"""Model sanity checks (all on CPU, no GPU required)."""

import torch

from src.datasets.tokenizer import Tokenizer
from src.model.transformer import DecoderTransformer

VARIANTS = ["absolute", "nope", "rope", "abs_shift"]


def _make_model(pe_variant):
    tok = Tokenizer()
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
        assert 0.7e6 <= n <= 0.9e6, f"{variant}: {n:,} params outside ~0.8M"


def test_forward_shape():
    tok = Tokenizer()
    for variant in VARIANTS:
        model = _make_model(variant)
        x = torch.randint(0, tok.vocab_size, (4, 20))
        assert model(input_ids=x)["logits"].shape == (4, 20, tok.vocab_size)


def test_generate_shape_and_stops():
    tok = Tokenizer()
    model = _make_model("rope")
    prompt = torch.randint(0, 10, (3, 8))
    gen = model.generate(prompt, max_new_tokens=10, eos_id=tok.eos_id)
    assert gen.shape[0] == 3 and gen.shape[1] <= 10


def test_absolute_pe_survives_ood_length():
    tok = Tokenizer()
    model = _make_model("absolute")
    # longest OOD sequence (~49 tokens) is below max_len=64 -> must not raise
    x = torch.randint(0, tok.vocab_size, (2, 49))
    assert model(input_ids=x)["logits"].shape == (2, 49, tok.vocab_size)


def test_abs_shift_eval_is_deterministic_and_offset_free():
    tok = Tokenizer()
    model = _make_model("abs_shift").eval()
    x = torch.randint(0, tok.vocab_size, (2, 20))
    logits1 = model(input_ids=x)["logits"]
    logits2 = model(input_ids=x)["logits"]
    assert torch.equal(logits1, logits2), "eval must use a fixed offset of 0"


def test_abs_shift_train_mode_handles_full_max_len():
    tok = Tokenizer()
    model = _make_model("abs_shift").train()
    # T == max_len -> the only legal offset is 0; must not raise or index OOB
    x = torch.randint(0, tok.vocab_size, (2, 64))
    assert model(input_ids=x)["logits"].shape == (2, 64, tok.vocab_size)


def test_abs_shift_param_count_matches_absolute():
    n_abs = _make_model("absolute").num_parameters()
    n_shift = _make_model("abs_shift").num_parameters()
    assert n_abs == n_shift, "abs_shift must only change indexing, not capacity"


def test_runs_on_cpu():
    model = _make_model("nope")
    assert next(model.parameters()).device.type == "cpu"
