"""Sanity checks for the Phase-2 architecture grid (~800k params, absolute PE)."""

from src.model.architectures import ARCH_NAMES, ARCHITECTURES
from src.model.transformer import DecoderTransformer


def test_all_arches_near_800k_and_nope_pe():
    for name in ARCH_NAMES:
        kw = ARCHITECTURES[name]
        assert kw["pe_variant"] == "nope"
        assert kw["n_layers"] == 4
        assert kw["max_len"] >= 128
        assert kw["d_model"] % kw["n_heads"] == 0
        assert kw["head_dim"] == kw["d_model"] // kw["n_heads"]
        model = DecoderTransformer(vocab_size=19, pad_id=13, **kw)
        n = model.num_parameters()
        assert 0.7e6 <= n <= 1.0e6, f"Arch {name}: {n:,} outside ~800k band"


def test_ood_addition_fits_in_max_len():
    """Longest OOD addition prompt+answer must fit under absolute PE max_len."""
    # length-30 operands: rev(a)+rev(b)=rev(s)<eos> <= 30+1+30+1+31+1 = 94
    kw = ARCHITECTURES["B"]
    model = DecoderTransformer(vocab_size=19, pad_id=13, **kw)
    assert model.max_len >= 94
