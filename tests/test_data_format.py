"""Guardrail tests for the reversed-digit format and answer-only loss mask.

Non-reversed digits or a wrong loss mask silently break the whole validation
premise (plan §2 guardrails), so these run before any training.
"""

import numpy as np

from src.datasets.tokenizer import EOS, EQ, PLUS, Tokenizer


def _reconstruct(symbols):
    """Recover (a, b, s) from ``rev(a) + rev(b) = rev(s) <eos>`` symbols."""
    plus = symbols.index(PLUS)
    eq = symbols.index(EQ)
    rev_a = symbols[:plus]
    rev_b = symbols[plus + 1 : eq]
    rev_s = [t for t in symbols[eq + 1 :] if t != EOS]
    a = int("".join(rev_a[::-1]))
    b = int("".join(rev_b[::-1]))
    s = int("".join(rev_s[::-1]))
    return a, b, s


def test_specific_example_matches_plan():
    tok = Tokenizer()
    symbols, answer_start = tok.example_symbols(58392, 74918)
    expected = (
        list("29385") + [PLUS] + list("81947") + [EQ] + list("013331") + [EOS]
    )
    assert symbols == expected
    assert answer_start == symbols.index(EQ) + 1


def test_reversed_format_roundtrip():
    tok = Tokenizer()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a = int(rng.integers(0, 10 ** int(rng.integers(1, 7))))
        b = int(rng.integers(0, 10 ** int(rng.integers(1, 7))))
        symbols, _ = tok.example_symbols(a, b)
        ra, rb, rs = _reconstruct(symbols)
        assert (ra, rb) == (a, b)
        assert rs == a + b  # sum is correct AND reversed correctly


def test_loss_mask_marks_answer_only():
    tok = Tokenizer()
    rng = np.random.default_rng(1)
    for _ in range(1000):
        a = int(rng.integers(0, 100000))
        b = int(rng.integers(0, 100000))
        input_ids, loss_mask = tok.make_example(a, b)
        symbols, answer_start = tok.example_symbols(a, b)

        # zero on the prompt (rev(a), '+', rev(b), '='), one on the answer
        assert loss_mask[:answer_start] == [0] * answer_start
        assert loss_mask[answer_start:] == [1] * (len(symbols) - answer_start)
        # exactly len(rev(s)) digits + <eos> are supervised
        assert sum(loss_mask) == len(str(a + b)) + 1
        # ids decode back to the exact symbol stream
        assert tok.decode(input_ids, stop_at_eos=False, strip_special=False) == symbols


def test_prompt_is_up_to_equals():
    tok = Tokenizer()
    prompt_ids = tok.make_prompt(123, 45)
    assert tok.decode(prompt_ids, stop_at_eos=False, strip_special=False) == (
        list("321") + [PLUS] + list("54") + [EQ]
    )


def test_vocab_size_is_14():
    assert Tokenizer().vocab_size == 14
