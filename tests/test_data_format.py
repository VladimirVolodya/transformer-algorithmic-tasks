"""Guardrail tests for ADDITION reversed-digit format and answer-only loss mask.

A wrong format or loss mask silently invalidates every benchmark number, so
these run before anything else.
"""

import numpy as np

from src.tasks.addition import AdditionTask
from src.tasks.vocab import EOS, EQ, PLUS, TaskTokenizer


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
    task, tok = AdditionTask(), TaskTokenizer()
    inst = {"a": 58392, "b": 74918}
    prompt = task.prompt_symbols(inst)
    answer = task.answer_symbols(inst)
    symbols = prompt + answer + [EOS]
    expected = (
        list("29385") + [PLUS] + list("81947") + [EQ] + list("013331") + [EOS]
    )
    assert symbols == expected
    input_ids, loss_mask = tok.make_example(prompt, answer)
    assert loss_mask[: len(prompt)] == [0] * len(prompt)
    assert loss_mask[len(prompt) :] == [1] * (len(answer) + 1)


def test_reversed_format_roundtrip():
    task = AdditionTask()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a = int(rng.integers(0, 10 ** int(rng.integers(1, 7))))
        b = int(rng.integers(0, 10 ** int(rng.integers(1, 7))))
        inst = {"a": a, "b": b}
        symbols = task.prompt_symbols(inst) + task.answer_symbols(inst) + [EOS]
        ra, rb, rs = _reconstruct(symbols)
        assert (ra, rb) == (a, b)
        assert rs == a + b


def test_loss_mask_marks_answer_only():
    task, tok = AdditionTask(), TaskTokenizer()
    rng = np.random.default_rng(1)
    for _ in range(1000):
        a = int(rng.integers(0, 100000))
        b = int(rng.integers(0, 100000))
        inst = {"a": a, "b": b}
        prompt, answer = task.prompt_symbols(inst), task.answer_symbols(inst)
        input_ids, loss_mask = tok.make_example(prompt, answer)
        symbols = prompt + answer + [EOS]

        assert loss_mask[: len(prompt)] == [0] * len(prompt)
        assert loss_mask[len(prompt) :] == [1] * (len(answer) + 1)
        assert sum(loss_mask) == len(str(a + b)) + 1
        assert tok.decode(input_ids, stop_at_eos=False, strip_special=False) == symbols


def test_prompt_ends_with_equals():
    task = AdditionTask()
    prompt = task.prompt_symbols({"a": 123, "b": 45})
    assert prompt == list("321") + [PLUS] + list("54") + [EQ]


def test_vocab_size_is_19():
    assert TaskTokenizer().vocab_size == 19
