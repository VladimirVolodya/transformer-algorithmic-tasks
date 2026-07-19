"""Guardrail tests for the multi-task benchmark (SORT / DYCK / INDEX + shared
tokenizer). Mirrors the addition-format guardrails: a wrong answer construction
or loss mask silently invalidates every benchmark number, so these run first.
"""

import numpy as np
import torch

from src.datasets.tokenizer import Tokenizer as LegacyTokenizer
from src.tasks import TASKS, TaskTokenizer
from src.tasks.data import make_eval_set, make_train_batch
from src.tasks.dyck import corrupt, is_balanced, sample_balanced
from src.tasks.vocab import EOS, EQ

ALL_TASKS = list(TASKS)


def _rng(seed):
    return np.random.default_rng(seed)


# --- shared vocabulary ------------------------------------------------------


def test_unified_vocab_extends_legacy():
    legacy, unified = LegacyTokenizer(), TaskTokenizer()
    # ids 0..13 identical -> digit d is id d, pad/eos ids unchanged
    assert unified.itos[: legacy.vocab_size] == legacy.itos
    assert unified.pad_id == legacy.pad_id
    assert unified.eos_id == legacy.eos_id


def test_make_example_masks_answer_only():
    tok = TaskTokenizer()
    rng = _rng(0)
    for name in ALL_TASKS:
        task = TASKS[name]()
        for _ in range(200):
            inst = task.sample_train(rng.integers, 1, 5)
            prompt, answer = task.prompt_symbols(inst), task.answer_symbols(inst)
            input_ids, loss_mask = tok.make_example(prompt, answer)
            assert loss_mask == [0] * len(prompt) + [1] * (len(answer) + 1)
            symbols = tok.decode(input_ids, stop_at_eos=False, strip_special=False)
            assert symbols == prompt + answer + [EOS]
            assert prompt[-1] == EQ  # generation seed is always '='


# --- addition (task wrapper must match the legacy format) -------------------


def test_addition_task_matches_legacy_tokenizer():
    task, legacy = TASKS["addition"](), LegacyTokenizer()
    rng = _rng(1)
    for _ in range(300):
        inst = task.sample_train(rng.integers, 1, 5)
        expected, answer_start = legacy.example_symbols(inst["a"], inst["b"])
        assert task.prompt_symbols(inst) == expected[:answer_start]
        assert task.answer_symbols(inst) + [EOS] == expected[answer_start:]


# --- sort -------------------------------------------------------------------


def test_sort_answer_is_sorted_multiset():
    task = TASKS["sort"]()
    rng = _rng(2)
    for _ in range(500):
        inst = task.sample_train(rng.integers, 1, 8)
        answer = [int(d) for d in task.answer_symbols(inst)]
        assert answer == sorted(inst["arr"])
        assert sorted(answer) == sorted(inst["arr"])  # same multiset


# --- dyck -------------------------------------------------------------------


def test_dyck_reference_checker():
    assert is_balanced(list("()"))
    assert is_balanced(list("([])()"))
    assert not is_balanced(list("([)]"))
    assert not is_balanced(list("((("))
    assert not is_balanced(list(")("))
    assert is_balanced([])


def test_dyck_generators():
    rng = _rng(3)
    for _ in range(300):
        n_pairs = int(rng.integers(1, 10))
        word = sample_balanced(n_pairs, rng.integers)
        assert len(word) == 2 * n_pairs
        assert is_balanced(word)
        assert not is_balanced(corrupt(word, rng.integers))


def test_dyck_labels_match_checker_and_are_mixed():
    task = TASKS["dyck"]()
    rng = _rng(4)
    labels = []
    for _ in range(500):
        inst = task.sample_train(rng.integers, 1, 5)
        assert len(inst["s"]) % 2 == 0
        assert inst["label"] == ("1" if is_balanced(inst["s"]) else "0")
        assert task.answer_symbols(inst) == [inst["label"]]
        labels.append(inst["label"])
    assert 0.3 < np.mean([label == "1" for label in labels]) < 0.7


# --- index ------------------------------------------------------------------


def test_index_answer_is_element_at_index():
    task = TASKS["index"]()
    rng = _rng(5)
    for _ in range(500):
        inst = task.sample_train(rng.integers, 1, 15)
        assert 0 <= inst["idx"] < min(len(inst["arr"]), 10)  # single-digit index
        assert task.answer_symbols(inst) == [str(inst["arr"][inst["idx"]])]


# --- determinism (same train/test values for every architecture) ------------


def test_eval_sets_are_deterministic():
    for name in ALL_TASKS:
        a = make_eval_set(TASKS[name](), length=7, n=20, seed=42)
        b = make_eval_set(TASKS[name](), length=7, n=20, seed=42)
        assert a == b


def test_eval_prompts_have_constant_length_per_task():
    tok = TaskTokenizer()
    for name in ALL_TASKS:
        task = TASKS[name]()
        for length in (3, 11):
            prompts = [
                tok.encode(task.prompt_symbols(inst))
                for inst in make_eval_set(task, length, n=30, seed=0)
            ]
            assert len({len(p) for p in prompts}) == 1


def test_train_stream_is_deterministic_given_seed():
    tok = TaskTokenizer()
    for name in ALL_TASKS:
        task = TASKS[name]()
        batches = []
        for _ in range(2):
            rng = np.random.RandomState(7)
            batches.append(make_train_batch(task, tok, rng.randint, 16, 1, 5))
        assert torch.equal(batches[0]["input_ids"], batches[1]["input_ids"])
        assert torch.equal(batches[0]["loss_mask"], batches[1]["loss_mask"])


# --- end-to-end smoke -------------------------------------------------------


def test_run_benchmark_smoke():
    """Tiny end-to-end run: 4 tasks x 5 steps, json-shaped output."""
    import json

    from src.benchmark import run_benchmark
    from src.model.transformer import DecoderTransformer

    def make_model(vocab_size, pad_id):
        return DecoderTransformer(
            vocab_size=vocab_size,
            d_model=32,
            n_layers=1,
            n_heads=1,
            head_dim=16,
            d_ff=64,
            max_len=64,
            pe_variant="nope",
            pad_id=pad_id,
        )

    results = run_benchmark(
        make_model,
        steps=5,
        batch_size=8,
        id_lengths=[1, 2],
        ood_lengths=[6],
        n_eval=4,
        device="cpu",
    )
    json.dumps(results)  # must be JSON-serializable
    assert set(results["tasks"]) == set(ALL_TASKS)
    for task_res in results["tasks"].values():
        assert set(task_res["in_domain"]) == {"1", "2"}
        assert set(task_res["ood"]) == {"6"}
        for acc in list(task_res["in_domain"].values()) + list(task_res["ood"].values()):
            assert 0.0 <= acc <= 1.0
