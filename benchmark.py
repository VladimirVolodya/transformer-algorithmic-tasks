"""Multi-task benchmark entry point.

Trains a fresh model per task (ADDITION, SORT, DYCK, INDEX) with the fixed
architecture from the `model` config group, evaluates in-domain and OOD exact
match per length, prints a summary table, and saves the metrics JSON.

    uv run python benchmark.py model.pe_variant=rope
    uv run python benchmark.py model.pe_variant=nope benchmark.steps=2000
"""

import json
import warnings
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.benchmark import run_benchmark
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="benchmark")
def main(config):
    bench = config.benchmark

    def make_model(vocab_size, pad_id):
        return instantiate(config.model, vocab_size=vocab_size, pad_id=pad_id)

    results = run_benchmark(
        make_model,
        tasks=list(bench.tasks),
        steps=bench.steps,
        batch_size=bench.batch_size,
        lr=bench.lr,
        train_min_len=bench.train_min_len,
        train_max_len=bench.train_max_len,
        id_lengths=list(bench.id_lengths),
        ood_lengths=list(bench.ood_lengths),
        n_eval=bench.n_eval,
        train_seed=bench.train_seed,
        eval_seed=bench.eval_seed,
        device=bench.device,
    )
    results["model"] = OmegaConf.to_container(config.model)

    # Resolve relative to the repo root (robust to Hydra's run dir).
    save_path = Path(ROOT_PATH) / bench.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {save_path}")

    print(f"\n{'task':<10} {'in-domain':>10} {'OOD mean':>10}")
    for name, task_res in results["tasks"].items():
        print(
            f"{name:<10} {task_res['in_domain_mean']:>10.3f} "
            f"{task_res['ood_mean']:>10.3f}"
        )


if __name__ == "__main__":
    main()
