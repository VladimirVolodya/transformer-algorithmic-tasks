import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Autoregressive-greedy exact-match evaluation at a single operand length.

    Loads a trained checkpoint, decodes ``inferencer.n_samples`` fixed-length
    examples, and prints the exact-match accuracy plus a few decoded examples.
    For the full per-length curve / money plot, use the analysis notebook.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    model = instantiate(config.model)
    print(model)

    # Resolve the checkpoint relative to the repo root (robust to Hydra's run dir).
    from_pretrained = str(ROOT_PATH / config.inferencer.from_pretrained)

    inferencer = Inferencer(
        model=model,
        device=device,
        from_pretrained=from_pretrained,
        seed=config.inferencer.seed,
        batch_size=config.inferencer.get("batch_size", 256),
    )

    length = config.inferencer.eval_length
    n = config.inferencer.n_samples
    num_show = config.inferencer.get("num_show", 5)

    accuracy, examples = inferencer.exact_match(length, n, return_examples=num_show)

    print(
        f"\nAutoregressive exact-match @ operand length {length} "
        f"(n={n}): {accuracy:.4f}\n"
    )
    for ex in examples:
        flag = "OK" if ex["correct"] else "XX"
        print(
            f"  [{flag}] {ex['a']} + {ex['b']}  "
            f"pred={ex['pred']!r}  target={ex['target']!r}"
        )


if __name__ == "__main__":
    main()
