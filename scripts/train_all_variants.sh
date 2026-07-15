#!/usr/bin/env bash
# Train the three positional-encoding variants with identical seed and budget.
# Any extra args are forwarded to train.py as Hydra overrides, e.g.:
#   bash scripts/train_all_variants.sh trainer.n_epochs=10 trainer.seed=1
set -euo pipefail
cd "$(dirname "$0")/.."

for variant in absolute nope rope; do
  echo "=================================================="
  echo "=== Training pe_variant=${variant} ==="
  echo "=================================================="
  uv run python train.py \
    model.pe_variant="${variant}" \
    writer.run_name="addition_${variant}" \
    "$@"
done

echo "Done. Checkpoints: saved/addition_{absolute,nope,rope}/model_best.pth"
