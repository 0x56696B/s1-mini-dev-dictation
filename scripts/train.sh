#!/usr/bin/env bash
# LoRA fine-tune S1-mini with mlx-lm on build/mlx (scripts/build_dataset.py writes it).
#   scripts/train.sh                # 4 epochs
#   EPOCHS=6 scripts/train.sh       # more epochs
#   ITERS=40 scripts/train.sh       # exact iteration count (smoke runs)
# Extra arguments go to mlx_lm.lora, e.g. scripts/train.sh --steps-per-eval 20
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && export PATH="$PWD/.venv/bin:$PATH"

[ -s build/mlx/train.jsonl ] || { echo "build/mlx/train.jsonl missing: run scripts/build_dataset.py first" >&2; exit 1; }

rows=$(wc -l < build/mlx/train.jsonl | tr -d ' ')
batch=$(python -c "import yaml; print(yaml.safe_load(open('config/lora.yaml'))['batch_size'])")
iters="${ITERS:-$(( (rows * ${EPOCHS:-4} + batch - 1) / batch ))}"
echo "train rows: $rows, batch size: $batch -> iters: $iters"

exec python scripts/train.py -c config/lora.yaml --iters "$iters" "$@"
