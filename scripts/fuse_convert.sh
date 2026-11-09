#!/usr/bin/env bash
# Merge the LoRA adapter into the S1-mini weights, then convert and quantize for llama.cpp/Ollama.
#   dist/hf/                      merged bf16 model + LICENSE + NOTICE (Hugging Face upload)
#   dist/gguf/<NAME>-Q8_0.gguf    GGUF builds
#   dist/gguf/<NAME>-Q4_K_M.gguf
#   HF_REPO=<user>/S1-mini-dev-dictation fills the <hf-user> placeholder in dist/hf/README.md (the model card)
# Homebrew's llama.cpp ships no converter, so convert_hf_to_gguf.py (+ its conversion/ and gguf-py/
# helpers) comes from a 3 MB sparse checkout of llama.cpp tag $LLAMA_CPP_REF in build/llama.cpp.
# Use the tag that matches your llama-quantize build: `llama-cli --version` prints "build 10450" -> b10450.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && export PATH="$PWD/.venv/bin:$PATH"
export HF_HUB_DISABLE_PROGRESS_BARS=1

BASE="${BASE:-superwhisper/s1-mini}"
NAME="${NAME:-S1-mini-dev-dictation}"      # keep "S1-mini" in the name (licence naming clause)
LLAMA_CPP_REF="${LLAMA_CPP_REF:-b10450}"
QUANTS="${QUANTS:-Q8_0 Q4_K_M}"

[ -s adapters/adapters.safetensors ] || { echo "adapters/ is empty: run scripts/train.sh first" >&2; exit 1; }
command -v llama-quantize >/dev/null || { echo "llama-quantize not found: brew install llama.cpp" >&2; exit 1; }

# Run a noisy command with its output in a log file; show the log tail only when it fails.
quiet() {
  local log=$1; shift
  "$@" >"$log" 2>&1 || { tail -n 30 "$log" >&2; echo "failed: $* (log: $log)" >&2; exit 1; }
}

echo "== fuse adapters/ into $BASE (bf16)"
rm -rf dist/hf
mlx_lm.fuse --model "$BASE" --adapter-path adapters --save-path dist/hf

echo "== restore the base config, tokenizer and licence files (LoRA changes the weights only)"
base_dir=$(python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$BASE', allow_patterns=['*.json', '*.jinja', 'merges.txt', 'LICENSE', 'NOTICE']))")
for f in LICENSE NOTICE config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt chat_template.jinja; do
  cp "$base_dir/$f" dist/hf/
done
rm -f dist/hf/model.safetensors.index.json     # single-file model: the index mlx-lm writes is not needed
# dist/hf/README.md is the model card (the converter reads licence/base model from its front matter).
if [ -n "${HF_REPO:-}" ]; then
  case "$HF_REPO" in */*) ;; *) echo "HF_REPO must look like <user>/S1-mini-dev-dictation" >&2; exit 1 ;; esac
  repo=$(printf '%s' "$HF_REPO" | sed 's/[&|\\]/\\&/g')
  LC_ALL=C sed "s|<hf-user>/S1-mini-dev-dictation|$repo|g" MODEL_CARD.md > dist/hf/README.md
else
  cp MODEL_CARD.md dist/hf/README.md
  echo "note: dist/hf/README.md keeps the <hf-user> placeholder; set HF_REPO=<user>/S1-mini-dev-dictation or edit it before an upload" >&2
fi

echo "== convert to GGUF (llama.cpp $LLAMA_CPP_REF)"
if [ "$(git -C build/llama.cpp describe --tags 2>/dev/null || true)" != "$LLAMA_CPP_REF" ]; then
  rm -rf build/llama.cpp
  git clone --quiet --depth 1 --branch "$LLAMA_CPP_REF" --filter=blob:none --sparse \
    https://github.com/ggml-org/llama.cpp build/llama.cpp 2>/dev/null
  git -C build/llama.cpp sparse-checkout set conversion gguf-py
fi
rm -rf build/gguf dist/gguf
mkdir -p build/gguf dist/gguf
quiet build/gguf/convert.log python build/llama.cpp/convert_hf_to_gguf.py dist/hf \
  --outtype bf16 --model-name "$NAME" --metadata config/gguf-metadata.json \
  --outfile "build/gguf/$NAME-bf16.gguf"

for quant in $QUANTS; do
  echo "== quantize $quant"
  quiet "build/gguf/quantize-$quant.log" llama-quantize "build/gguf/$NAME-bf16.gguf" "dist/gguf/$NAME-$quant.gguf" "$quant"
done
ls -lh dist/gguf
