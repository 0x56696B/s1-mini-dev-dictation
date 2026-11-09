# S1-mini-dev-dictation: fine-tuning repo

Fine-tunes [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) (a 0.6B Qwen3 text normalizer
for speech-to-text) on your own dictation: software engineering, DevOps and AI-tooling vocabulary. It runs
locally on Apple Silicon (MLX LoRA), converts the result to GGUF, and serves it through Ollama.

```
 vocab.yaml + templates.yaml ─┐
 hand-written seed pairs ─────┼─▶ generate ─▶ review ─▶ build_dataset ─▶ train ─▶ fuse_convert ─▶ ollama create ─▶ eval
 stock S1-mini (general rows) ┘      (Ollama labels)  (you)   (MLX chat format)  (LoRA)   (bf16 + GGUF)
```

## Layout

```
prompts/system-s1.txt     the fixed S1-mini system prompt (verbatim, no trailing newline)
prompts/teacher.md        your cleanup prompt for the labeling model; {{TRANSCRIPT}} is replaced per row
config/lora.yaml          mlx_lm.lora settings (rank 16, 4 epochs, lr 1e-4, prompt masked)
data/vocab.yaml           terms: written form, spoken forms, category
data/templates.yaml       sentence templates (+ plain "general" transcripts)
data/seed/pairs.jsonl     hand-written gold pairs
data/eval/heldout.jsonl   held-out cases: never trained on, used by scripts/eval.py
data/real/*.jsonl         your own transcripts, optional (git-ignored: real dictations must not go public)
data/generated/           scripts/generate.py output; commit it after review
scripts/                  generate.py review.py build_dataset.py train.sh train.py fuse_convert.sh
                          eval.py common.py (shared helpers)
ollama/Modelfile          Ollama definition of the fine-tune
MODEL_CARD.md             Hugging Face model card (README.md of the model repo)
THIRD_PARTY_NOTICES.md    MIT notice for prompts/teacher.md (adapted from OpenWhispr)
build/ adapters/ dist/    generated, git-ignored
```

## Prerequisites

- Apple Silicon Mac with macOS. Tested on an M3 Pro with 18 GB.
- [`uv`](https://docs.astral.sh/uv/) (or Python 3.12 with `venv`). Use Python 3.12: MLX wheels exist for
  3.10 to 3.14, not for 3.15, which is what a plain `python3` may be on Homebrew.
- [Ollama](https://ollama.com) running (tested with 0.40.14) and these models:
  ```bash
  ollama pull qwen3.5:9b                                  # teacher that labels template rows without `clean`
  ollama create s1-mini-dictation -f ollama/Modelfile.stock  # stock S1-mini: baseline + labeler for general rows
  ```
- `brew install llama.cpp` for `llama-quantize`. Homebrew ships no converter, so
  `scripts/fuse_convert.sh` fetches `convert_hf_to_gguf.py` (see "GGUF conversion path").
- `git`. No Hugging Face account or token is needed: the base model downloads anonymously.
- About 1.2 GB for the base model in the Hugging Face cache and 3 GB for `dist/`.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt        # no uv: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate                 # the .sh scripts add .venv/bin to PATH themselves
```

The base model downloads into `~/.cache/huggingface` the first time a script needs it.

## Data

Every data file is JSONL with one pair per line:

```json
{"id": "seed-0001", "control": "[Styling: semi-formal] [Structure: prose] [Context: general]",
 "raw": "so the a d o ticket needs updating", "clean": "The ADO ticket needs updating.",
 "source": "seed", "reviewed": true}
```

- `control`: `[Styling: casual|semi-casual|semi-formal|formal] [Structure: prose|lists] [Context: general|email]`.
- `raw`: lowercase, unpunctuated, ASR-like. `clean`: the target; an empty string means "filler only, output nothing".
- `source`: `seed` (hand-written), `template` (filled by a template that has `clean`), `teacher` (labeled by
  `qwen3.5:9b`), `s1-general` (labeled by stock S1-mini), `real` (your own transcripts: one file or more in
  `data/real/*.jsonl`, same row format, `"source": "real"`, `"reviewed": true` once you checked the label).
  `data/real/` is git-ignored because real dictations must not go public. A model trained on them can repeat
  them, so publish only weights trained without that folder.
- `reviewed`: only `true` rows are trained on. `scripts/review.py` flips it.

`scripts/generate.py` reads two YAML files:

```yaml
# data/vocab.yaml
terms:
  - written: "ADO"                    # canonical output form (quote it: YAML reads ON/NO/Y as booleans)
    spoken: ["a d o", "ay dee oh"]    # raw ASR forms, lowercase; one is picked per row
    category: acronym                 # free text

# data/templates.yaml
templates:
  - id: "ado-ticket-1"
    control: "[Styling: semi-formal] [Structure: prose] [Context: general]"   # optional (this is the default)
    raw: "so um the {term} ticket needs updating"   # lowercase, no punctuation; {term} or {term:category}
    clean: "The {term} ticket needs updating."      # optional: same slots, filled with `written`
    reviewed: true                                  # optional: trust this template's labels
general:                                            # plain non-technical transcripts, no slots, no clean
  - "so um i need to send the report by friday no wait thursday"
```

Rules the generator follows:

- Each slot in `raw` is an independent draw, least-used term first, so every term appears about equally often.
  A `{term:acronym}` slot only draws terms with that category. The n-th `{term:cat}` in `clean` is the n-th in
  `raw`, so both need the same slots. A slot is filled with a random `spoken` form in `raw` and with `written`
  in `clean`.
- Template with `clean`: row has `source: template` and no model is called. `reviewed` is true only if the
  template says so. Template without `clean`: `qwen3.5:9b` labels it (`source: teacher`, `reviewed: false`).
  Those rows need the default control line, because the teacher prompt knows only one style.
- The script adds ASR noise to `raw` (never to `clean`): fillers (`um`, `uh`, `er`, `you know`, `like`), a
  leading `so`, and self-corrections at a slot (`a d o no wait m c p`: the clean text keeps only the last term).
  Tune with `--filler-prob` and `--correction-prob`.
- `--general N` adds N rows from the `general` list with random control lines, labeled by stock S1-mini. They
  keep the fine-tune from forgetting the other styles (anti-forgetting).
- Rows already in `data/seed`, `data/generated` or the held-out file are never generated again.

## Workflow

### 1. Generate

```bash
scripts/generate.py --dry-run -n 20                # print rendered rows; calls nothing, writes nothing
scripts/generate.py -n 600 --general 400           # writes data/generated/gen-<time>.jsonl (never overwrites)
```

Teacher calls take about 2.5 s each and only happen for templates without `clean`. Both Ollama calls use
`think: false`, temperature 0 and no repetition or presence penalty. The `qwen3.5:9b` model ships
`presence_penalty 1.5`; with it the teacher changed the case of tool names and added backticks in 2 of 4 test
rows, so the script sets the penalty to 0. Every run draws new rows: the seed comes from the clock and is printed
(`seed: N`), and `--seed N` repeats a draw.

### 2. Review

```bash
scripts/review.py data/generated/gen-<time>.jsonl            # a accept, e edit, r reject, s skip, q quit
scripts/review.py FILE --source s1-general --accept-all      # accepts a whole source: read every row first
```

Teacher labels need the most attention: check the capitalization of tool names, spoken paths, and any added or
dropped words. Stock S1-mini labels need a full read too: in the last run 60 of 400 needed a fix (a kept `um`, bullets
under `Structure: prose`, a changed amount, a kept leading `So`, a formal contraction, a dropped word).
`build_dataset.py` ignores rows that are not reviewed. The file is saved after each decision.

### 3. Build the training files

```bash
scripts/build_dataset.py
```

Reads `data/seed/pairs.jsonl`, `data/generated/*.jsonl` and `data/real/*.jsonl`, keeps `reviewed: true` rows,
removes rows whose id or raw text is in `data/eval/heldout.jsonl`, removes duplicates, checks the rows (below),
and writes `build/mlx/train.jsonl` and `valid.jsonl` (10 %, at least one batch). Then it renders one row through
the real tokenizer and stops with an error if the text differs from the S1 literal format (see "Training
format"). `--include-unreviewed` is for smoke runs only.

Row checks (`check_content` in `scripts/build_dataset.py`): when a row it would train on breaks one of these
rules, the build stops and prints `id (file:line) reason` for every such row. Fix `clean` in the file, or set
`reviewed` to false.

- Numbers: every number in `clean` is one that `raw` says. Spoken numbers become digits (`two hundred and forty`
  is 240, `twenty twenty six` is 2026, `four seven one two` is 4712 for an ID said digit by digit, `half past
  two` allows 30), and any other digit string in `clean` fails. This catches `$245` for `two hundred and forty
  dollars and fifty cents`, and numbered lists (`1.`, `2.`), because raw never says those digits.
- Fillers: no standalone `um`, `uh` or `er` in `clean`, in any styling.
- Bullets: no `- ` bullet line in `clean` when the control line says `Structure: prose`.
- Opener: with `Styling: semi-formal` or `formal`, `clean` does not start with `So`, `So yeah`, `Okay so` or
  `Alright so` when `raw` starts with it. Leading um/uh are ignored, and `so far` or `so much` pass. The check
  skips casual and semi-casual.

### 4. Train

```bash
scripts/train.sh                  # 4 epochs: iters = 4 x train rows / batch size
EPOCHS=6 scripts/train.sh         # or ITERS=40 scripts/train.sh for a smoke run
```

It prints `Val loss` every 50 iterations and `Train loss` every 10 (loss per completion token). On the current
data (1,123 training rows, batch 8, 562 iterations) the first validation loss was 0.25 and the lowest 0.049
(iterations 300 and 500); training used 7.7 GB, ran at about 1.1 it/s and took about 10 minutes.
Adapters are saved to `adapters/` every 100 iterations (`0000100_adapters.safetensors` and so on).
If validation loss rises while train loss keeps falling, the model is overfitting: lower `EPOCHS` or
`learning_rate`, or copy an earlier checkpoint over `adapters/adapters.safetensors` before step 5.

### 5. Merge, convert, quantize

```bash
scripts/fuse_convert.sh           # about 20 s
```

Produces `dist/hf/` (merged bf16 model with the base LICENSE and NOTICE) and
`dist/gguf/S1-mini-dev-dictation-Q8_0.gguf` plus `...-Q4_K_M.gguf`.

`dist/hf/README.md` is `MODEL_CARD.md` with the `<hf-user>/S1-mini-dev-dictation` placeholder replaced by
`$HF_REPO` when you set it: `HF_REPO=<user>/S1-mini-dev-dictation scripts/fuse_convert.sh`. Without `HF_REPO`
the placeholder stays, so edit `dist/hf/README.md` before a manual upload.

### 6. Run it in Ollama

```bash
ollama create s1-mini-dev -f ollama/Modelfile
curl -s localhost:11434/api/chat -d '{
  "model": "s1-mini-dev", "stream": false, "think": false,
  "messages": [
    {"role": "system", "content": "You are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text."},
    {"role": "user", "content": "[Styling: semi-formal] [Structure: prose] [Context: general]\nso the a d o ticket needs the omp config updated"}]}'
```

Send `"think": false` (or `"reasoning_effort": "none"` on `/v1`) with every request. The Modelfile uses the
Q4_K_M build; change `FROM` to the Q8_0 file if your eval shows Q4_K_M losing accuracy (on the smoke run
Q4_K_M differed from the bf16 weights on 2 of 5 cases, while Q8_0 matched them on all 5).

### 7. Evaluate

```bash
scripts/eval.py --baseline s1-mini-dictation       # data/eval/heldout.jsonl through both models
```

It prints a Markdown table (exact match, word accuracy, character accuracy), a per-case table and a word
diff for every miss. It is a report: a low score never changes the exit code. Replace the table in the
"Evaluation" section of `MODEL_CARD.md` with the one it prints.

## Training format (checked, not assumed)

S1-mini was trained on this exact text, and any deviation degrades it:

```
<|im_start|>system\n{system prompt}<|im_end|>\n
<|im_start|>user\n{control line}\n{raw transcript}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n{clean text}<|im_end|>\n
```

`build/mlx/*.jsonl` uses mlx-lm's chat format, `{"messages": [system, user, assistant]}`, and `--mask-prompt`.
`scripts/build_dataset.py` prints the proof on every build (`--show seed-0001` picks the row). Real output:

```
format check for seed-0001: 109 tokens, loss on the last 16
  masked prompt  : '<|im_start|>system\nYou are a text normalizer for speech-to-text transcripts. The input begins with a control line specifying the styling, structure, and context settings; clean the transcript to match those settings and output only the cleaned text.<|im_end|>\n<|im_start|>user\n[Styling: semi-formal] [Structure: prose] [Context: general]\ni switched to sherpa on x because it runs on the c p u<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  loss completion: 'I switched to sherpa-onnx because it runs on the CPU.<|im_end|>\n'
  matches the S1 literal format: yes
```

The masked prompt equals the text transformers builds with `apply_chat_template(..., enable_thinking=False)`
(checked string for string), and the loss covers only what follows it.

What was found in mlx-lm 0.31.3 and why `scripts/train.py` exists (it is `mlx_lm.lora` plus two small patches):

1. **The chat path enables thinking.** The Qwen3 template already writes the empty `<think>` block for the last
   assistant turn, so the full text is right. But mlx-lm passes `enable_thinking=True` for any tokenizer that
   has `<think>` tokens, so the prompt/completion boundary lands before the empty think block and the loss would
   cover it. The pre-rendered prompt/completion fallback does not work either: mlx-lm's `prompt`/`completion`
   rows go through the chat template again. `train.py` sets `has_thinking` to false, which moves the boundary
   behind the block.
2. **The loss also scores one padding token.** `default_loss` uses `steps <= length`, so the first pad token
   (id 0, `!`) after each row is a target. Checked on a batch: the targets were `...<|im_end|>\n!`. The fine-tuned
   smoke model then answered `!!!!...` to every request without `think: false`. With `steps < length`
   (`completion_loss` in `train.py`) the same smoke run answers normally in all four request variants, and the
   first validation loss drops from 1.07 to 0.30: the pad target was one of about 12 targets per row, which
   accounts for the gap. The loss now covers exactly the cleaned text, `<|im_end|>` and the newline after it.

## GGUF conversion path

`mlx_lm.fuse` (bf16, no re-quantization) writes `dist/hf`. The script then copies the base `config.json`,
`generation_config.json`, tokenizer files, chat template, `LICENSE` and `NOTICE` over mlx-lm's rewritten
copies, so the repo differs from S1-mini only in the weights (and the README).

`convert_hf_to_gguf.py` is not in Homebrew's llama.cpp, and at tag `b10450` it is a thin script that imports a
`conversion/` package and `gguf-py/`. `scripts/fuse_convert.sh` therefore does a partial, sparse checkout of the
tag that matches the installed build (`llama-cli --version` prints `build 10450`) into `build/llama.cpp`
(3 MB, a few seconds), runs the converter with the venv's Python (it imports torch, numpy and transformers; it
loads `gguf` from the checkout's `gguf-py/`, and `gguf==0.19.0` in `requirements.txt` is the same version and
installs its dependencies) to make a bf16 GGUF, and quantizes it with the Homebrew `llama-quantize` to Q8_0
and Q4_K_M. Use another build with `LLAMA_CPP_REF=b1234 scripts/fuse_convert.sh`.

S1-mini stores `lm_head.weight` as a copy of the embedding (checked: identical). mlx-lm drops it for tied
models, so the GGUF has 310 tensors with tied embeddings and is smaller than the stock one (about 378 MB
against 484 MB for Q4_K_M).

## Licence

The code and data in this repository are licensed under Apache 2.0 (`LICENSE`).

`prompts/teacher.md` is adapted from OpenWhispr's cleanup prompt (MIT, Copyright (c) 2024 OpenWhispr Team);
see `THIRD_PARTY_NOTICES.md` for the notice.

The fine-tuned model is a derivative of S1-mini, which is Apache 2.0 plus one additional term. The model
inherits both:

- Keep S1-mini's `LICENSE` and `NOTICE` unchanged in every redistribution of the weights.
  `fuse_convert.sh` copies them into `dist/hf`.
- State your changes. `MODEL_CARD.md` has a "What changed" section (LoRA recipe, data, merge, GGUF).
  Keep it true if you change `config/lora.yaml`.
- Naming term: any use or distribution must continue to identify the model as **"S1-mini" by "Superwhisper"**
  with that exact capitalization, whatever other name you give it. Local Ollama tags are lowercase by Ollama
  convention; use the exact capitalization in anything public.
- Training data goes public with the model: no secrets, hostnames, internal URLs, customer details or real
  colleague names. Public product and company names are fine. The teacher (`qwen3.5:9b`) is Apache 2.0.

## Troubleshooting

- **Empty output from Ollama**: the request lacks `"think": false`.
- **`Dataset must have at least batch_size=8 examples`**: too few reviewed rows. `build_dataset.py` needs at
  least twice `batch_size` rows (train and validation) and says so; lower `batch_size` in `config/lora.yaml`
  or review more rows.
- **`convert_hf_to_gguf.py` fails after an upgrade**: set `LLAMA_CPP_REF` to the tag of your llama.cpp build.
- **`No module named yaml` or `mlx_lm`**: activate the venv (`source .venv/bin/activate`).
