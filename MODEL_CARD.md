---
license: other
license_name: s1-mini-license
license_link: https://huggingface.co/superwhisper/s1-mini/blob/main/LICENSE
base_model: superwhisper/s1-mini
base_model_relation: finetune
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - asr
  - automatic-speech-recognition
  - text-normalization
  - inverse-text-normalization
  - punctuation
  - truecasing
  - speech-to-text
  - dictation
  - post-processing
  - qwen3
  - lora
  - software-engineering
  - devops
---

# S1-mini-dev-dictation

> This card is the README of the Hugging Face model repo. The `LICENSE` and `NOTICE` files in that repo are S1-mini's own files.

This model is **"S1-mini" by "Superwhisper"**, fine-tuned for dictating software engineering, DevOps
and AI-tooling text. It is a modified version of
[superwhisper/s1-mini](https://huggingface.co/superwhisper/s1-mini), itself a fine-tune of
[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). The input format, the output style and the
control line are unchanged: it is a drop-in replacement for S1-mini.

## Naming (licence term)

The LICENSE adds one term to Apache 2.0: any use, distribution, or integration of this model, whether
unmodified or as part of a derivative work or product, must continue to identify it by its original name,
"S1-mini" by "Superwhisper", using that exact capitalization, regardless of any other name under which the
model or a product incorporating it is marketed or distributed. "S1-mini-dev-dictation" is only the name of
this fine-tune. The model is S1-mini by Superwhisper.

## What changed

Modified from S1-mini (Apache 2.0, section 4(b) notice):

- LoRA fine-tune (rank 16, scale 4.0, dropout 0.05) on every linear layer of all 28 blocks, trained with
  [mlx-lm](https://github.com/ml-explore/mlx-lm) for 4 epochs at learning rate 1e-4, batch size 8.
  The loss covers the cleaned text only; the system prompt, control line, transcript and the empty
  `<think>` block are masked.
- The adapter is merged into the bf16 weights. Architecture, vocabulary, tokenizer, chat template and
  system prompt are the original S1-mini files.
- GGUF builds (`Q8_0`, `Q4_K_M`) are converted from the merged weights with llama.cpp.

Target behaviour added by the fine-tune (these are training targets, not measured results):

| Spoken (ASR) | Written |
|---|---|
| `so the a d o ticket needs the m c p config` | `The ADO ticket needs the MCP config.` |
| `queen three point five four b through sherpa on x` | `Qwen3.5-4B through sherpa-onnx.` |
| `tilde slash dot omp slash agent slash watchdog dot yml` | `~/.omp/agent/WATCHDOG.yml` |
| `whisper dot c p p with the d e r metric` | `whisper.cpp with the DER metric` |

A leading filler "so" is dropped. Dictated requests are cleaned, never answered.

## Training data

Pairs of raw ASR-style transcripts and cleaned text, 1,248 rows in all (1,123 for training, 125 for validation):

- 249 pairs for the failure cases above, drafted with an AI assistant from the author's dictation style;
- 564 rows from sentence templates filled from a vocabulary list (written form, spoken forms, category),
  with injected fillers and self-corrections; the template outputs were drafted the same way;
- 36 transcripts cleaned by a larger model (Qwen3.5-9B) with a fixed instruction prompt, then read row by row
  (32 labels corrected);
- 399 synthetic general non-technical transcripts cleaned by stock S1-mini and then read row by row (59
  labels corrected, 1 row dropped), to keep its general behaviour (anti-forgetting).

The rows cover all four styles (1,013 semi-formal, 81 formal, 78 casual, 76 semi-casual) and include 175
`Context: email` rows. 69 rows show a bullet list. Every row passes four checks: no number that the transcript does not
say, no `um`, `uh` or `er`, no bullets under `Structure: prose`, and no kept leading "so" in the semi-formal
and formal styles.

No private, customer or credential data is included. A held-out set was never trained on.

## Usage

Identical to S1-mini. Send the system prompt and the control line exactly, decode greedily, and turn
thinking off:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "<hf-user>/S1-mini-dev-dictation"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")

SYSTEM = (
    "You are a text normalizer for speech-to-text transcripts. The input begins "
    "with a control line specifying the styling, structure, and context settings; "
    "clean the transcript to match those settings and output only the cleaned text."
)
control = "[Styling: semi-formal] [Structure: prose] [Context: general]"
messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"{control}\nso the a d o ticket needs the m c p config"},
]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tok(text, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

llama.cpp: `llama-server -hf <hf-user>/S1-mini-dev-dictation:Q4_K_M --jinja --chat-template-kwargs '{"enable_thinking":false}' --temp 0`.

Ollama: create a Modelfile that starts the assistant turn with the empty think block, set
`PARAMETER repeat_penalty 1` and `PARAMETER temperature 0`, and send `"think": false` with every request.

## Evaluation

Held-out set: 59 cases never used in training, scored on the Q4_K_M GGUF through Ollama (greedy).

| model | cases | exact match | word accuracy | char accuracy |
|---|---|---|---|---|
| S1-mini (base) | 59 | 32.2% | 78.2% | 91.0% |
| S1-mini-dev-dictation (this model) | 59 | 84.7% | 93.8% | 97.3% |

On the ten examples in the S1-mini card (semi-formal, formal and casual, plus a list and an email) this model
matches the card output in 9 and the base Q4_K_M build in 7.

## Limitations

- English only, tuned to one speaker's dictation vocabulary. Terms outside the training vocabulary get no
  special handling: `tsconfig.json` and `parakeet-unified-en-0.6b` are not built from the spoken parts.
- `Structure: lists` and `Context: email` are not a focus of the fine-tune. It reproduces the list example
  in the S1-mini card but drops that example's leading "So", and one three-item `lists` case in the held-out
  set stayed prose.
- Known held-out failures (9 of 59): the `WATCHDOG.yml` file name comes out lowercase, `pyannote` and
  `WezTerm` get a wrong capital, a sentence after a spoken "period" can start lowercase, spoken "colon" and
  "new line" can stay words, and one email case gets an extra comma.
- Long run-on dictations: in a spot check of 15 long real dictations kept out of the training data, 4 outputs
  lost or changed words (a dropped clause, a dropped opening phrase, a rarely seen name replaced by a trained
  term). The base model kept more of the text there. Compare long outputs with the input.
- Inherits the limits of S1-mini and Qwen3-0.6B. Check the output before you rely on it.

## License

Apache License 2.0 plus the S1-mini naming term, the same terms as S1-mini. `LICENSE` and `NOTICE` are
included unchanged. S1-mini is Copyright 2026 Superwhisper and a derivative of Qwen3-0.6B, Copyright 2024
Alibaba Cloud.

## Citation

```bibtex
@misc{s1mini2026,
  title  = {S1-mini: a small text normalizer for speech-to-text output},
  author = {Superwhisper},
  year   = {2026},
  url    = {https://huggingface.co/superwhisper/s1-mini}
}

@misc{qwen3technicalreport,
  title         = {Qwen3 Technical Report},
  author        = {Qwen Team},
  year          = {2025},
  eprint        = {2505.09388},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2505.09388}
}
```
