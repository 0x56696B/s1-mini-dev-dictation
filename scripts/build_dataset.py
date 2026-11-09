#!/usr/bin/env python3
"""Build the MLX training files from reviewed pair rows.

  scripts/build_dataset.py                        # data/seed, data/generated, data/real: reviewed rows only
  scripts/build_dataset.py --include-unreviewed   # smoke runs only
  scripts/build_dataset.py --inputs data/seed/pairs.jsonl --out /tmp/mlx   # chosen inputs and output dir

Writes build/mlx/train.jsonl and valid.jsonl as {"messages": [system, user, assistant]}.
Drops rows whose id or raw text is in the held-out eval set, and duplicate (control, raw) rows.
Stops with the row id and the reason when a row it would train on breaks a rule (see check_content and the
README section "Row checks"): a number in clean that raw never says, um/uh/er in clean, bullets under
Structure: prose, or a kept leading "So" in semi-formal and formal styling.
Then renders the first row the way training does and compares it with the literal S1 format.
"""
import argparse
import glob
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from common import CONTROL_RE, ROOT, SYSTEM_PROMPT, check_row, read_jsonl, user_message, write_jsonl

DEFAULT_INPUTS = ["data/seed/pairs.jsonl", "data/generated/*.jsonl", "data/real/*.jsonl"]
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


# ---- Row checks: a row that breaks one of these rules stops the build. Keep them simple.
FILLER = re.compile(r"\b(?:[Uu]m+|[Uu]h+|[Ee]r+m?)\b")
OPENERS = (("so", "yeah"), ("okay", "so"), ("alright", "so"), ("so",))  # longest first
NOT_OPENERS = {"far", "much", "many", "long", "good", "bad", "great", "little", "few"}  # "so far", "so much", ...
LEAD_FILLERS = {"um", "uh", "er", "hmm"}
PAUSES = {"um", "uh", "er", "like", "you", "know"}  # fillers the ASR may put inside a spoken number
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)}
ORDINALS = dict(zip(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
    "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth thirtieth fortieth fiftieth".split(),
    [*range(1, 20), 20, 30, 40, 50]))
SCALES = {"hundred": 100, "thousand": 10**3, "million": 10**6, "billion": 10**9}
# Digits that clean may show for a spoken idiom without a number word of their own ("half past two" -> 2:30).
IDIOM_DIGITS = {"half past": {"30"}, "quarter past": {"15"}, "quarter to": {"45"}, "o'clock": {"0", "00"},
                "noon": {"12"}, "midnight": {"0", "00", "12"}, "dozen": {"12"}}


def word_kind(word):
    """(kind, value) of a spoken number word, or (None, None). kind: zero, unit, teen, tens, hundred, scale."""
    if word == "zero":
        return "zero", 0
    if word in SCALES:
        return ("hundred" if word == "hundred" else "scale"), SCALES[word]
    value = UNITS.get(word) or ORDINALS.get(word) or TENS.get(word)
    if value is None:
        return None, None
    return ("unit" if value < 10 else "teen" if value < 20 else "tens"), value


def cardinal(words):
    """Value of a spoken whole number ("two hundred and forty" -> 240), or None if `words` is not one."""
    total = current = 0
    prev = None  # kind of the previous word; "tens_unit" is a unit after a tens word, "and" is the connector
    for word in words:
        if word == "and":
            if prev not in ("hundred", "scale"):
                return None
            prev = "and"
            continue
        kind, value = word_kind(word)
        if kind == "zero":
            return 0 if len(words) == 1 else None
        if kind == "unit" and prev in (None, "tens", "hundred", "scale", "and"):
            current, prev = current + value, ("tens_unit" if prev == "tens" else "unit")
        elif kind in ("teen", "tens") and prev in (None, "hundred", "scale", "and"):
            current, prev = current + value, kind
        elif kind == "hundred" and prev in (None, "unit", "teen"):
            current, prev = (current or 1) * 100, "hundred"
        elif kind == "scale" and prev in (None, "unit", "teen", "tens", "tens_unit", "hundred"):
            total, current, prev = total + (current or 1) * value, 0, "scale"
        else:
            return None
    return None if prev in (None, "and") else total + current


def number_runs(raw):
    """The digit strings raw already contains, and its runs of spoken number words."""
    tokens = re.findall(r"[a-z0-9']+", raw.lower())
    digits, runs, run = set(), [], []
    for i, token in enumerate(tokens):
        following = tokens[i + 1] if i + 1 < len(tokens) else ""
        if run and token in PAUSES:  # "seven two uh six" is still 726
            continue
        if word_kind(token)[0] or (token == "oh" and (run or word_kind(following)[0])):
            run.append("zero" if token == "oh" else token)
            continue
        if token == "and" and run and run[-1] in SCALES and word_kind(following)[0]:
            run.append(token)
            continue
        if run:
            runs.append(run)
            run = []
        digits.update(re.findall(r"\d+", token))
    if run:
        runs.append(run)
    return digits, runs


def derivable(number, run):
    """True if the digit string `number` is what a stretch of the spoken-number `run` says.

    That is one whole number or several written one after the other: "two hundred and forty" -> 240,
    "four seven one two" -> 4712 (an ID said digit by digit), "twenty twenty six" -> 2026 (20, then 26).
    """
    values = {}
    for start in range(len(run)):
        todo, seen = [(start, 0)], set()
        while todo:
            i, pos = todo.pop()
            if pos == len(number):
                return True
            if (i, pos) in seen:
                continue
            seen.add((i, pos))
            for j in range(i + 1, len(run) + 1):
                if (i, j) not in values:
                    values[i, j] = cardinal(run[i:j])
                if values[i, j] is not None and number.startswith(str(values[i, j]), pos):
                    todo.append((j, pos + len(str(values[i, j]))))
    return False


def number_problems(raw, clean):
    """Every number in clean must be spoken in raw (as words or digits), so a label never changes a value."""
    digits, runs = number_runs(raw)
    for phrase, extra in IDIOM_DIGITS.items():
        if phrase in raw.lower():
            digits |= extra
    problems, seen = [], set()
    for found in re.findall(r"\d{1,3}(?:,\d{3})+|\d+", clean):
        number = found.replace(",", "")
        if number not in seen and number not in digits and not any(derivable(number, run) for run in runs):
            problems.append(f"number {number} is in clean but raw does not say it")
        seen.add(number)
    return problems


def leading_opener(text):
    """The opener a text starts with ("so", "so yeah", "okay so", "alright so"), ignoring case and commas."""
    words = re.sub(r"[^\w\s']", " ", text.lower()).split()
    while words and words[0] in LEAD_FILLERS:
        words.pop(0)
    for opener in OPENERS:
        rest = words[len(opener):][:1]
        if tuple(words[:len(opener)]) == opener and not (rest and rest[0] in NOT_OPENERS):
            return " ".join(opener)
    return None


def check_content(row):
    """Rule problems of one row, as a list of strings (empty list = the row passes)."""
    styling, structure, _ = CONTROL_RE.fullmatch(row["control"]).groups()
    clean, problems = row["clean"], []
    if match := FILLER.search(clean):
        problems.append(f"filler word {match.group()!r} in clean")
    if structure == "prose" and (clean.startswith("- ") or "\n- " in clean):
        problems.append("bullets in clean under Structure: prose")
    if styling in ("semi-formal", "formal") and leading_opener(row["raw"]) and (kept := leading_opener(clean)):
        problems.append(f"clean keeps the leading {kept!r}; {styling} drops it")
    return problems + number_problems(row["raw"], clean)


def to_messages(row):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(row["control"], row["raw"])},
            {"role": "assistant", "content": row["clean"]},
        ]
    }


def verify_format(model, row):
    """Tokenize `row` exactly as scripts/train.py will and compare with the literal S1 text."""
    from huggingface_hub import snapshot_download
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.utils import load_tokenizer
    from train import patch_tokenizer

    patch_tokenizer()
    path = model if Path(model).is_dir() else snapshot_download(
        model, allow_patterns=["*.json", "*.jinja", "merges.txt"]
    )
    tokenizer = load_tokenizer(path)
    example = to_messages(row)
    tokens, offset = ChatDataset([example], tokenizer, mask_prompt=True).process(example)
    prompt, completion = tokenizer.decode(tokens[:offset]), tokenizer.decode(tokens[offset:])

    want_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_message(row['control'], row['raw'])}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    want_completion = f"{row['clean']}<|im_end|>\n"
    print(f"format check for {row['id']}: {len(tokens)} tokens, loss on the last {len(tokens) - offset}")
    print("  masked prompt  :", repr(prompt))
    print("  loss completion:", repr(completion))
    if (prompt, completion) != (want_prompt, want_completion):
        sys.exit("FORMAT MISMATCH: the rendered row differs from the S1 literal format\n"
                 f"  want prompt    : {want_prompt!r}\n  want completion: {want_completion!r}")
    print("  matches the S1 literal format: yes")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS, help="jsonl files or globs")
    parser.add_argument("--heldout", default="data/eval/heldout.jsonl")
    parser.add_argument("--out", help="output dir (default: `data` in config/lora.yaml)")
    parser.add_argument("--model", help="base model for the format check (default: `model` in config/lora.yaml)")
    parser.add_argument("--include-unreviewed", action="store_true", help="also use reviewed=false rows")
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true", help="skip the tokenizer format check")
    parser.add_argument("--show", metavar="ID", help="row id for the format check (default: first training row)")
    args = parser.parse_args()

    os.chdir(ROOT)
    config = yaml.safe_load(Path("config/lora.yaml").read_text())
    out = Path(args.out or config["data"])
    valid_min = config["batch_size"]  # mlx-lm needs at least one full batch to validate

    files = sorted({f for pattern in args.inputs for f in glob.glob(pattern)})
    if not files:
        sys.exit(f"no input files match {args.inputs}")
    rows = []
    for path in files:
        for number, row in enumerate(read_jsonl(path), 1):
            if problems := check_row(row):
                sys.exit(f"{path}:{number}: " + "; ".join(problems))
            rows.append({**row, "raw": row["raw"].strip(), "clean": row["clean"].strip(), "where": f"{path}:{number}"})

    held = read_jsonl(args.heldout) if Path(args.heldout).exists() else []
    if not held:
        print(f"warning: no held-out rows at {args.heldout}; nothing is excluded from training", file=sys.stderr)
    held_ids = {r["id"] for r in held}
    held_raws = {r["raw"].strip() for r in held}

    usable, seen, dropped = [], set(), Counter()
    for row in rows:
        if not (row["reviewed"] or args.include_unreviewed):
            dropped["unreviewed"] += 1
        elif row["id"] in held_ids or row["raw"] in held_raws:
            dropped["held-out"] += 1
        elif (row["control"], row["raw"]) in seen:
            dropped["duplicate"] += 1
        else:
            seen.add((row["control"], row["raw"]))
            usable.append(row)

    if failures := [f"  {r['id']}  ({r['where']})  {problem}" for r in usable for problem in check_content(r)]:
        sys.exit(f"{len(failures)} row check failure(s). Fix `clean` or set reviewed=false in the file:\n"
                 + "\n".join(failures))

    if len(usable) < 2 * valid_min:
        sys.exit(f"{len(usable)} usable rows from {len(rows)}; need at least {2 * valid_min} "
                 f"(batch size {valid_min} for train and for valid). Dropped: {dict(dropped)}")

    random.Random(args.seed).shuffle(usable)
    n_valid = max(valid_min, round(len(usable) * args.valid_frac))
    valid, train = usable[:n_valid], usable[n_valid:]
    write_jsonl(out / "train.jsonl", [to_messages(r) for r in train])
    write_jsonl(out / "valid.jsonl", [to_messages(r) for r in valid])

    print(f"read {len(rows)} rows from {len(files)} file(s); dropped {dict(dropped) or 'none'}")
    print(f"sources: {dict(Counter(r['source'] for r in usable))}")
    print(f"wrote {out}/train.jsonl ({len(train)} rows) and valid.jsonl ({len(valid)} rows)")

    if not args.no_verify:
        example = train[0]
        if args.show:
            example = next((r for r in usable if r["id"] == args.show), None)
            if example is None:
                sys.exit(f"--show: no usable row with id {args.show!r}")
        verify_format(args.model or config["model"], example)


if __name__ == "__main__":
    main()
