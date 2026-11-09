#!/usr/bin/env python3
"""Score Ollama models on the held-out set. This is a report: the exit code never depends on the score.

  scripts/eval.py                                     # s1-mini-dev on data/eval/heldout.jsonl
  scripts/eval.py --baseline s1-mini-dictation        # also run stock S1-mini and compare
  scripts/eval.py --limit 10 --out /tmp/eval.jsonl    # first 10 cases, per-case results as JSONL

Per case, on the exact output text (case and punctuation count):
  exact      output equals the reference
  word acc   1 - word edit distance / reference words
  char acc   1 - character edit distance / reference characters
Every request is /api/chat with the S1 system prompt, the control line, think off, temperature 0.
"""
import argparse
import json
import os
import sys
from difflib import SequenceMatcher

from common import ROOT, SYSTEM_PROMPT, check_row, ollama_chat, read_jsonl, user_message


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def accuracy(reference, output):
    if not reference:
        return 1.0 if not output else 0.0
    return max(0.0, 1 - edit_distance(reference, output) / len(reference))


def word_diff(want, got):
    """want/got as one line: [-missing-] {+extra+}."""
    a, b = want.split(), got.split()
    parts = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(" ".join(a[i1:i2]))
            continue
        if i2 > i1:
            parts.append("[-" + " ".join(a[i1:i2]) + "-]")
        if j2 > j1:
            parts.append("{+" + " ".join(b[j1:j2]) + "+}")
    return " ".join(parts)


def run(model, rows, host):
    results = []
    for number, row in enumerate(rows, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(row["control"], row["raw"])},
        ]
        got = ollama_chat(model, messages, host)
        want = row["clean"].strip()
        results.append({
            "id": row["id"], "model": model, "got": got,
            "exact": got == want,
            "word_acc": accuracy(want.split(), got.split()),
            "char_acc": accuracy(want, got),
        })
        print(f"\r{model}: {number}/{len(rows)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="s1-mini-dev", help="Ollama model to score (default: s1-mini-dev)")
    parser.add_argument("--baseline", help="second Ollama model to score and compare, e.g. s1-mini-dictation")
    parser.add_argument("--heldout", default="data/eval/heldout.jsonl")
    parser.add_argument("--host", help="Ollama URL (default: $OLLAMA_HOST or http://localhost:11434)")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--out", help="write per-case results as JSONL")
    parser.add_argument("--summary-only", action="store_true", help="skip the per-case table and diffs")
    args = parser.parse_args()

    os.chdir(ROOT)
    rows = read_jsonl(args.heldout)[: args.limit]
    if not rows:
        sys.exit(f"no cases in {args.heldout}")
    for number, row in enumerate(rows, 1):
        if problems := check_row(row):
            sys.exit(f"{args.heldout}:{number}: " + "; ".join(problems))

    models = [args.model] + ([args.baseline] if args.baseline else [])
    results = {model: run(model, rows, args.host) for model in models}

    def mean(model, key):
        return 100 * sum(r[key] for r in results[model]) / len(rows)

    print("\n| model | cases | exact match | word accuracy | char accuracy |")
    print("|---|---|---|---|---|")
    for model in reversed(models):  # baseline first, like the table in MODEL_CARD.md
        print(f"| {model} | {len(rows)} | {mean(model, 'exact'):.1f}% | "
              f"{mean(model, 'word_acc'):.1f}% | {mean(model, 'char_acc'):.1f}% |")

    if not args.summary_only:
        width = max(len(row["id"]) for row in rows)
        col = max(len(m) for m in models)
        print(f"\n{'case'.ljust(width)}  " + "  ".join(m.ljust(col) for m in models))
        for i, row in enumerate(rows):
            marks = ["ok" if results[m][i]["exact"] else "DIFF" for m in models]
            print(f"{row['id'].ljust(width)}  " + "  ".join(mark.ljust(col) for mark in marks))
        for i, row in enumerate(rows):
            if all(results[m][i]["exact"] for m in models):
                continue
            print(f"\n{row['id']}  {row['control']}")
            print(f"  raw : {row['raw']}")
            print(f"  want: {row['clean']}")
            for m in models:
                got = results[m][i]["got"]
                print(f"  {m}: {'ok' if results[m][i]['exact'] else word_diff(row['clean'].strip(), got)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for model in models:
                for r in results[model]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
