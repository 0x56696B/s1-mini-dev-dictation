#!/usr/bin/env python3
"""Review generated pairs one by one and flip `reviewed`.

  scripts/review.py data/generated/gen-20261109-131500.jsonl
  scripts/review.py FILE --source teacher                   # only rows from one source
  scripts/review.py FILE --source s1-general --accept-all   # accept a whole source after a spot check
  scripts/review.py FILE --all                              # also revisit rows that are already reviewed

Keys: a accept | e edit the clean text, then accept | r reject (deletes the row) | s skip | q quit.
Write \\n in the edit prompt for a line break. The file is saved after every decision.
"""
import argparse
import json
import sys
from collections import Counter

from common import check_row, read_jsonl, write_jsonl


def edit(text):
    """Ask for a new clean text, pre-filled with the old one when readline is available."""
    try:
        import readline
        readline.set_startup_hook(lambda: readline.insert_text(text.replace("\n", "\\n")))
    except ImportError:
        readline = None
    try:
        return input("clean> ").replace("\\n", "\n").strip()
    finally:
        if readline:
            readline.set_startup_hook()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="a jsonl file from scripts/generate.py")
    parser.add_argument("--source", help="only rows with this source (template, teacher, s1-general, ...)")
    parser.add_argument("--all", action="store_true", help="include rows that are already reviewed")
    parser.add_argument("--accept-all", action="store_true", help="accept every selected row (needs --source)")
    args = parser.parse_args()
    if args.accept_all and not args.source:
        sys.exit("--accept-all needs --source, so you never accept a whole file by accident")

    rows = read_jsonl(args.file)
    for number, row in enumerate(rows, 1):
        if problems := check_row(row):
            sys.exit(f"{args.file}:{number}: " + "; ".join(problems))
    todo = [r for r in rows if (args.all or not r["reviewed"]) and args.source in (None, r["source"])]

    done = Counter()
    if args.accept_all:
        for row in todo:
            row["reviewed"] = True
        write_jsonl(args.file, rows)
        print(f"accepted {len(todo)} {args.source} rows")
        return

    try:
        for number, row in enumerate(todo, 1):
            print(f"\n[{number}/{len(todo)}] {row['id']}  ({row['source']})\n{row['control']}")
            print(f"  raw  : {row['raw']}\n  clean: {json.dumps(row['clean'], ensure_ascii=False)}")
            key = ""
            while key not in tuple("aersq"):
                key = input("[a]ccept  [e]dit  [r]eject  [s]kip  [q]uit > ").strip().lower()
            if key == "q":
                break
            if key == "e":
                row["clean"] = edit(row["clean"])
            if key in ("a", "e"):
                row["reviewed"] = True
            if key == "r":
                rows = [r for r in rows if r is not row]
            done[{"a": "accepted", "e": "edited", "r": "rejected", "s": "skipped"}[key]] += 1
            write_jsonl(args.file, rows)
    except (EOFError, KeyboardInterrupt):
        print()
    print(f"{dict(done) or 'no changes'}; {sum(not r['reviewed'] for r in rows)} unreviewed rows left in {args.file}")


if __name__ == "__main__":
    main()
