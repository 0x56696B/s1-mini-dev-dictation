#!/usr/bin/env python3
"""Generate training pairs from data/vocab.yaml and data/templates.yaml.

  scripts/generate.py --dry-run -n 20        # render a few rows, no Ollama calls, nothing written
  scripts/generate.py -n 400 --general 150   # render, label, write data/generated/gen-<time>.jsonl

Row sources
  template    raw is rendered from a template that has `clean`; the template labels the row
  teacher     raw is rendered from a template without `clean`; Ollama `qwen3.5:9b` labels it with prompts/teacher.md
  s1-general  --general N: plain transcripts from `general:` with random control lines, labeled by stock
              S1-mini (`s1-mini-dictation` in Ollama) so the fine-tune keeps its general behaviour
Every row gets reviewed=false (template rows: unless the template says `reviewed: true`).
Review them with scripts/review.py; scripts/build_dataset.py trains on reviewed rows only.

data/vocab.yaml
  terms:
    - written: "ADO"                    # canonical output form (quote it: YAML reads ON/NO/Y as booleans)
      spoken: ["a d o", "ay dee oh"]    # how ASR writes it, lowercase; one is picked per row
      category: acronym                 # free text; selects terms for {term:acronym} slots

data/templates.yaml
  templates:
    - id: ado-ticket-1
      control: "[Styling: semi-formal] [Structure: prose] [Context: general]"   # optional, this is the default
      raw: "so um the {term} ticket needs updating"   # lowercase, no punctuation; {term} or {term:category}
      clean: "The {term} ticket needs updating."      # optional; same slots, filled with `written`
      reviewed: true                                  # optional; trust this template's labels
  general:
    - "so um i need to send the report by friday no wait thursday"   # plain transcript, no slots

Slots: every slot in `raw` is an independent draw (least-used term first). The n-th `{term:cat}` in `clean`
is the n-th `{term:cat}` in `raw`, so both need the same slots. Fillers and self-corrections ("a d o no wait
m c p": the clean text keeps only the last term) are added to `raw` by this script, not by the templates.
Rows without `clean` need the default control line: the teacher prompt does not know the others.

Seed: every run draws new rows (the seed comes from the clock and is printed). `--seed N` repeats a draw as
long as data/seed, data/generated and the held-out file have not changed, because known rows are skipped.
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from hashlib import sha1
from pathlib import Path

import yaml

from common import CONTROL_RE, DEFAULT_CONTROL, ROOT, SYSTEM_PROMPT, ollama_chat, read_jsonl, user_message

SLOT = re.compile(r"\{term(?::([^{}]+))?\}")
FILLERS = ["um", "uh", "er", "um", "uh", "you know", "like"]
LEADING = ["so", "so um", "um", "uh", "um so"]
OPENERS = {"so", "um", "uh", "er", "okay", "well", "yeah", "hmm", "oh"}  # raw already starts like this: add nothing
CORRECTION_MARKERS = ["no wait", "wait no", "i mean", "i meant", "sorry i meant", "scratch that", "no sorry"]
STYLINGS = ["casual", "semi-casual", "semi-formal", "formal"]


def category(match):
    return (match.group(1) or "").strip().lower() or None


def clean_raw(text):
    """ASR-like text: lowercase, no punctuation except apostrophes."""
    return " ".join(re.sub(r"[^\w\s']", "", text.lower()).split())


def load_data(vocab_path, templates_path):
    vocab = yaml.safe_load(Path(vocab_path).read_text(encoding="utf-8")) or {}
    spec = yaml.safe_load(Path(templates_path).read_text(encoding="utf-8")) or {}
    errors, terms, templates = [], [], []

    for n, t in enumerate(vocab.get("terms") or [], 1):
        spoken = t.get("spoken")
        if not isinstance(t.get("written"), str) or not isinstance(spoken, list) or not spoken \
                or not all(isinstance(s, str) for s in spoken):
            errors.append(f"{vocab_path} term {n} ({t.get('written')!r}): `written` and a `spoken` list of strings are required")
            continue
        terms.append({"written": t["written"], "spoken": [clean_raw(s) for s in spoken],
                      "category": str(t.get("category") or "").strip().lower()})
    categories = {t["category"] for t in terms}

    for n, t in enumerate(spec.get("templates") or [], 1):
        name = t.get("id") or f"template-{n}"
        control = t.get("control", DEFAULT_CONTROL)
        raw, clean = t.get("raw"), t.get("clean")
        if not isinstance(raw, str) or not raw.strip():
            errors.append(f"{name}: `raw` is required")
            continue
        if not CONTROL_RE.fullmatch(control):
            errors.append(f"{name}: bad control line {control!r}")
        if clean is None and control != DEFAULT_CONTROL:
            errors.append(f"{name}: no `clean`, so the teacher labels it; that needs the default control line")
        raw_slots = Counter(category(m) for m in SLOT.finditer(raw))
        if clean is not None and raw_slots != Counter(category(m) for m in SLOT.finditer(clean)):
            errors.append(f"{name}: `raw` and `clean` need the same slots")
        errors += [f"{name}: no vocab term has category {c!r}" for c in raw_slots if c and c not in categories]
        if not terms and raw_slots:
            errors.append(f"{name}: has slots but {vocab_path} has no usable terms")
        if re.search(r"\b(a|an) \{term", raw + " " + (clean or "")):
            print(f"warning: {name}: a/an before a slot cannot agree with every term", file=sys.stderr)
        templates.append({"id": name, "control": control, "raw": raw, "clean": clean,
                          "reviewed": bool(t.get("reviewed", False))})

    general = [clean_raw(s) for s in spec.get("general") or [] if isinstance(s, str) and s.strip()]
    if any(SLOT.search(s) for s in spec.get("general") or []):
        errors.append("`general` entries are plain transcripts: no {term} slots")
    if errors:
        sys.exit("\n".join(errors))
    return terms, templates, general


class Sampler:
    """Picks vocab terms least-used first, so every term shows up about equally often."""

    def __init__(self, terms, rng):
        self.terms, self.rng, self.uses = terms, rng, Counter()

    def pick(self, cat, exclude=()):
        pool = [t for t in self.terms if cat in (None, t["category"])]
        fresh = [t for t in pool if t["written"] not in exclude] or pool
        term = min(fresh, key=lambda t: (self.uses[t["written"]], self.rng.random()))
        spoken = min(term["spoken"], key=lambda s: (self.uses[term["written"], s], self.rng.random()))
        self.uses[term["written"]] += 1
        self.uses[term["written"], spoken] += 1
        return term, spoken

    def decoy(self, term):
        """A spoken form of another term (same category if possible), for a self-correction."""
        others = [t for t in self.terms if t is not term]
        same = [t for t in others if t["category"] == term["category"]]
        return self.rng.choice(self.rng.choice(same or others or [term])["spoken"])


def add_fillers(tokens, rng, prob):
    if rng.random() < prob:
        for _ in range(rng.choice([1, 1, 2])):
            tokens.insert(rng.randint(0, len(tokens)), rng.choice(FILLERS))
    if tokens and rng.random() < prob / 2 and not OPENERS & set(tokens[:2]):
        tokens.insert(0, rng.choice(LEADING))


def render(template, sampler, rng, args):
    """One (raw, clean) pair from a template; clean is None when the teacher has to label it."""
    draws, used = [], set()  # (category, term, spoken) in raw order
    for m in SLOT.finditer(template["raw"]):
        term, spoken = sampler.pick(category(m), used)
        used.add(term["written"])
        draws.append((category(m), term, spoken))

    spoken = [s for _, _, s in draws]
    if draws and rng.random() < args.correction_prob:
        k = rng.randrange(len(draws))
        spoken[k] = f"{sampler.decoy(draws[k][1])} {rng.choice(CORRECTION_MARKERS)} {spoken[k]}"
    spoken_iter = iter(spoken)
    tokens = [SLOT.sub(lambda _: next(spoken_iter), token) for token in template["raw"].split()]
    add_fillers(tokens, rng, args.filler_prob)
    raw = clean_raw(" ".join(tokens))

    if template["clean"] is None:
        return raw, None
    written = {c: iter([t["written"] for cat, t, _ in draws if cat == c]) for c, _, _ in draws}
    return raw, SLOT.sub(lambda m: next(written[category(m)]), template["clean"]).strip()


def random_control(rng):
    if rng.random() < 0.4:
        return DEFAULT_CONTROL
    return (f"[Styling: {rng.choice(STYLINGS)}] [Structure: {rng.choice(['prose', 'lists'])}] "
            f"[Context: {rng.choice(['general', 'email'])}]")


def make_row(source, tag, control, raw, clean=None, reviewed=False):
    digest = sha1(f"{control}\n{raw}".encode()).hexdigest()[:10]
    return {"id": f"{source}-{tag}-{digest}" if tag else f"{source}-{digest}", "control": control, "raw": raw,
            "clean": clean, "source": source, "reviewed": reviewed}


def template_rows(templates, sampler, rng, args, seen):
    rows, queue, attempts = [], [], 0
    while templates and len(rows) < args.count and attempts < 50 * args.count:
        attempts += 1
        queue = queue or rng.sample(templates, len(templates))
        template = queue.pop()
        raw, clean = render(template, sampler, rng, args)
        if (template["control"], raw) in seen:
            continue
        seen.add((template["control"], raw))
        source = "template" if clean is not None else "teacher"
        rows.append(make_row(source, template["id"], template["control"], raw, clean,
                             reviewed=template["reviewed"] and clean is not None))
    if len(rows) < args.count:
        print(f"note: only {len(rows)} of {args.count} template rows are new after {attempts} draws (the rest "
              "repeat rows from seed, generated or held-out data, or from this run); add templates or vocab, "
              "or lower -n", file=sys.stderr)
    return rows


def general_rows(general, rng, args, seen):
    rows, queue, attempts = [], [], 0
    while general and len(rows) < args.general and attempts < 50 * args.general:
        attempts += 1
        queue = queue or rng.sample(general, len(general))
        raw, control = queue.pop(), random_control(rng)
        if (control, raw) not in seen:
            seen.add((control, raw))
            rows.append(make_row("s1-general", "", control, raw))
    if len(rows) < args.general:
        print(f"note: only {len(rows)} general rows possible; add transcripts to `general`", file=sys.stderr)
    return rows


def known_pairs(heldout):
    """(control, raw) of every row in seed, generated or held-out data: never generate those again."""
    files = [Path("data/seed/pairs.jsonl"), Path(heldout), *sorted(Path("data/generated").glob("*.jsonl"))]
    return {(r["control"], r["raw"]) for f in files if f.exists() for r in read_jsonl(f)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--count", type=int, default=200, help="rows to render from templates (default 200)")
    parser.add_argument("--general", type=int, default=0, help="extra anti-forgetting rows labeled by stock S1-mini")
    parser.add_argument("--seed", type=int, help="random seed (default: from the clock; printed, so --seed repeats a run)")
    parser.add_argument("--filler-prob", type=float, default=0.6, help="chance a row gets um/uh/like (default 0.6)")
    parser.add_argument("--correction-prob", type=float, default=0.12, help="chance a slot gets a self-correction")
    parser.add_argument("--vocab", default="data/vocab.yaml")
    parser.add_argument("--templates", default="data/templates.yaml")
    parser.add_argument("--heldout", default="data/eval/heldout.jsonl", help="never generate these inputs")
    parser.add_argument("--out", help="output file (default data/generated/gen-<time>.jsonl); never overwritten")
    parser.add_argument("--teacher", default="qwen3.5:9b", help="Ollama model for template rows without `clean`")
    parser.add_argument("--s1-model", default="s1-mini-dictation", help="stock S1-mini in Ollama, for --general")
    parser.add_argument("--agent-name", default="Assistant", help="fills {{agentName}} in prompts/teacher.md")
    parser.add_argument("--host", help="Ollama URL (default: $OLLAMA_HOST or http://localhost:11434)")
    parser.add_argument("--dry-run", action="store_true", help="print the rendered rows, call nothing, write nothing")
    args = parser.parse_args()

    os.chdir(ROOT)
    terms, templates, general = load_data(args.vocab, args.templates)
    seed = args.seed if args.seed is not None else time.time_ns() & 0x7FFFFFFF
    print(f"seed: {seed}", file=sys.stderr)
    rng = random.Random(seed)
    seen = known_pairs(args.heldout)
    rows = template_rows(templates, Sampler(terms, rng), rng, args, seen) + general_rows(general, rng, args, seen)

    if args.dry_run:
        for r in rows:
            clean = r["clean"] if r["clean"] is not None else "(labeled by Ollama when not a dry run)"
            print(f"{r['source']}  {r['control']}\n  raw  : {r['raw']}\n  clean: {clean}")
        print(f"\n{len(rows)} rows: {dict(Counter(r['source'] for r in rows))}")
        return

    out = Path(args.out or f"data/generated/gen-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    if out.exists():
        sys.exit(f"{out} exists: pick another --out (reviewed rows live in these files)")
    out.parent.mkdir(parents=True, exist_ok=True)
    teacher_prompt = Path("prompts/teacher.md").read_text(encoding="utf-8").replace("{{agentName}}", args.agent_name)

    kept = Counter()
    with open(out, "w", encoding="utf-8") as f:
        for number, row in enumerate(rows, 1):
            print(f"\r[{number}/{len(rows)}] {row['source']}", end="", file=sys.stderr, flush=True)
            if row["source"] == "teacher":
                prompt = teacher_prompt.replace("{{TRANSCRIPT}}", row["raw"])
                row["clean"] = ollama_chat(args.teacher, [{"role": "user", "content": prompt}], args.host)
            elif row["source"] == "s1-general":
                row["clean"] = ollama_chat(args.s1_model, [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(row["control"], row["raw"])}], args.host)
            if not row["clean"] and row["source"] != "template":
                print(f"\nskipped {row['id']}: model returned nothing for {row['raw']!r}", file=sys.stderr)
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            kept[row["source"]] += 1
    print(f"\nwrote {out}: {dict(kept)}\nnext: scripts/review.py {out}")


if __name__ == "__main__":
    main()
