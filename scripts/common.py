"""Shared helpers: repo paths, the S1 input format, JSONL I/O and a small Ollama client."""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Fixed by S1-mini: changing one character of the system prompt or control line degrades it.
SYSTEM_PROMPT = (ROOT / "prompts" / "system-s1.txt").read_text(encoding="utf-8").rstrip("\n")
DEFAULT_CONTROL = "[Styling: semi-formal] [Structure: prose] [Context: general]"
CONTROL_RE = re.compile(
    r"\[Styling: (casual|semi-casual|semi-formal|formal)\] "
    r"\[Structure: (prose|lists)\] \[Context: (general|email)\]"
)

SOURCES = ("seed", "template", "teacher", "s1-general", "real")
ROW_KEYS = ("id", "control", "raw", "clean", "source", "reviewed")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Write via a temp file and rename, so an interrupted review never truncates a data file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def user_message(control, raw):
    """The user turn S1-mini expects: control line, newline, raw transcript."""
    return f"{control}\n{raw}"


def check_row(row):
    """Return a list of schema problems for one pair row (empty list = valid)."""
    problems = [f"missing key '{k}'" for k in ROW_KEYS if k not in row]
    if problems:
        return problems
    if not CONTROL_RE.fullmatch(row["control"]):
        problems.append(f"bad control line {row['control']!r}")
    if not isinstance(row["raw"], str) or not row["raw"].strip():
        problems.append("raw must be a non-empty string")
    if not isinstance(row["clean"], str):
        problems.append("clean must be a string (empty string = filler-only input)")
    if row["source"] not in SOURCES:
        problems.append(f"source {row['source']!r} not in {SOURCES}")
    if not isinstance(row["reviewed"], bool):
        problems.append("reviewed must be true or false")
    return problems


def ollama_chat(model, messages, host=None, timeout=600):
    """One greedy, non-streaming /api/chat call with thinking off. Returns the reply text."""
    base = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
    if "://" not in base:
        base = "http://" + base
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,  # S1-mini returns nothing usable with thinking on
        # Greedy, no penalties: qwen3.5 ships presence_penalty 1.5, which punishes copying input words.
        "options": {"temperature": 0, "num_ctx": 4096, "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1},
    }
    request = urllib.request.Request(
        base.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ollama {e.code} for model {model!r}: {e.read().decode()[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"ollama not reachable at {base}: {e.reason}")
