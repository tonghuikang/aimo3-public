"""Generate runs.jsonl — one record per run/question with table-view metadata.

Each record carries the info runs.html needs for its default table: run timestamp,
question id, final answer, findings count, time taken, and per-solver rows
(index, tokens, tool calls, answer, answer history from \\boxed{} occurrences,
backtrack flag). Full solution content and full findings stay on disk and are
fetched lazily when a row is clicked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"
OUTPUT = Path(__file__).parent / "runs.jsonl"

BACKTRACK_RE = re.compile(r"(\d+)-(\d+)-(\d+)-backtrack-(\d+)-(\w+)-text\.txt")
NORMAL_RE = re.compile(r"(\d+)-(\d+)-(\d+)-(\w+)-text\.txt")
BOXED_RE = re.compile(r"\\boxed\{(\d+)\}")
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")
# Directory names like "424e18_5" are attempt #5 of question 424e18. A 6-hex base
# distinguishes these from real qids such as "pe_818" (Project Euler).
QID_ATTEMPT_RE = re.compile(r"^([0-9a-f]{6})_(\d+)$")


def parse_solution_file(filename: str, path: Path) -> dict | None:
    m = BACKTRACK_RE.match(filename)
    is_bt = False
    bt_num: int | None = None
    if m:
        is_bt = True
        idx, tokens, calls, bt_raw, answer = m.groups()
        bt_num = int(bt_raw)
    else:
        m = NORMAL_RE.match(filename)
        if not m:
            return None
        idx, tokens, calls, answer = m.groups()

    try:
        content = path.read_text(errors="replace")
        answer_history = BOXED_RE.findall(content)
    except OSError:
        answer_history = []

    return {
        "file": filename,
        "index": int(idx),
        "tokens": int(tokens),
        "toolCalls": int(calls),
        "answer": answer,
        "answerHistory": answer_history,
        "isBacktrack": is_bt,
        "backtrackNum": bt_num,
    }


def main() -> None:
    records: list[dict] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir() or not RUN_RE.match(run_dir.name):
            continue
        solutions_root = run_dir / "solutions"
        findings_root = run_dir / "findings"
        if not solutions_root.is_dir():
            continue
        for qdir in sorted(solutions_root.iterdir()):
            if not qdir.is_dir():
                continue
            qid = qdir.name
            sols: list[dict] = []
            for f in sorted(qdir.iterdir()):
                if not f.name.endswith("-text.txt"):
                    continue
                parsed = parse_solution_file(f.name, f)
                if parsed:
                    sols.append(parsed)
            if not sols:
                continue

            sols.sort(
                key=lambda s: (
                    s["index"],
                    1 if s["isBacktrack"] else 0,
                    s["backtrackNum"] or 0,
                )
            )

            m = QID_ATTEMPT_RE.match(qid)
            if m:
                base_qid = m.group(1)
                attempt: int | None = int(m.group(2))
            else:
                base_qid = qid
                attempt = None

            rec: dict = {
                "run": run_dir.name,
                "qid": base_qid,
                "attempt": attempt,
                "solutions": sols,
            }

            fpath = findings_root / f"{qid}.json"
            if fpath.exists():
                try:
                    fdata = json.loads(fpath.read_text())
                except (OSError, json.JSONDecodeError):
                    fdata = None
                if fdata is not None:
                    rec["final_answer"] = fdata.get("final_answer")
                    rec["time_taken"] = fdata.get("time_taken")
                    findings = fdata.get("findings", [])
                    rec["findings_count"] = (
                        len(findings) if isinstance(findings, list) else 0
                    )

            records.append(rec)

    with OUTPUT.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
