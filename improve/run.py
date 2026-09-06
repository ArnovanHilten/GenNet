#!/usr/bin/env python3
"""Improve-lab CLI: scan, propose, one-task, sim."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from improve import providers, yamlutil

BUGS = ROOT / "improve" / "backlog" / "bugs.yaml"
FEATURES = ROOT / "improve" / "backlog" / "features.yaml"
SIMS = ROOT / "improve" / "backlog" / "sims.yaml"
CURRENT = ROOT / "improve" / "backlog" / "current.md"
ROLES = ROOT / "improve" / "roles"
TODO_RE = re.compile(r"(?i)#\s*to.?do\b.*")
PYTEST_CMD = [
    sys.executable,
    "-m",
    "pytest",
    "tests/test_import.py",
    "tests/test_conversion.py",
    "tests/test_interpret.py",
    "tests/test_improve.py",
    "tests/test_GenNet.py",
    "-q",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="GenNet improve-lab runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="Run pytest and harvest TODOs into bugs.yaml")
    p_propose = sub.add_parser("propose", help="Optional LLM hunter + planner")
    p_propose.add_argument("--dry-run", action="store_true")
    p_one = sub.add_parser("one-task", help="Triage one bug into current.md")
    p_one.add_argument("--dry-run", action="store_true")
    p_sim = sub.add_parser("sim", help="Run synthetic sims from sims.yaml")
    p_sim.add_argument("--id", default=None, help="Run only this sim id")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.chdir(ROOT)
    if args.cmd == "scan":
        return cmd_scan()
    if args.cmd == "propose":
        return cmd_propose(dry_run=args.dry_run)
    if args.cmd == "one-task":
        return cmd_one_task(dry_run=args.dry_run)
    if args.cmd == "sim":
        return cmd_sim(sim_id=args.id)
    return 1


def cmd_scan() -> int:
    pytest_ok, pytest_out = _run_pytest()
    doc = yamlutil.load_doc(BUGS)
    harvested = 0
    skip_dirs = {".git", "results", "improve", "__pycache__", ".pytest_cache"}
    for path in ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in path.parts):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("GenNet_utils/hase/"):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if TODO_RE.search(line):
                item_id = f"todo-{_slug(rel)}-{i}"
                if any(x.get("id") == item_id for x in doc["items"]):
                    continue
                yamlutil.upsert_item(
                    doc,
                    {
                        "id": item_id,
                        "title": line.strip()[:120],
                        "severity": "low",
                        "status": "open",
                        "evidence": f"{rel}:{i}",
                        "notes": "Harvested by scan",
                        "source": "scan",
                    },
                )
                harvested += 1
    yamlutil.dump_doc(BUGS, doc)
    print(f"pytest_ok={pytest_ok} harvested={harvested}")
    if pytest_out:
        print(pytest_out[-2000:])
    return 0 if pytest_ok else 1


def cmd_propose(dry_run: bool) -> int:
    hunter = (ROLES / "bug_hunter.md").read_text(encoding="utf-8")
    planner = (ROLES / "planner.md").read_text(encoding="utf-8")
    user = (
        "Repository: GenNet\n"
        f"Known bugs:\n{BUGS.read_text(encoding='utf-8')}\n"
        f"Known features:\n{FEATURES.read_text(encoding='utf-8')}\n"
        "Return only YAML items that should be appended. Do not invent file paths."
    )
    if dry_run or not providers.configured():
        print("=== bug hunter prompt ===")
        print(hunter)
        print("=== planner prompt ===")
        print(planner)
        print("=== user context (truncated) ===")
        print(user[:1500])
        if not providers.configured():
            print("IMPROVE_API_KEY unset; dry-run only.")
        return 0
    hunter_out = providers.chat(hunter, user)
    planner_out = providers.chat(planner, user)
    (ROOT / "improve" / "backlog" / "last_hunter.txt").write_text(hunter_out, encoding="utf-8")
    (ROOT / "improve" / "backlog" / "last_planner.txt").write_text(planner_out, encoding="utf-8")
    print(hunter_out)
    print(planner_out)
    print("Wrote last_hunter.txt and last_planner.txt. Review before merging into YAML.")
    return 0


def cmd_one_task(dry_run: bool) -> int:
    doc = yamlutil.load_doc(BUGS)
    chosen = None
    for item in doc["items"]:
        if item.get("status") in ("open", "accepted"):
            chosen = item
            break
    if chosen is None:
        print("No open or accepted bugs.")
        return 0
    body = (
        f"# Current task\n\n"
        f"- id: {chosen['id']}\n"
        f"- title: {chosen.get('title')}\n"
        f"- evidence: {chosen.get('evidence')}\n\n"
        f"## Problem\n\n{chosen.get('notes', '')}\n\n"
        f"## Files\n\n{chosen.get('evidence', '')}\n\n"
        f"## Verify\n\n"
        f"`CUDA_VISIBLE_DEVICES=-1 pytest tests/test_import.py tests/test_conversion.py "
        f"tests/test_interpret.py tests/test_GenNet.py`\n"
    )
    if dry_run:
        print(body)
        return 0
    if chosen.get("status") == "open":
        chosen["status"] = "accepted"
        yamlutil.dump_doc(BUGS, doc)
    CURRENT.write_text(body, encoding="utf-8")
    print(f"Wrote {CURRENT} for {chosen['id']}")
    if providers.configured():
        fixer = (ROLES / "fixer.md").read_text(encoding="utf-8")
        proposal = providers.chat(fixer, body)
        proposal_path = ROOT / "improve" / "backlog" / "proposed_fix.md"
        proposal_path.write_text(proposal, encoding="utf-8")
        print(f"Wrote LLM proposal to {proposal_path}. Review before applying.")
    return 0


def cmd_sim(sim_id: str | None) -> int:
    doc = yamlutil.load_doc(SIMS)
    failed = 0
    for item in doc["items"]:
        if sim_id and item.get("id") != sim_id:
            continue
        command = item["command"]
        timeout = int(item.get("max_seconds") or 180)
        print(f"Running {item['id']}: {command}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
            )
            ok = proc.returncode == 0
            item["last_status"] = "passed" if ok else "failed"
            item["last_notes"] = (proc.stdout + proc.stderr)[-500:].replace("\n", " ")
            if not ok:
                failed += 1
                print(proc.stdout)
                print(proc.stderr)
        except subprocess.TimeoutExpired:
            item["last_status"] = "failed"
            item["last_notes"] = f"timeout after {timeout}s"
            failed += 1
    yamlutil.dump_doc(SIMS, doc)
    return 1 if failed else 0


def _run_pytest() -> tuple[bool, str]:
    proc = subprocess.run(
        PYTEST_CMD,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]


if __name__ == "__main__":
    sys.exit(main())
