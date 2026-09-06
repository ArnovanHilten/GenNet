"""Minimal YAML subset for improve/backlog files (no extra dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_doc(path: Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    list_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line == "items:":
            continue
        if line.startswith("  - "):
            if current:
                items.append(current)
            current = {}
            list_key = None
            rest = line[4:]
            if ":" in rest:
                key, val = rest.split(":", 1)
                current[key.strip()] = _scalar(val)
            continue
        if current is None:
            continue
        if line.startswith("      - "):
            if list_key:
                current.setdefault(list_key, [])
                if not isinstance(current[list_key], list):
                    current[list_key] = []
                current[list_key].append(_scalar(line[8:]))
            continue
        if line.startswith("    ") and ":" in line:
            key, val = line.strip().split(":", 1)
            key = key.strip()
            val = val.strip()
            if val == [] or val == "":
                current[key] = []
                list_key = key
            else:
                current[key] = _scalar(val)
                list_key = None
    if current:
        items.append(current)
    return {"items": items}


def dump_doc(path: Path, doc: dict[str, Any]) -> None:
    lines = ["items:"]
    for item in doc.get("items", []):
        first = True
        for key, val in item.items():
            if first:
                lines.append(f"  - {key}: {_fmt(val)}")
                first = False
            elif isinstance(val, list):
                if not val:
                    lines.append(f"    {key}: []")
                else:
                    lines.append(f"    {key}:")
                    for v in val:
                        lines.append(f"      - {_fmt(v)}")
            else:
                lines.append(f"    {key}: {_fmt(val)}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scalar(val: str) -> Any:
    val = val.strip()
    if val in ("[]", ""):
        return []
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
        return int(val)
    return val


def _fmt(val: Any) -> str:
    if isinstance(val, list):
        return "[]"
    return str(val)


def upsert_item(doc: dict[str, Any], item: dict[str, Any]) -> None:
    items = doc.setdefault("items", [])
    for i, existing in enumerate(items):
        if existing.get("id") == item.get("id"):
            merged = dict(existing)
            merged.update({k: v for k, v in item.items() if v is not None})
            items[i] = merged
            return
    items.append(item)
