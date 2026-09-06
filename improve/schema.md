# Improve-lab backlog schema

All backlog files are YAML maps with an `items` list. Agents must keep existing `id` values stable.

## `bugs.yaml` item

- `id` (string, required): stable id, e.g. `bug-001`
- `title` (string, required)
- `severity`: `high` | `medium` | `low`
- `status`: `open` | `accepted` | `fixed` | `wontfix`
- `evidence` (string): file path and optional line
- `notes` (string): extra context
- `source`: `seed` | `scan` | `hunter` | `human`

Do not delete a `fixed` item. Change `status` only.

## `features.yaml` item

- `id` (string, required): e.g. `feat-001`
- `title` (string, required)
- `motivation` (string)
- `dependencies` (list of ids)
- `risk`: `high` | `medium` | `low`
- `suggested_pr` (string): first small change
- `status`: `idea` | `accepted` | `in_progress` | `done`

## `sims.yaml` item

- `id` (string, required)
- `kind`: `smoke` | `planted`
- `command` (string): run from repo root
- `max_seconds` (int)
- `last_status`: `unrun` | `passed` | `failed`
- `last_notes` (string)

## `current.md`

Written by triage. One task only: problem, files, how to verify.
