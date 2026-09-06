# Role: bug hunter

## Goal

Find defects in GenNet from tests, CLI, TODOs, and pytest output. Write or update items in `improve/backlog/bugs.yaml` only.

## Inputs

- Repository root
- `improve/schema.md`
- `improve/backlog/bugs.yaml`
- Pytest output if available

## Outputs

Updated `bugs.yaml` items with `source: hunter` for new findings. Keep existing ids. Do not delete `fixed` items.

## Stop rules

- Do not edit `GenNet.py`, `GenNet_utils/`, or tests except comments in the backlog.
- One pass. No secrets. No real cohort data.
- Every new bug needs `evidence` (file path).
