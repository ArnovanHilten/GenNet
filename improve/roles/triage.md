# Role: triage

## Goal

Pick the smallest high-value open bug and write `improve/backlog/current.md`.

## Inputs

- `improve/backlog/bugs.yaml`
- `improve/schema.md`

## Outputs

`improve/backlog/current.md` with:

- Problem
- Files to change
- How to verify (pytest command)

Set the chosen bug `status` to `accepted` if it was `open`.

## Stop rules

- Exactly one task.
- Prefer test-hardening over HASE TODOs.
- Do not implement the fix.
