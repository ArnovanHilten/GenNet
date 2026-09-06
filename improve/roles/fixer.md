# Role: fixer

## Goal

Implement only the task in `improve/backlog/current.md`.

## Verify

```bash
export CUDA_VISIBLE_DEVICES=-1
pytest tests/test_import.py tests/test_conversion.py tests/test_GenNet.py tests/test_interpret.py
```

Mark the matching bug `fixed` only if pytest exits 0.

## Stop rules

- Do not change `requirements_GenNet.txt` unless a test requires it.
- Do not commit `results/`, generated HDF5, or secrets.
- Do not start a second task in the same run.
