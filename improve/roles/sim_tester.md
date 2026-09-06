# Role: sim tester

## Goal

Run synthetic experiments listed in `improve/backlog/sims.yaml`. Update `last_status` and `last_notes`.

## Commands

From repo root, with `env_GenNet` and `CUDA_VISIBLE_DEVICES=-1`:

```bash
python improve/run.py sim
```

## Stop rules

- Synthetic data only. Do not load real cohorts.
- Fail planted sims if the causal gene is not ranked first (or in the top 2).
- Do not commit generated run folders under `results/`.
