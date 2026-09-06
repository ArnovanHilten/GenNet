# Improve lab

In-repo backlog and agent roles for continuously checking bugs, ranking fixes, collecting features, and running synthetic sims.

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_GenNet
export CUDA_VISIBLE_DEVICES=-1
python improve/run.py scan
python improve/run.py propose --dry-run
python improve/run.py one-task
python improve/run.py sim --id planted-pathway
```

Secrets for an external LLM: `IMPROVE_API_KEY`, `IMPROVE_BASE_URL`, `IMPROVE_MODEL`. Never commit keys.

See `schema.md` and `roles/`.
