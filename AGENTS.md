# GenNet agent guide

This repository is the ComPopBio fork of [ArnovanHilten/GenNet](https://github.com/ArnovanHilten/GenNet): a command-line framework for interpretable neural networks for genetic phenotype prediction.

Work on a feature branch. Open pull requests against `master`. Do not train on large real genotypes in this environment; use the bundled toy examples.

## Environment

Activate the project conda env before running Python, tests, or the CLI:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_GenNet
```

- Python 3.10.12, TensorFlow 2.11, pins in `requirements_GenNet.txt`
- This machine has GPUs, but TF 2.11 needs CUDA 11 libraries that are not installed. Use CPU (the project is designed for CPU sparse matmuls):

```bash
export CUDA_VISIBLE_DEVICES=-1
```

- Working directory for CLI and pytest: repository root

## Commands

```bash
python GenNet.py --help
python GenNet.py train --help
python GenNet.py convert --help
python GenNet.py topology --help
python GenNet.py plot --help
python GenNet.py interpret --help

# Smoke train (classification toy example)
python GenNet.py train -path ./examples/example_classification/ -ID 1 -epochs 2

# Regression toy example
python GenNet.py train -path ./examples/example_regression/ -ID 2 -problem_type regression -epochs 2

# Test suite used by CI
CUDA_VISIBLE_DEVICES=-1 pytest tests/test_import.py tests/test_conversion.py tests/test_GenNet.py
```

A train run writes `results/GenNet_experiment_<ID>_/`. That directory is gitignored.

## Layout

| Path | Role |
| --- | --- |
| `GenNet.py` | CLI entry (`convert`, `train`, `plot`, `topology`, `interpret`) |
| `GenNet_utils/` | Training, topology, plots, interpretation, custom Keras layers |
| `examples/` | Toy classification, regression, covariates, plink, A-to-Z tutorial |
| `tests/` | Pytest coverage of convert/train/plot/interpret |
| `interpretation/` | Extra interpretation scripts (NID, DFIM, RLIPP, …) |
| `Dockerfile` | Reproducible `env_GenNet` image |

A run folder needs `genotype.h5`, `subjects.csv`, and `topology.csv`. See the README for column conventions. Indices are 0-based.

## Improve lab

Self-improvement roles and backlog live in `improve/`. Pick a role file and follow it:

```bash
# Deterministic scan (no API key)
python improve/run.py scan

# Print hunter/planner prompts, or call IMPROVE_API_KEY / IMPROVE_BASE_URL / IMPROVE_MODEL
python improve/run.py propose --dry-run

# Write improve/backlog/current.md from the next open bug
python improve/run.py one-task

# Synthetic sims (CPU)
python improve/run.py sim --id planted-pathway
```

Roles: `improve/roles/bug_hunter.md`, `triage.md`, `fixer.md`, `planner.md`, `sim_tester.md`. Schema: `improve/schema.md`.

## Testing rules

- Run the pytest suite above before considering a change done.
- Keep tests short: few epochs, toy examples only.
- Do not commit `results/`, `processed_data/`, `.pytest_cache/`, `examples/A_to_Z/new_run_folder/`, or `improve/sims/_planted_run/`.

## Constraints

- Do not weaken pins in `requirements_GenNet.txt` unless a test proves the new versions work.
- Do not add secrets, real cohort data, or large generated HDF5 files.
- Issues are disabled on this fork (`has_issues: false`). Track work in PRs and this file.
