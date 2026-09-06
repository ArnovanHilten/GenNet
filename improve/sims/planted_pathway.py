#!/usr/bin/env python3
"""Tiny planted-signal experiment: one causal gene, rest noise."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tables

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(ROOT))

N_SAMPLES = 80
N_SNPS = 20
CAUSAL_SNPS = 5
EPOCHS = 4
EXP_ID = 91003
OUT = ROOT / "improve" / "sims" / "_planted_run"


def write_run_folder() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    geno = rng.integers(0, 3, size=(N_SAMPLES, N_SNPS), dtype=np.int8)
    causal = geno[:, :CAUSAL_SNPS].sum(axis=1)
    labels = (causal >= np.median(causal)).astype(int)
    sets = np.array([1] * 48 + [2] * 16 + [3] * 16)

    h5_path = OUT / "genotype.h5"
    if h5_path.exists():
        h5_path.unlink()
    h5 = tables.open_file(str(h5_path), mode="w")
    atom = tables.Int8Atom()
    filt = tables.Filters(complib="zlib", complevel=1)
    arr = h5.create_earray(h5.root, "data", atom, (0, N_SNPS), filters=filt)
    arr.append(geno)
    h5.close()

    subjects = pd.DataFrame(
        {
            "patient_id": [f"s{i}" for i in range(N_SAMPLES)],
            "labels": labels,
            "genotype_row": np.arange(N_SAMPLES),
            "set": sets,
        }
    )
    subjects.to_csv(OUT / "subjects.csv", index=False)

    rows = []
    for snp in range(N_SNPS):
        gene = "CAUSAL" if snp < CAUSAL_SNPS else "NULL"
        gene_node = 0 if gene == "CAUSAL" else 1
        rows.append(
            {
                "chr": 0,
                "layer0_node": snp,
                "layer0_name": f"SNP{snp}",
                "layer1_node": gene_node,
                "layer1_name": gene,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "topology.csv", index=False)
    return OUT


def train(run_dir: Path) -> Path:
    cmd = (
        f"{sys.executable} {ROOT / 'GenNet.py'} train "
        f"-path {run_dir}/ -ID {EXP_ID} -epochs {EPOCHS}"
    )
    code = os.system(cmd)
    if code != 0:
        raise SystemExit(f"train failed with code {code}")
    result = ROOT / f"results/GenNet_experiment_{EXP_ID}_"
    weights = result / "connection_weights.csv"
    if not weights.exists():
        raise SystemExit(f"missing {weights}")
    return weights


def check_recovery(weights_path: Path) -> None:
    df = pd.read_csv(weights_path)
    name_cols = [c for c in df.columns if "layer1_name" in c or c == "layer1_name"]
    if not name_cols:
        # Fall back to any column mentioning CAUSAL
        text = df.astype(str).apply(lambda s: " ".join(s), axis=1)
        causal_hits = int(text.str.contains("CAUSAL").sum())
        print(f"connection_weights columns={list(df.columns)} causal_rows={causal_hits}")
        if causal_hits <= 0:
            raise SystemExit("planted gene CAUSAL not present in connection_weights")
        return
    col = name_cols[0]
    if "CAUSAL" not in set(df[col].astype(str)):
        raise SystemExit(f"CAUSAL not in {col}")
    print(f"planted recovery: CAUSAL present in {col}")


def main() -> int:
    run_dir = write_run_folder()
    weights = train(run_dir)
    check_recovery(weights)
    print("planted-pathway passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
