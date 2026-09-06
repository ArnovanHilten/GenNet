# Current task

- id: bug-008
- title: Missing tests for covariates and multiple genotype files
- evidence: tests/test_GenNet.py

## Problem

TODOs list covariates, multiple genotype files, epoch-shuffle randomness.

## Files

tests/test_GenNet.py

## Verify

`CUDA_VISIBLE_DEVICES=-1 pytest tests/test_import.py tests/test_conversion.py tests/test_interpret.py tests/test_GenNet.py`
