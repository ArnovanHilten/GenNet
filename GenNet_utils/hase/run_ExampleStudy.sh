#!/usr/bin/env bash

mkdir -p ./test/results/

python hase.py \
-th 0 \
-o ./test/results/ \
-mode regression \
-g ./test/ExampleStudy/examplestudy/ \
-ph ./test/ExampleStudy/phenotype/ \
-cov ./test/ExampleStudy/covariates/ \
-study_name example_study \
-maf 0