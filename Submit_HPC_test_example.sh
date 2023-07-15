#!/bin/bash

#SBATCH -p short
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-node=1
#SBATCH -J test
#SBATCH --mem=20G
#SBAYCH --mem-per-gpu=30G
#SBATCH -o out.log
#SBATCH -e error.log

# Load the modules


module purge
module load Python/3.7.4-GCCcore-8.3.0
module load libs/cuda/10.1.243
module load libs/cudnn/7.6.5.32-CUDA-10.1.243
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4

source $HOME/venv_GenNet_37/bin/activate


python GenNet.py train -path ./examples/example_classification/ -ID 111311211214261 -epochs 25 -one_hot
