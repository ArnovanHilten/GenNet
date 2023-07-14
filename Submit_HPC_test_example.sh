#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 5-00:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gpus-per-node=1
#SBATCH -J test
#SBATCH --mem=20G
#SBAYCH --mem-per-gpu=30G
#SBATCH -o ./GenNet_utils/SLURM_logs/out_%j.log
#SBATCH -e ./GenNet_utils/SLURM_logs/error_%j.log

# Load the modules


module purge
module load 2021
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

source $HOME/env_GenNet/bin/activate

cd /home/ahilten/repositories/GenNet/

python GenNet.py train -path ./examples/example_classification/ -ID 11131121121424
