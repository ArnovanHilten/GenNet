#!/bin/bash
#SBATCH --mem=10G
#SBATCH --ntasks=6
#SBATCH -p express 
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00
#SBATCH -o /trinity/home/avanhilten/repositories/GenNet/GenNet_utils/SLURM_logs/slurm_logs/out_%j.log
#SBATCH -e /trinity/home/avanhilten/repositories/GenNet/GenNet_utils/SLURM_logs/error_%j.log

# Load the modules

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load libs/cuda/10.1.243
module load libs/cudnn/7.6.5.32-CUDA-10.1.243
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4

source $HOME/env_GenNet_dev/bin/activate

cd /trinity/home/avanhilten/repositories/GenNet/

python GenNet.py train ./examples/example_classification/ 111
