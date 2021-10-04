#!/bin/bash
#SBATCH --mem=40G
#SBATCH --ntasks=6
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH -o //data/scratch/avanhilten/GenNet_logs/out_%j.log
#SBATCH -e //data/scratch/avanhilten/GenNet_logs/error_%j.log

# Load the modules

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load libs/cuda/10.1.243
module load libs/cudnn/7.6.5.32-CUDA-10.1.243
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4

source $HOME/venv_GenNet_37/bin/activate

cd  /trinity/home/avanhilten/repositories/Dev/GenNet/

python /trinity/home/avanhilten/repositories/Dev/GenNet/GenNet.py train /trinity/home/avanhilten/repositories/UK_biobank/imputed/Red_hair_colour/equal/ $1 -genotype_path /trinity/home/avanhilten/data/genotype/ -bs $2