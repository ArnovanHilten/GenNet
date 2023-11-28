#!/bin/bash
#SBATCH --mem=40G
#SBATCH --ntasks=15
#SBATCH --job-name=DFIM_GenNet
#SBATCH --partition=short,long
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o out.log
#SBATCH -e error.log


# Load the modules
module purge
source /trinity/home/avanhilten/miniconda3/etc/profile.d/conda.sh
conda init
conda activate env_GenNet_tf28

CUDA_VISIBLE_DEVICES='' python GenNet.py interpret  -resultpath ./results/GenNet_experiment_101_/  -type pathexplain -num_eval 100  -start_rank 0  -end_rank 5  & 
CUDA_VISIBLE_DEVICES='' python GenNet.py interpret  -resultpath ./results/GenNet_experiment_101_/  -type pathexplain -num_eval 100  -start_rank 5  -end_rank 10 &
CUDA_VISIBLE_DEVICES='' python GenNet.py interpret  -resultpath ./results/GenNet_experiment_101_/  -type pathexplain -num_eval 100  -start_rank 10 -end_rank 15 &
CUDA_VISIBLE_DEVICES='' python GenNet.py interpret  -resultpath ./results/GenNet_experiment_101_/  -type pathexplain -num_eval 100  -start_rank 15 -end_rank 20 &

wait

echo "All completed"