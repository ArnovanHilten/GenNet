#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=40G
#SBATCH -p thin
#SBATCH -t 5-00:00:00
#SBATCH -o ./SLURM_logs/out_%j.log
#SBATCH -e ./SLURM_logs/error_%j.log

# Load the modules


source $HOME/venv_GenNet_37/bin/activate

python Convert.py -j $1 -w $2 -lr $3 -bs $4 -l1 $5 -mt $6 -pn $7
