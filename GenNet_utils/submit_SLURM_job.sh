#!/bin/bash
#SBATCH -p normal
#SBATCH -t 5-00:00:00
#SBATCH -o ./SLURM_logs/out_%j.log
#SBATCH -e ./SLURM_logs/error_%j.log

# Load the modules

source $HOME/venv_GenNet/bin/activate
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4
python Convert.py -j $1 -w $2 -lr $3 -bs $4 -l1 $5 -mt $6 -pn $7
