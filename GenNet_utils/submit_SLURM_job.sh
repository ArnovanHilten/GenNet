#!/bin/bash
#SBATCH -p normal
#SBATCH -t 5-00:00:00
#SBATCH -o ./GenNet_utils/SLURM_logs/out_%j.log
#SBATCH -e ./GenNet_utils/SLURM_logs/error_%j.log

# Load the modules
module load 2019
source $HOME/venv_GenNet_dev/bin/activate
python ./GenNet_utils/Convert.py -job_begins $1 -job_tills $2 -job_n $3 -study_name $4 -outfolder $5 -tcm $6
