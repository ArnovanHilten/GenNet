#!/bin/bash
#SBATCH -p normal
#SBATCH -t 5-00:00:00
#SBATCH -o ./GenNet_utils/SLURM_logs/out_%j.log
#SBATCH -e ./GenNet_utils/SLURM_logs/error_%j.log

## MAKE SURE THIS IS EXECUTABLE! For example run the following command: chmod u+x submit_SLURM_job.sh
# Load the modules
module load 2019
# activate venv
source $HOME/venv_GenNet_dev/bin/activate
# run conversion
python ./GenNet_utils/Convert.py -job_begins $1 -job_tills $2 -job_n $3 -study_name $4 -outfolder $5 -tcm $6
