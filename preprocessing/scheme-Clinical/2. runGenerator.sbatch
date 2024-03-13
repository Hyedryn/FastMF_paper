#!/bin/bash
#SBATCH --time=48:00:00 # hh:mm:ss
#SBATCH --array=1-22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000 # megabytes

source .env
cd $SCRIPT_PATH

python runGenerator_MF.py $DOTENV_PATH $SLURM_ARRAY_TASK_ID "CSD"
python runGenerator_MF.py $DOTENV_PATH $SLURM_ARRAY_TASK_ID "GROUNDTRUTH"

python runGenerator_NNLS.py $DOTENV_PATH $SLURM_ARRAY_TASK_ID "CSD"

python runGenerator_SH.py $DOTENV_PATH $SLURM_ARRAY_TASK_ID 
