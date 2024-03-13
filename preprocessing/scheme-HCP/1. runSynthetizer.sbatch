#!/bin/bash
#SBATCH --time=01:00:00 # hh:mm:ss
#SBATCH --array=1-22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000 # megabytes

source .env
cd $SCRIPT_PATH

echo "Task ID: synthetizer $SLURM_ARRAY_TASK_ID"
python runSynthetizer.py $DOTENV_PATH $SLURM_ARRAY_TASK_ID
