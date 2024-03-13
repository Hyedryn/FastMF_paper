#!/bin/bash
#SBATCH --time=01:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4000 # megabytes

source .env
cd $SCRIPT_PATH

python runSynthetizerStructured.py $DOTENV_PATH 
