#!/bin/bash
#SBATCH --time=05:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=20000 # megabytes

source .env
cd $SCRIPT_PATH

python runFormatter.py $DOTENV_PATH
