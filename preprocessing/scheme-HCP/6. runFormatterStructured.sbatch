#!/bin/bash
#SBATCH --time=01:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000 # megabytes

source .env
cd $SCRIPT_PATH

echo "Task ID: formatter structured"
python runFormatterStructured.py $DOTENV_PATH 
