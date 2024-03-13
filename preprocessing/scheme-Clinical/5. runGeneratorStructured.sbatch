#!/bin/bash
#SBATCH --time=48:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000 # megabytes

source .env
cd $SCRIPT_PATH

python runGeneratorStructured_MF.py $DOTENV_PATH "CSD"
python runGeneratorStructured_MF.py $DOTENV_PATH "GROUNDTRUTH"

python runGeneratorStructured_NNLS.py $DOTENV_PATH "CSD"

python runGeneratorStructured_SH.py $DOTENV_PATH

