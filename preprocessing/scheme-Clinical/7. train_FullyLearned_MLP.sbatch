#!/bin/bash
#SBATCH --time=02:00:00 # hh:mm:ss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000 # megabytes

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

source .env
cd $SCRIPT_PATH

start=$(date +%s)
python train_FullyLearned_MLP.py $DOTENV_PATH 2000000 
end=$(date +%s)
echo "Elapsed Time for train_FullyLearned_MLP 2000000: $(($end-$start)) seconds"
