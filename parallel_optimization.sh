#!/bin/bash

#SBATCH -N 2            # Number of nodes
#SBATCH -J VAE_MNIST      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=gchen4@lbl.gov


module load python
conda activate VAE_MNIST

export OPTUNA_STORAGE="mysql://username:password@localhost:3306/optuna_database"  # Set Optuna storage to your MySQL instance
export OUTPUT_VAE_DIR=$SCRATCH/output_VAE_MNIST

echo "jobstart $(date)";pwd
python /path_to_your_script/optuna_testing.py &>> $OUTPUT_VAE_DIR/log.txt # Run your script
echo "jobend $(date)";pwd
