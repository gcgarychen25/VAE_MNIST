#!/bin/bash

#SBATCH -N 2            # Number of nodes
#SBATCH -J VAE_MNIST      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o ./logs/output_log_%j.txt
#SBATCH -e ./logs/error_log_%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=gchen4@lbl.gov


module load python
module load pytorch/2.0.1
conda activate VAE_MNIST

srun --nodes=1 --ntasks=1 python optuna_testing.py &
srun --nodes=1 --ntasks=1 python optuna_testing.py &
wait
