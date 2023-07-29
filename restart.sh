#!/bin/bash

#SBATCH -N 1            # Number of nodes
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
#SBATCH --time-min=00:02:00

export VAE_MNIST_DIR=$1
export WORKING_DIR=$SCRATCH/output_VAE_MNIST
export LATENT_DIM=$2
export BATCH_SIZE=$3
export N_EPOCHS=$4
export EXPR_ID=trial-LD-$LATENT_DIM-BS-$BATCH_SIZE-N_EPOCHS-$N_EPOCHS
cd $WORKING_DIR
mkdir -p trials
mkdir -p trials/$EXPR_ID
module load pytorch/2.0.1

module load dmtcp nersc_cr
start_coordinator -i 1
./dmtcp_restart_script.sh  