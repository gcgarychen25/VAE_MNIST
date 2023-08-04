#!/bin/bash
#SBATCH -q preempt
#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH --time=00:02:00
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out
#SBATCH --comment=00:10:00  #desired time limit
#SBATCH --signal=B:USR1@30  #sig_time (30 seconds) should match your checkpoint overhead time
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=gchen4@lbl.gov

export VAE_MNIST_DIR=/global/homes/g/gchen4/Optuna_Testing/VAE_MNIST
export WORKING_DIR=$SCRATCH/output_VAE_MNIST
export LATENT_DIM=2
export BATCH_SIZE=8
export N_EPOCHS=100
export EXPR_ID=trial-LD-$LATENT_DIM-BS-$BATCH_SIZE-N_EPOCHS-$N_EPOCHS
cd $WORKING_DIR
mkdir -p trials
mkdir -p trials/$EXPR_ID
module load pytorch/2.0.1

python /global/homes/g/gchen4/Optuna_Testing/VAE_MNIST/main.py --latent_dim $LATENT_DIM --batch_size $BATCH_SIZE \
                --epochs $N_EPOCHS --trial_id $EXPR_ID \
                --working_dir $WORKING_DIR > $WORKING_DIR/trials/$EXPR_ID/log_$EXPR_ID.txt