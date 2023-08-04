#!/bin/bash
export VAE_MNIST_DIR=/global/homes/g/gchen4/Optuna_Testing/VAE_MNIST
export WORKING_DIR=$SCRATCH/output_VAE_MNIST
export LATENT_DIM=2
export BATCH_SIZE=64
export N_EPOCHS=400
export EXPR_ID=final-auto-cr-testing-2
cd $WORKING_DIR
mkdir -p trials
mkdir -p trials/$EXPR_ID
module load pytorch/2.0.1

python /global/homes/g/gchen4/Optuna_Testing/VAE_MNIST/main.py --latent_dim $LATENT_DIM --batch_size $BATCH_SIZE \
                --epochs $N_EPOCHS --trial_id $EXPR_ID \
                --working_dir $WORKING_DIR > $WORKING_DIR/trials/$EXPR_ID/log_$EXPR_ID.txt