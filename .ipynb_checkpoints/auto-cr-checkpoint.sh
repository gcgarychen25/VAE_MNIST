#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J CT_NVAE_CR      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:02:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o ./logs/output_log_%j.txt
#SBATCH -e ./logs/error_log_%j.txt
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=gchen4@lbl.gov
#SBATCH --time-min=00:02:00
#SBATCH --comment=00:16:00
#SBATCH --signal=B:USR1@60 
#SBATCH --requeue 
#SBATCH --open-mode=append

export VAE_MNIST_DIR=/global/homes/g/gchen4/Optuna_Testing/VAE_MNIST
export WORKING_DIR=$SCRATCH/output_VAE_MNIST
export LATENT_DIM=2
export BATCH_SIZE=64
export N_EPOCHS=180
export EXPR_ID=trial-LD-$LATENT_DIM-BS-$BATCH_SIZE-N_EPOCHS-$N_EPOCHS
cd $WORKING_DIR
mkdir -p trials
mkdir -p trials/$EXPR_ID
module load pytorch/2.0.1

#for c/r jobs
module load dmtcp nersc_cr
start_coordinator -i 30
#c/r jobs
if [[ $(restart_count) == 0 ]]; then

    #user setting
    dmtcp_launch -j python /global/homes/g/gchen4/Optuna_Testing/VAE_MNIST/main.py --latent_dim $LATENT_DIM --batch_size $BATCH_SIZE \
                --epochs $N_EPOCHS --trial_id $EXPR_ID \
                --working_dir $WORKING_DIR > $WORKING_DIR/trials/$EXPR_ID/log_$EXPR_ID.txt &
elif [[ $(restart_count) > 0 ]] && [[ -e dmtcp_restart_script.sh ]]; then

    ./dmtcp_restart_script.sh &
else

    echo "Failed to restart the job, exit"; exit
fi

# requeueing the job if remaining time >0
ckpt_command=ckpt_dmtcp    #additional checkpointing right before the job hits the wall limit 
requeue_job func_trap USR1

wait