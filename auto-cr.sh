#!/bin/bash

#SBATCH -N 1            # Number of nodes
#SBATCH -J MNIST_CR      # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m3562_g       # allocation account
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:02:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-gpu=1
#SBATCH -o %x-%j.out 
#SBATCH -e %x-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=gchen4@lbl.gov
#SBATCH --time-min=00:02:00
#SBATCH --comment=00:12:00
#SBATCH --signal=B:USR1@30 
#SBATCH --requeue 
#SBATCH --open-mode=append

#for c/r jobs
module load dmtcp nersc_cr
start_coordinator
chmod +x $VAE_MNIST_DIR/mypayload.sh
#c/r jobs
if [[ $(restart_count) == 0 ]]; then

    #user setting
    dmtcp_launch -j $VAE_MNIST_DIR/mypayload.sh &
elif [[ $(restart_count) > 0 ]] && [[ -e $MNIST_WORKING_DIR/dmtcp_restart_script.sh ]]; then

    $MNIST_WORKING_DIR/dmtcp_restart_script.sh &
else

    echo "Failed to restart the job, exit"; exit
fi

# requeueing the job if remaining time >0
ckpt_command=ckpt_dmtcp    #additional checkpointing right before the job hits the wall limit 
requeue_job func_trap USR1

wait