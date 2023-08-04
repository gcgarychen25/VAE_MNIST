#!/bin/bash 
#SBATCH -J test
#SBATCH -A m3562_g  
#SBATCH -q debug 
#SBATCH -N 1             
#SBATCH -C gpu
#SBATCH -t 00:06:00 
#SBATCH -e %x-%j.err 
#SBATCH -o %x-%j.out
#SBATCH --time-min=00:06:00  
#SBATCH --comment=00:17:00
#SBATCH --signal=B:USR1@60
#SBATCH --requeue
#SBATCH --open-mode=append

#for c/r jobs
module load dmtcp nersc_cr

start_coordinator
chmod +x $VAE_MNIST_DIR/payload.sh
#c/r jobs
if [[ $(restart_count) == 0 ]]; then

    #user setting
    dmtcp_launch -j $VAE_MNIST_DIR/payload.sh &
elif [[ $(restart_count) > 0 ]] && [[ -e $MNIST_WORKING_DIR/dmtcp_restart_script.sh ]]; then

    $MNIST_WORKING_DIR/dmtcp_restart_script.sh &
else

    echo "Failed to restart the job, exit"; exit
fi

# requeueing the job if remaining time >0
ckpt_command=ckpt_dmtcp    #additional checkpointing right before the job hits the wall limit 
requeue_job func_trap USR1

wait
