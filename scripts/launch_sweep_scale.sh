#!/bin/bash
#SBATCH --job-name=color-filter
#SBATCH --output=/n/holyscratch01/sham_lab/color-scale/logs/%A_%a.log
#SBATCH --nodes=4              
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=24
#SBATCH --time=72:00:00
#SBATCH --mem=1000GB		
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner_h100
#SBATCH --constraint=h100
#SBATCH --array=1-6
#SBATCH --exclude=holygpu8a15401

# sleep $((RANDOM % 120))

# Custom environment
source ~/.bashrc
conda deactivate
conda activate loss-to-loss

export HF_DATASETS_OFFLINE=1 # Only use cached data

export CONFIG=$1

# Accept sweep config as argument
export SWEEP_CONFIG=$2

# Accept job index as argument if there is a second argument
if [ -z "$3" ]
then
    echo $SLURM_ARRAY_TASK_ID
else
    export SLURM_ARRAY_TASK_ID=$3
fi

# Set default path for checkpoints if not set
export CHECKPOINTS_PATH="/n/holyscratch01/sham_lab/color-scale/ckpts"

# TODO: does this help?
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set ntasks if not set
if [ -z "$SLURM_NTASKS_PER_NODE" ]
then
    export SLURM_NTASKS_PER_NODE=1
fi

# Boilerplate environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPICH_GPU_SUPPORT_ENABLED=1
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}

export PYTHONPATH=.:${PYTHONPATH}

export PYTORCH_KERNEL_CACHE_PATH=/tmp/pytorch_kernel_cache/
mkdir -p $PYTORCH_KERNEL_CACHE_PATH

python scripts/run_sweep.py config=${CONFIG} sweep_config=${SWEEP_CONFIG}