#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --account=kempner_sham_lab
#SBATCH --output=/n/holyscratch01/kempner_fellows/Lab/data/slurm_logs/%A_%a.log
#SBATCH --nodes=1             
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1    
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=250GB		
#SBATCH --partition=kempner
#SBATCH --array=0-0
#SBATCH --exclude=holygpu8a15401,holygpu8a19103
#SBATCH --spread-job

source ~/.bashrc
conda deactivate
conda activate loss-to-loss

# python scripts/datatrove/merge_tokens.py --data_name=fineweb-recent-1T
# python scripts/datatrove/merge_tokens.py --data_name=fineweb-edu-100B

# python scripts/datatrove/merge_tokens.py --data_name=python-edu
# python scripts/datatrove/merge_tokens.py --data_name=fineweb-edu-dedup
# python scripts/datatrove/merge_tokens.py --data_name=cosmopedia-v2
# python scripts/datatrove/merge_tokens.py --data_name=open-web-math

python scripts/datatrove/merge_tokens.py --data_name=starcoder