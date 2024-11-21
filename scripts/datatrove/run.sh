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
#SBATCH --array=0-5
#SBATCH --exclude=holygpu8a15401,holygpu8a19103,holygpu8a19105
#SBATCH --spread-job

source ~/.bashrc
conda deactivate
conda activate loss-to-loss

# python scripts/datatrove/fineweb_to_tokens.py --data_name=fineweb-recent-1T
# python scripts/datatrove/fineweb-edu_to_tokens.py --data_name=fineweb-edu-100B
# python scripts/datatrove/slimpajama_to_tokens.py --data_name=slimpajama-chunk1

# python scripts/datatrove/pythonedu_to_parquet.py
# python scripts/datatrove/pythonedu_to_tokens.py # 130

# python scripts/datatrove/smollm_to_tokens.py --data_name=fineweb-edu-dedup # 234
# python scripts/datatrove/smollm_to_tokens.py --data_name=cosmopedia-v2 # 104
# python scripts/datatrove/smollm_to_tokens.py --data_name=open-web-math # 114

# python scripts/datatrove/starcoder_to_tokens.py
python scripts/datatrove/proofpile_to_tokens.py
