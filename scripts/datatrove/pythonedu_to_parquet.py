from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data_name", type=str, required=False, default="python-edu")
args = args.parse_args()

DATASET_NAME = args.data_name
HF_PATH = "hf://datasets/HuggingFaceTB/smollm-corpus"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# DESTINATION = "/n/holyscratch01/kempner_fellows/Lab/data"
DESTINATION = "/n/vast-scratch/kempner_fellows/data/smollm-corpus"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")


dist_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=f"{HF_PATH}/python-edu",
            glob_pattern="*.parquet",
            text_key="blob_id",
            id_key="blob_id",
        ),
        ParquetWriter(
            output_folder=f"{DESTINATION}/{DATASET_NAME}-parquet",
            max_file_size=5 * 2**20,  # 5mb
        ),
    ],
    tasks=N_TASKS_PER_NODE * NODES,
    workers=-1,
    logging_dir=f"{DESTINATION}/logs/datatrove/{DATASET_NAME}-parquet",
    # local flags
    local_tasks=N_TASKS_PER_NODE,
    local_rank_offset=RANK * N_TASKS_PER_NODE,
    start_method="fork",
)

dist_executor.run()
