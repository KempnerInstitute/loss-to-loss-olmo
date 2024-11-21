from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data_name", type=str, required=False, default="fineweb-edu-100B")
args = args.parse_args()

DATASET_NAME = args.data_name
HF_PATH = "hf://datasets/HuggingFaceFW/fineweb-edu"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# DESTINATION = "/n/holyscratch01/kempner_fellows/Lab/data"
DESTINATION = "/n/vast-scratch/kempner_fellows/data"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")


if "100B" in DATASET_NAME:
    reader = ParquetReader(
        HF_PATH,  # read directly from huggingface
        glob_pattern="sample/100BT/*.parquet",
        text_key="text",
        id_key="id",
    )
elif "all" in DATASET_NAME:
    reader = ParquetReader(
        HF_PATH,  # read directly from huggingface
        paths_file="data//*/*.parquet",
        text_key="text",
        id_key="id",
    )
else:
    raise ValueError(f"Unknown dataset {DATASET_NAME}")

dist_executor = LocalPipelineExecutor(
    pipeline=[
        reader,
        DocumentTokenizer(
            output_folder=f"{DESTINATION}/{DATASET_NAME}-tokenized",
            tokenizer_name_or_path=TOKENIZER,
            eos_token="</s>",
            shuffle=True,
            seed=0,
        ),
    ],
    tasks=N_TASKS_PER_NODE * NODES,
    workers=-1,
    logging_dir=f"{DESTINATION}/logs/datatrove/{DATASET_NAME}",
    # local flags
    local_tasks=N_TASKS_PER_NODE,
    local_rank_offset=RANK * N_TASKS_PER_NODE,
    start_method="fork",
)
dist_executor.run()
