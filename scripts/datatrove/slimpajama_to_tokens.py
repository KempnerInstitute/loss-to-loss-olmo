from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data_name", type=str, required=False, default="slimpajama-chunk1")
args = args.parse_args()

DATASET_NAME = args.data_name
HF_PATH = "hf://datasets/cerebras/SlimPajama-627B"
TOKENIZER = "meta-llama/Llama-2-7b-hf"

N_TASKS_PER_NODE = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
NODES = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
RANK = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# DESTINATION = "/n/holyscratch01/kempner_fellows/Lab/data"
DESTINATION = "/n/vast-scratch/kempner_fellows/data"

print(f"Running with {N_TASKS_PER_NODE} tasks per node, {NODES} nodes, and rank {RANK}")


if "chunk1" in DATASET_NAME:
    reader = JsonlReader(
        data_folder=f"{HF_PATH}/train/chunk1",
        glob_pattern="*.jsonl.zst",
        compression="zstd",
        text_key="text",
        id_key="id",
    )
    val_reader = JsonlReader(
        data_folder=f"{HF_PATH}/validation/chunk1",
        glob_pattern="*.jsonl.zst",
        compression="zstd",
        text_key="text",
        id_key="id",
    )
elif "all" in DATASET_NAME:
    reader = JsonlReader(
        data_folder=f"{HF_PATH}/train",
        glob_pattern="*.jsonl.zst",
        compression="zstd",
        text_key="text",
        id_key="id",
    )
    val_reader = JsonlReader(
        data_folder=f"{HF_PATH}/validation",
        glob_pattern="*.jsonl.zst",
        compression="zstd",
        text_key="text",
        id_key="id",
    )
else:
    raise ValueError(f"Unknown dataset {DATASET_NAME}")


def make_executor(reader, val=False):
    if val:
        out_path = f"{DESTINATION}/{DATASET_NAME}-val-tokenized"
    else:
        out_path = f"{DESTINATION}/{DATASET_NAME}-tokenized"
    return LocalPipelineExecutor(
        pipeline=[
            reader,
            DocumentTokenizer(
                output_folder=out_path,
                tokenizer_name_or_path=TOKENIZER,
                eos_token="</s>",
                shuffle=True,
                seed=0,
            ),
        ],
        tasks=N_TASKS_PER_NODE * NODES,
        workers=-1,
        logging_dir=f"{DESTINATION}/logs/datatrove/{DATASET_NAME}-val-{val}",
        # local flags
        local_tasks=N_TASKS_PER_NODE,
        local_rank_offset=RANK * N_TASKS_PER_NODE,
        start_method="fork",
    )


dist_executor = make_executor(reader)
dist_executor.run()

dist_executor = make_executor(val_reader, val=True)
dist_executor.run()
