from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens import DocumentTokenizer
import os
import argparse

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

import boto3
import botocore
from botocore.exceptions import ClientError
from smart_open import open

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


def file_to_doc(s3, blob_id):
    try:
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as s3bucket:
            text = s3bucket.read().decode("utf-8", errors="ignore")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            print(f"File not found: {blob_id}")
            text = ""
        else:
            print(f"other error: {blob_id}")
            text = ""
    except Exception as e:
        print(f"Not client error: {blob_id}")
        text = ""
    return Document(id=blob_id, text=text, metadata={})


class StackExpander(PipelineStep):
    def run(self, data, rank: int = 0, world_size: int = 1):
        s3 = boto3.client(
            "s3", region_name="us-west-2", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
        )
        for document in data:
            with self.track_time():
                yield file_to_doc(s3, document.text)


dist_executor = LocalPipelineExecutor(
    pipeline=[
        ParquetReader(
            data_folder=f"/n/vast-scratch/kempner_fellows/data/python-edu-parquet",
            glob_pattern="*.parquet",
            text_key="text",
            id_key="text",
        ),
        StackExpander(),
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
