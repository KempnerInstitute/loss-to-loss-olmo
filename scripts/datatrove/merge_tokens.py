from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.tokens import DocumentTokenizerMerger
import os
import shutil
import argparse

args = argparse.ArgumentParser()
args.add_argument("--data_name", type=str, required=False, default="fineweb-recent-1T")
args = args.parse_args()

DATASET_NAME = args.data_name
FILE_SIZE = int(5e8)

# DESTINATION = "/n/holyscratch01/kempner_fellows/Lab/data"
DESTINATION = "/n/vast-scratch/kempner_fellows/data"


# Remove unshuffled files (these are just empty/incomplete files if they exist)
for file in os.listdir(f"{DESTINATION}/{DATASET_NAME}-tokenized"):
    if "unshuffled" in file:
        os.remove(os.path.join(f"{DESTINATION}/{DATASET_NAME}-tokenized", file))

# Execute merger
dist_executor = LocalPipelineExecutor(
    pipeline=[
        DocumentTokenizerMerger(
            input_folder=f"{DESTINATION}/{DATASET_NAME}-tokenized",
            output_folder=f"{DESTINATION}/{DATASET_NAME}-merged",
            max_tokens_per_file=FILE_SIZE,
            save_filename=f"{DATASET_NAME}",
        ),
    ],
    tasks=1,
    logging_dir=f"{DESTINATION}/logs/datatrove-merge/{DATASET_NAME}",
    # local flags
    local_tasks=1,
    start_method="fork",
)
dist_executor.run()

# Create validation set
os.makedirs(f"{DESTINATION}/{DATASET_NAME}-val-merged", exist_ok=True)
# Move 000_{DATASET_NAME}.ds
shutil.move(
    f"{DESTINATION}/{DATASET_NAME}-merged/000_{DATASET_NAME}.ds",
    f"{DESTINATION}/{DATASET_NAME}-val-merged/000_{DATASET_NAME}.ds",
)
# Move 000_{DATASET_NAME}.ds.index
shutil.move(
    f"{DESTINATION}/{DATASET_NAME}-merged/000_{DATASET_NAME}.ds.index",
    f"{DESTINATION}/{DATASET_NAME}-val-merged/000_{DATASET_NAME}.ds.index",
)
# Move 000_{DATASET_NAME}.ds.metadata
shutil.move(
    f"{DESTINATION}/{DATASET_NAME}-merged/000_{DATASET_NAME}.ds.metadata",
    f"{DESTINATION}/{DATASET_NAME}-val-merged/000_{DATASET_NAME}.ds.metadata",
)
