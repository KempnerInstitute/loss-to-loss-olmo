import os


def remove_files_with_bad_name(root_dir, bad_name=None, remove_sharded=True):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if (bad_name is not None and bad_name in str(file_path)) or (
                remove_sharded and "step" in str(file_path) and "unsharded" not in str(file_path)
            ):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


# Example usage
root_directory = "/n/holyscratch01/sham_lab/color-scale/ckpts"
bad_name = "train_data/global_indices.npy"  # TODO: handle train_data_0, train_data_1, etc.

remove_files_with_bad_name(root_directory, bad_name, remove_sharded=True)
