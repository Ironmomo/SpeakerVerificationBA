import os
import sys
import time


# Count files in folder "path" and its subfolders, also calculate total size
def count_files_and_size(path):
    count = 0
    total_size = 0
    for root, dirs, files in os.walk(path):
        count += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
    return count, total_size

# Function to convert bytes to gigabytes with appropriate precision
def bytes_to_gb(bytes):
    return bytes / 1024 / 1024 / 1024

# Count and size
path = "unbalanced_flac"
count, size = count_files_and_size(path)

# Print count and size
print(f"Number of files in folder {path}: {count}")
print(f"Total size: {bytes_to_gb(size):.2f} GB")