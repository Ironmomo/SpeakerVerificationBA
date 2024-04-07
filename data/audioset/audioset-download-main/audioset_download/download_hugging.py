import os
import requests
from pyunpack import Archive
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
BASE_URL = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/"
OUTPUT_DIR = "unbal_train_audio"
NUM_FILES = 870  # Total number of files you listed
MAX_WORKERS = 32  # Number of parallel download/processing tasks

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_file(i):
    filename = f"unbal_train{i:03}.tar"
    file_url = f"{BASE_URL}{filename}?download=true"
    tar_path = os.path.join(OUTPUT_DIR, filename)
    
    # Download
    print(f"Downloading {filename} from {file_url}...")
    response = requests.get(file_url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    
    # Define extraction path
    extraction_path = os.path.join(OUTPUT_DIR, f"unbal_train{i:03}")
    if not os.path.exists(extraction_path):
        os.makedirs(extraction_path)

    # Extract
    print(f"Extracting {filename}...")
    Archive(tar_path).extractall(extraction_path)
    os.remove(tar_path)
    
    # Convert and delete FLAC files
    for root, dirs, files in os.walk(extraction_path):
        for audio_file in files:
            if audio_file.endswith(".flac"):
                input_path = os.path.join(root, audio_file)
                output_path = input_path.replace(".flac", "_16kHz_16bit_mono.flac")
                cmd = f"ffmpeg -i {input_path} -ar 16000 -ac 1 -sample_fmt s16 {output_path}"
                os.system(cmd)
                os.remove(input_path)

# Use ProcessPoolExecutor to parallelize
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks to the executor
    futures = [executor.submit(process_file, i) for i in range(NUM_FILES)]
    
    # Wait for all futures to complete
    for future in as_completed(futures):
        future.result()  # This will re-raise any exception occurred in a worker

print("All files downloaded, extracted, converted, and cleaned up.")
