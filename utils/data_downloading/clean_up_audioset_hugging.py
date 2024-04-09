import os
import requests
from pyunpack import Archive
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Constants
BASE_URL = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/"
OUTPUT_DIR = "unbal_train_audio"
NUM_FILES = 870  # Total number of files you listed
MAX_WORKERS = 32  # Number of parallel download/processing tasks

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def is_processed(i):
    """Check if the file has already been processed based on your folder structure."""
    extraction_base_path = os.path.join(OUTPUT_DIR, f"unbal_train{i:03}")
    extraction_path = os.path.join(extraction_base_path, 'audio', 'unbal_train')
    return os.path.exists(extraction_path) and any(
        f.endswith(".flac") for _, _, files in os.walk(extraction_path) for f in files
    )

def process_file(i):
    """Process a single file by downloading, extracting, and converting its contents."""
    filename = f"unbal_train{i:03}.tar"
    if is_processed(i):
        print(f"{filename} is already processed.")
        return
    
    file_url = f"{BASE_URL}{filename}?download=true"
    tar_path = os.path.join(OUTPUT_DIR, filename)
    
    # Download
    print(f"Downloading {filename} from {file_url}...")
    response = requests.get(file_url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    
    # Extract
    print(f"Extracting {filename}...")
    extraction_path = os.path.join(OUTPUT_DIR, f"unbal_train{i:03}")
    if not os.path.exists(extraction_path):
        os.makedirs(extraction_path)
    
    Archive(tar_path).extractall(extraction_path)
    os.remove(tar_path)
    
    # Conversion process in the specified nested structure
    target_extraction_path = os.path.join(extraction_path, 'audio', 'unbal_train')
    if os.path.exists(target_extraction_path):
        for root, _, files in os.walk(target_extraction_path):
            for audio_file in files:
                if audio_file.endswith(".flac"):
                    input_path = os.path.join(root, audio_file)
                    output_path = input_path.replace(".flac", "_16kHz_16bit_mono.flac")
                    cmd = f"ffmpeg -i {input_path} -ar 16000 -ac 1 -sample_fmt s16 {output_path}"
                    os.system(cmd)
                    os.remove(input_path)

# Use ProcessPoolExecutor to parallelize
with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    list(tqdm(executor.map(process_file, range(NUM_FILES)), total=NUM_FILES))

print("All missing files processed.")
