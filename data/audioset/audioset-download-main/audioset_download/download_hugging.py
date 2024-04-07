import os
import requests
from pyunpack import Archive

# Constants
BASE_URL = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/"
OUTPUT_DIR = "unbal_train_audio"
NUM_FILES = 870  # Total number of files you listed

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for i in range(NUM_FILES):
    filename = f"unbal_train{i:03}.tar"
    # Append ?download=true to the URL to attempt a direct download
    file_url = f"{BASE_URL}{filename}?download=true"
    
    print(f"Downloading {filename} from {file_url}...")
    tar_path = os.path.join(OUTPUT_DIR, filename)
    response = requests.get(file_url, stream=True)
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    
    # Define the extraction path
    extraction_path = os.path.join(OUTPUT_DIR, f"unbal_train{i:03}")
    if not os.path.exists(extraction_path):
        os.makedirs(extraction_path)

    # Extract the TAR file
    print(f"Extracting {filename}...")
    Archive(tar_path).extractall(extraction_path)
    os.remove(tar_path)  # Remove the TAR file immediately to save space
    
    # Convert and delete FLAC files
    for root, dirs, files in os.walk(extraction_path):
        for audio_file in files:
            if audio_file.endswith(".flac"):
                input_path = os.path.join(root, audio_file)
                output_path = input_path.replace(".flac", "_16kHz_16bit_mono.flac")
                os.system(f"ffmpeg -i {input_path} -ar 16000 -ac 1 -sample_fmt s16 {output_path}")
                os.remove(input_path)  # Remove the original FLAC file to save space

print("All files downloaded, extracted, converted, and cleaned up.")
