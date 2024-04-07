import os
from pydub.utils import mediainfo
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

# Initial count and size
path = "unbalanced_flac"
initial_count, initial_size = count_files_and_size(path)
initial_time = time.time()

# Wait for some time to estimate speed; adjust sleep time as needed
time.sleep(10)

# Updated count and size after the interval
updated_count, updated_size = count_files_and_size(path)
updated_time = time.time()

# Calculate the speed of acquiring new files and their sizes
delta_count = updated_count - initial_count
delta_size = updated_size - initial_size
delta_time = updated_time - initial_time

# Estimate speed (files per second and bytes per second)
if delta_time > 0:
    speed_files_per_sec = delta_count / delta_time
    speed_bytes_per_sec = delta_size / delta_time
else:
    speed_files_per_sec = 0
    speed_bytes_per_sec = 0

# Estimate total time and storage requirements for 2 million files
target_files = 1600000
if speed_files_per_sec > 0:
    remaining_files = target_files - updated_count
    estimated_time_sec = remaining_files / speed_files_per_sec
    estimated_end_time = time.time() + estimated_time_sec
    estimated_total_size_bytes = (target_files * (updated_size / updated_count)) if updated_count > 0 else 0
    estimated_total_size_gb = bytes_to_gb(estimated_total_size_bytes)
    current_storage_used_gb = bytes_to_gb(updated_size)

    print(f"Current number of files: {updated_count}")
    print(f"Current storage used: {current_storage_used_gb:.2f} GB")
    print(f"Estimated completion time: {time.ctime(estimated_end_time)}")
    print(f"Estimated total storage required: {estimated_total_size_gb:.2f} GB")
else:
    print("Insufficient data to estimate download speed.")


# Path to the directory containing the audio file
audio_dir = 'unbalanced_flac'  # Update this path to your actual directory

# Function to find the first audio file in a directory
def find_first_audio_file(directory):
    supported_audio_formats = ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(supported_audio_formats):
                return os.path.join(root, file)
    return None

# Find the first audio file
file_path = find_first_audio_file(audio_dir)

if file_path:
    # print(f"File:  {os.path.basename(file_path)}")
    
    # Get media info of the file
    try:
        audio_info = mediainfo(file_path)
    except Exception as e:
        print(f"Error retrieving media info: {e}")
        exit()

    # Extract and print the sample rate and channels
    sample_rate = audio_info.get('sample_rate')
    channels = audio_info.get('channels')
    resol = audio_info.get('resolution')

    if sample_rate and channels:
        print(f"File {os.path.basename(file_path)} is sampled at {sample_rate} Hz with {channels} channels and resolution {resol}.")
    else:
        print("Failed to extract media information. Please check the file and dependencies.")
else:
    print("No audio files found in the directory.")
