import os
import concurrent.futures
from pydub import AudioSegment
from tqdm import tqdm

def convert_file_to_flac(file_path, source_folder, target_folder, target_sample_rate=16000, target_channels=1):
    """
    Convert a single audio file to FLAC format with the specified sample rate and channels.
    """
    try:
        relative_path = os.path.relpath(file_path, source_folder)
        target_file_path = os.path.join(target_folder, os.path.splitext(relative_path)[0] + '.flac')
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(target_sample_rate).set_channels(target_channels)
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
        audio.export(target_file_path, format='flac')
        return None
    except Exception as e:
        return file_path

def batch(iterable, n=1):
    """
    Splits the iterable into batches of size n.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def convert_audio_to_flac_parallel(source_folder, target_folder, target_sample_rate=16000, target_channels=1):
    """
    Convert audio files in parallel, providing updates via a single progress bar for all files.
    """
    os.makedirs(target_folder, exist_ok=True)
    files_to_convert = [os.path.join(subdir, file) for subdir, dirs, files in os.walk(source_folder) for file in files if file.lower().endswith('.mp3')]
    
    cpu_count = os.cpu_count()
    print("Available CPU cores:", cpu_count)
    max_workers = cpu_count // 2
    print("Using max workers:", max_workers)

    error_count = 0
    total_files = len(files_to_convert)

    # Initialize progress bar outside the batch loop
    with tqdm(total=total_files, desc="Converting files") as progress:
        for file_batch in batch(files_to_convert, max_workers * 10):  # Batch size can be adjusted based on your observation
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(convert_file_to_flac, file, source_folder, target_folder, target_sample_rate, target_channels): file for file in file_batch}

                for future in concurrent.futures.as_completed(future_to_file):
                    if future.result() is not None:
                        error_count += 1
                    # Update the progress bar after each file is processed
                    progress.update(1)
                    progress.set_postfix(errors=error_count)

# Define and execute conversion
source_folder = 'unbalanced'
target_folder = 'unbalanced_flac'
convert_audio_to_flac_parallel(source_folder, target_folder)
