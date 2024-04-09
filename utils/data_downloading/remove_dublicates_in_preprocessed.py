import os
import hashlib
from pydub import AudioSegment
from tqdm import tqdm

def file_hash(filepath):
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def find_and_delete_duplicates(directory):
    """
    Find duplicate .flac files in the given directory and its subdirectories and delete them,
    keeping only the first file of each set of duplicates. It also prints the number of duplicates
    in relation to the total number of files.
    """
    hashes = {}
    all_subdirs = [x[0] for x in os.walk(directory)]
    total_files = 0
    deleted_count = 0

    for subdir in tqdm(all_subdirs, desc="Processing directories"):
        files = [f for f in os.listdir(subdir) if f.lower().endswith('.flac')]
        total_files += len(files)
        for filename in files:
            filepath = os.path.join(subdir, filename)
            audio_hash = file_hash(filepath)

            # If hash is already seen, it's a duplicate; delete this file
            if audio_hash in hashes:
                os.remove(filepath)
                deleted_count += 1
            else:
                hashes[audio_hash] = filepath

    return deleted_count, total_files

# Example usage
if __name__ == "__main__":
    directory = '/home/bosfab01/SpeakerVerificationBA/data/preprocessed'  # Replace with your directory
    deleted_count, total_files = find_and_delete_duplicates(directory)
    print(f"{deleted_count}/{total_files} duplicate files were deleted.")
