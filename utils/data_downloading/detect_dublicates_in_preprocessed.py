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

def find_duplicates(directory, output_file):
    """
    Find duplicate .flac files in the given directory and its subdirectories,
    write the paths of duplicates to an output file, and print the number of
    duplicates as a fraction of the total number of files.
    """
    hashes = {}
    all_subdirs = [x[0] for x in os.walk(directory)]
    total_files = 0
    duplicates_count = 0

    with open(output_file, 'w') as f_out:
        for subdir in tqdm(all_subdirs, desc="Processing directories"):
            files = [f for f in os.listdir(subdir) if f.lower().endswith('.flac')]
            total_files += len(files)
            for filename in files:
                filepath = os.path.join(subdir, filename)
                audio_hash = file_hash(filepath)

                # If hash is already seen, it's a duplicate
                if audio_hash in hashes:
                    f_out.write(f"{hashes[audio_hash]}\n{filepath}\n\n")
                    duplicates_count += 1
                else:
                    hashes[audio_hash] = filepath

    return duplicates_count, total_files

# Example usage
if __name__ == "__main__":
    directory = '/home/bosfab01/SpeakerVerificationBA/data/preprocessed'  # Replace with your directory
    output_file = 'duplicates.txt'  # Path to the output text file
    duplicates_count, total_files = find_duplicates(directory, output_file)
    print(f"{duplicates_count}/{total_files} are duplicates. Duplicate file paths have been written to {output_file}.")
