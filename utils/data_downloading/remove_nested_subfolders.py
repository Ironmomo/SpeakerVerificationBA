import os
import shutil

# this script removes one level of nested subfolders in the directory structure of the audioset (can be adapted to other datasets)

# Define the root directory containing the dataset
root_dir = "/home/bosfab01/SpeakerVerificationBA/data/audioset_unbal_train_audio"

# Walk through the directory structure
for subdir, dirs, files in os.walk(root_dir):
    # Identify directories that match the pattern 'unbal_trainXXX/audio/unbal_train'
    if os.path.basename(subdir) == 'unbal_train' and os.path.basename(os.path.dirname(subdir)) == 'audio':
        parent_dir = os.path.dirname(subdir)  # This should be the 'audio' directory
        # Move each FLAC file up one directory level to the 'audio' directory
        for file in files:
            if file.endswith('.flac'):
                src_path = os.path.join(subdir, file)
                dst_path = os.path.join(parent_dir, file)
                shutil.move(src_path, dst_path)
        # Remove the empty 'unbal_train' directory
        os.rmdir(subdir)
