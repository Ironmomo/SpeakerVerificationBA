#!/bin/bash

# Function to display error message and exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Check if wget command is available
command -v wget >/dev/null 2>&1 || { error_exit "wget command not found. Please install wget."; }

# Check if tar command is available
command -v tar >/dev/null 2>&1 || { error_exit "tar command not found. Please install tar."; }

if [ "$1" == "librispeech" ]; then
    # List of download URLs
    download_urls=(
        "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
        "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
        "https://www.openslr.org/resources/12/train-other-500.tar.gz"
    )
    download_dir="./data/librispeech"

    # Create directory if it doesn't exist
    mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"

    # Iterate over the list of download URLs
    for ((i=0; i<${#download_urls[@]}; i++)); do
        download_url="${download_urls[i]}"
        tar_file="$download_dir/file$i.tar.gz"

        # Download the data
        echo "Downloading data from $download_url ..."
        wget -O "$tar_file" "$download_url" || error_exit "Failed to download data from $download_url"

        # Extract the data
        echo "Extracting data to $download_dir ..."
        tar -xzvf "$tar_file" -C "$download_dir" || error_exit "Failed to extract data"

        # Remove the tar file
        rm "$tar_file"

        # Find Subdir with data
        subdirs=$(find "$download_dir"/LibriSpeech -mindepth 1 -maxdepth 1 -type d -not -name ".*" -exec basename {} \;)

        for subdir in $subdirs; do
            # Move the contents of each subdirectory to $download_dir
            mv "$download_dir"/LibriSpeech/"$subdir"/* "$download_dir" || error_exit "Failed to move files"
        done
        
        # Remove src dir
        rm -r "$download_dir"/LibriSpeech || error_exit "Failed to remove source dir"

        echo "Data extraction completed successfully."
    done

elif [ "$1" == "voxceleb" ]; then
    # List of download URLs
    download_urls=(
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_test_wav.zip?download=true"
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox1/vox1_dev_wav.zip?download=true"
        #"https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_aac_1.zip?download=true"
        #"https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_aac_2.zip?download=true"
    )
    download_dir="./data/voxcelebtemp"
    dest_dir="./data/voxceleb"

    # Create directory if it doesn't exist
    mkdir -p "$dest_dir" || error_exit "Failed to create directory: $dest_dir"

    # Iterate over the list of download URLs
    for ((i=0; i<${#download_urls[@]}; i++)); do
        # Create directory
        mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"
        download_url="${download_urls[i]}"
        zip_file="$download_dir/file$i.zip"

        # Download the data
        echo "Downloading data from $download_url ..."
        wget -O "$zip_file" "$download_url" || error_exit "Failed to download data from $download_url"

        # Extract the data
        echo "Extracting data to $download_dir ..."
        unzip "$zip_file" -d "$download_dir" || error_exit "Failed to extract data"

        # Remove the tar file
        rm "$zip_file"

        # Find Subdir with data
        subdirs=$(find "$download_dir" -mindepth 1 -maxdepth 1 -type d -not -name ".*" -exec basename {} \;)

        for subdir in $subdirs; do
            echo "Moving from $subdir to $dest_dir"
            # Move the contents of each subdirectory to $download_dir
            mv "$download_dir"/"$subdir"/* "$dest_dir" || error_exit "Failed to move files"
            # Remove src dir
            rm -r "$download_dir"/"$subdir" || error_exit "Failed to remove source dir"
        done
        
        # Remove Download dir
        rm -r "$download_dir"

    echo "Data extraction completed successfully."
    done
else
    echo "Invalid argument. Please provide either 'librispeech' or 'voxceleb'."
fi
