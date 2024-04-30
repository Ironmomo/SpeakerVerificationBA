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
    for url in "${download_urls[@]}"; do
        tar_file="$download_dir/file$i.tar.gz"

        # Download the data
        echo "Downloading data from $url ..."
        wget -O "$tar_file" "$url" || error_exit "Failed to download data from $url"

        # Extract the data
        echo "Extracting data to $download_dir ..."
        tar -xzvf "$tar_file" -C "$download_dir" || error_exit "Failed to extract data"

        # Remove the tar file
        rm "$tar_file"

        # Find Subdir with data
        subdirs=$(find "$download_dir"/LibriSpeech -mindepth 2 -maxdepth 2 -type d)

        for subdir in $subdirs; do
	    files=$(find "$subdir" -mindepth 1 -type f -name "*.flac")
            speakerId=$(basename $subdir)
            for file in $files; do
            	fileName=$(basename $file)
            	mkdir -p "$download_dir/$speakerId" || error_exit "Failed to create directory: $download_dir/$speakerId"
                # Move the contents of each subdirectory to $download_dir
                mv "$file" "$download_dir/$speakerId/$fileName" || error_exit "Failed to move files"
            done
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
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_aac_1.zip?download=true"
        "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_aac_2.zip?download=true"
    )
    download_dir="./data/voxcelebtemp"
    dest_dir="./data/voxceleb"

    # Create directory if it doesn't exist
    mkdir -p "$dest_dir" || error_exit "Failed to create directory: $dest_dir"

    # Iterate over the list of download URLs
    for url in "${download_urls[@]}"; do

        # Create directory
        mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"
        zip_file="$download_dir/file.zip"

        # Download the data
        echo "Downloading data from $url ..."
        wget -O "$zip_file" "$url" || error_exit "Failed to download data from $url"

        # Extract the data
        echo "Extracting data to $download_dir ..."
        unzip "$zip_file" -d "$download_dir" || error_exit "Failed to extract data"

        # Remove the tar file
        rm "$zip_file"

        # Find Subdir with data
        subdirs=$(find "$download_dir" -type d  -name "id[0-9]*")

        for subdir in $subdirs; do
            
            files=$(find "$subdir" -type f)
            speakerId=$(basename "$subdir")

            # Create speaker dir if not exists
            mkdir -p "$dest_dir/$speakerId" || error_exit "Failed to create directory: $dest_dir/$speakerId"

            for file in $files; do
                mv "$file" "$dest_dir/$speakerId"
            done
        done
        
        # Remove Download dir
        rm -r "$download_dir"

    done
    echo "Data extraction completed successfully."

elif [ "$1" == "audioset" ]; then

    processUrl() {
        local download_dir=$1
        local dest_dir=$2
        local url=$3

        tar_file="$download_dir/file.tar.gz"

        mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"

        # Download the data
        echo "Downloading data from $url ..."
        wget -O "$tar_file" "$url" || error_exit "Failed to download data from $url"

        # Extract the data
        echo "Extracting data to $download_dir ..."
        tar -xvf "$tar_file" -C "$download_dir" || error_exit "Failed to extract data"

        # Remove the tar file
        rm "$tar_file"

        # Find Subdir with data
        files=$(find "$download_dir" -type f -name "*.flac";)

        for file in $files; do
            basename=$(basename "$file")
            uid=$(uuidgen)
            # Resample file to 16khz mono
            ffmpeg -i "$file" -ar 16000 -ac 1 -sample_fmt s16 "$dest_dir/$uid$basename"
        done

        # Remove src dir
        rm -r "$download_dir" || error_exit "Failed to remove source dir"

    }

    # Define the base URL
    base_url="https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/unbal_train"

    download_dir="./data/audiosettemp"
    dest_dir="./data/audioset/unbal"

    # Create directory if it doesn't exist
    mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"
    mkdir -p "$dest_dir" || error_exit "Failed to create directory: $dest_dir"

    # Define the total number of links you need
    total_links=869

    # Loop to generate the URLs
    for ((i=0; i<=total_links-4; i+=4)); do
        # Generate the URL and add it to the array
        url1="$base_url$(printf '%03d' "$((i))").tar?download=true"
        url2="$base_url$(printf '%03d' "$((i+1))").tar?download=true"
        url3="$base_url$(printf '%03d' "$((i+2))").tar?download=true"
        url4="$base_url$(printf '%03d' "$((i+3))").tar?download=true"

        processUrl "$download_dir/1" "$dest_dir" "$url1" &
        processUrl "$download_dir/2" "$dest_dir" "$url2" &
        processUrl "$download_dir/3" "$dest_dir" "$url3" &
        processUrl "$download_dir/4" "$dest_dir" "$url4" &

        wait

    done

    url1="$base_url$(printf '%03d' "868").tar?download=true"
    url2="$base_url$(printf '%03d' "869").tar?download=true"
    processUrl "$download_dir/1" "$dest_dir" "$url1" &
    processUrl "$download_dir/2" "$dest_dir" "$url2" &

    wait


    download_urls=(
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval00.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval01.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval02.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval03.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval04.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval05.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval06.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval07.tar?download=true"
        "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/eval08.tar?download=true"
    )

    download_dir="./data/audiosettemp"
    dest_dir="./data/audioset/eval"

    # Create directory if it doesn't exist
    mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"
    mkdir -p "$dest_dir" || error_exit "Failed to create directory: $dest_dir"

    # Iterate over the list of download URLs
    for url in "${download_urls[@]}"; do
        tar_file="$download_dir/file.tar.gz"
        
        # Create directory if it doesn't exist
        mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"

        # Download the data
        echo "Downloading data from $url ..."
        wget -O "$tar_file" "$url" || error_exit "Failed to download data from $url"

        # Extract the data
        echo "Extracting data to $download_dir ..."
        tar -xvf "$tar_file" -C "$download_dir" || error_exit "Failed to extract data"

        # Remove the tar file
        rm "$tar_file"

        # Find Subdir with data
        files=$(find "$download_dir"/audio -mindepth 1 -type f -name "*.flac";)

        for file in $files; do
            basename=$(basename "$file")
            uid=$(uuidgen)
            # Resample file to 16khz mono
            ffmpeg -i "$file" -ar 16000 -ac 1 -sample_fmt s16 "$dest_dir/$uid$basename"
        done

        # Remove src dir
        rm -r "$download_dir" || error_exit "Failed to remove source dir"
        
    done

    echo "Data extraction completed successfully."

else
    echo "Invalid argument. Please provide either 'librispeech', 'audioset' or 'voxceleb'."
fi