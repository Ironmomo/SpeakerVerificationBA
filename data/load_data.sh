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

# Set download URL and directory
download_url="https://www.openslr.org/resources/12/dev-clean.tar.gz"
download_dir="./data"
tar_file="$download_dir/dev-clean.tar.gz"

# Create directory if it doesn't exist
mkdir -p "$download_dir" || error_exit "Failed to create directory: $download_dir"

# Download the data
echo "Downloading data from $download_url ..."
wget -O "$download_dir/dev-clean.tar.gz" "$download_url" || error_exit "Failed to download data from $download_url"

# Extract the data
echo "Extracting data to $download_dir ..."
tar -xzvf "$tar_file" -C "$download_dir" || error_exit "Failed to extract data"

# Remove the tar file
rm "$tar_file"

echo "Data extraction completed successfully."

