import json
import os
import argparse
import csv


def add_csv_augmentation(file_name, entries):
    if not os.path.exists(file_name):
        
        with open(file_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            # Write header
            csv_writer.writerow(['index','mid','display_name'])
            
            # Write rows
            for idx, (filename) in enumerate(entries):
                csv_writer.writerow([idx, "/m/03l9g", "/m/03l9g"])
    
    else:
        
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write rows
            for idx, (filename) in enumerate(entries):
                csv_writer.writerow([idx, "/m/03l9g", "/m/03l9g"])


def add_json_augmentation(json_file, entries):
    if not os.path.exists(json_file):
        data = { "data":[] }
    
    else:
        
        with open(json_file, 'r') as file:
            data = json.load(file)
    
    # Append entrie
    for filename in entries:
        data['data'].append({"wav": filename, "labels": "/m/03l9g"})
    
    
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)


def preprocess_data(data_path, csv_augmentation, json_augmentation):
    
    # find all audio files
    audio_files = []
    for root, dirs, files in os.walk(data_path):
        for file_name in files:
            # Check if the file is a regular file is audio file
            file_ending = file_name.split(".")[-1]

            if file_ending.lower() in ["flac", "wav", "aac", "mp3"]:
                audio_files.append(os.path.join(data_path, file_name))

    # add entries to augmentation
    add_csv_augmentation(csv_augmentation, audio_files)
    add_json_augmentation(json_augmentation, audio_files)
                            
            
            
# Example usage
if __name__ == "__main__":
    
    root_path = os.path.join(os.getcwd(), 'data')

    parser = argparse.ArgumentParser(description='Your CLI description here')
    parser.add_argument('-d', '--dataset', help='Specify the dataset path relative to data')
    parser.add_argument('-n', '--new', help='Specify the destination folder where to store the preprocessed data and the augmentation.csv relative to data')
    args = parser.parse_args()

    if args.dataset and args.new:
        print("Starting to create Augmentation entries")
        data_path = os.path.join(root_path, args.dataset)
        new_data_path = os.path.join(root_path, args.new)
        os.makedirs(new_data_path, exist_ok=True)
        CSV_AUGMENTATION_FILE = os.path.join(new_data_path, 'augmentation.csv')
        JSON_AUGMENTATION_FILE = os.path.join(new_data_path, 'augmentation.json')
        preprocess_data(data_path=data_path, csv_augmentation=CSV_AUGMENTATION_FILE, json_augmentation=JSON_AUGMENTATION_FILE)
        print("Finished")
    else:
        print('No value provided for dataset and/or new data path')