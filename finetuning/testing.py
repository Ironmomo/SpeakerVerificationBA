# Navigate up one level to the 'pretraining' directory, where 'dataloader.py' is located
import os, sys
import torch

sys.path.append(os.path.abspath('../SpeakerVerificationBA'))

import dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fintune = True

json_f = "/home/fassband/ba/SpeakerVerificationBA/data/prep/augmentation.json"
csv_f = "/home/fassband/ba/SpeakerVerificationBA/data/prep/augmentation.csv"
audio_conf = {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': 'asli',
            'mean': -3.6925695,
            'std': 4.020388,
            'noise': False,
            'mode': 'train',
            'shuffle_frames': False,
            'finetuning': fintune
        }

aset = dataloader.AudioDataset(audio_conf=audio_conf, dataset_json_file=json_f, label_csv=csv_f)

train_loader = torch.utils.data.DataLoader(
    aset,
    batch_size=24,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=True
)

# Create an iterator from the DataLoader
data_iterator = iter(train_loader)

if fintune:
    # Fetch the first batch
    audio_input1, audio_input2, labels = next(data_iterator)
    audio_input1, audio_input2, labels = audio_input1.to(device), audio_input2.to(device), labels.to(device)
    # Print out the details to see what the batch contains
    print("Audio input shape:", audio_input1.shape)
    print("Audio input 2 shape:", audio_input2.shape)
    print("Labels shape:", labels.shape)
else:
        # Fetch the first batch
    audio_input1, labels = next(data_iterator)

    # Print out the details to see what the batch contains
    print("Audio input shape:", audio_input1.shape)
    print("Labels shape:", labels.shape)
    
exit(0)