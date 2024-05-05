import pickle
import os
import sys
import numpy as np
import torch
import torchaudio
import matplotlib
import matplotlib.pyplot as plt
import argparse
import psutil
import GPUtil
import time
from tqdm import tqdm

# Navigate up one level to the 'pretraining' directory, where 'dataloader.py' is located
sys.path.append(os.path.abspath('../'))

import dataloader


dataset_mean=-5.0716844 
dataset_std=4.386603
train_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(
        dataset_json_file='/home/bosfab01/SpeakerVerificationBA/data/audioset2M_librispeech960.json',
        audio_conf={
            'num_mel_bins': 128,
            'target_length': 998,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': 'asli',
            'mean': dataset_mean,
            'std': dataset_std,
            'noise': False,
            'mode': 'train',
            'shuffle_frames': True
        },
        label_csv='/home/bosfab01/SpeakerVerificationBA/data/label_information.csv'
    ),
    batch_size=1000,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    drop_last=True
)

mse_list = []

# Processing each batch with tqdm for progress bar
for i, (audio_input, labels) in enumerate(tqdm(train_loader, desc='Processing batches')):
    
    # Randomly select 218 frames from each spectrogram in the batch
    selected_indices = torch.randperm(998)[:218].repeat(audio_input.size(0), 1)
    selected_frames = torch.gather(audio_input, 1, selected_indices.unsqueeze(2).expand(-1, -1, 128))
    
    # Calculate the mean vector of the selected frames
    mean_vector = selected_frames.mean(dim=1)

    # Calculate MSE for the remaining frames
    remaining_indices = torch.tensor([i for i in range(998) if i not in selected_indices[0].tolist()]).repeat(audio_input.size(0), 1)
    remaining_frames = torch.gather(audio_input, 1, remaining_indices.unsqueeze(2).expand(-1, -1, 128))
    
    mse = ((remaining_frames - mean_vector.unsqueeze(1)) ** 2).mean(dim=[1, 2])

    mse_list.extend(mse.tolist())

# Calculate the mean of MSEs
overall_mean_mse = np.mean(mse_list)
print(f'Overall mean MSE: {overall_mean_mse}')