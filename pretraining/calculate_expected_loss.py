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

script_dir = os.path.dirname(__file__) # Get the directory of the script being run (which is in the current directory)
parent_dir = os.path.dirname(script_dir) # Move up to the parent directory (one level up)
sys.path.append(parent_dir) # Add the parent directory to sys.path

import dataloader

task = 'eval'
L = 998

dataset_mean=-5.0716844 
dataset_std=4.386603

if task == 'eval':
    dataset_json_file = 'data/audioset_eval.json'
    N = 400
elif task == 'train':
    dataset_json_file = 'data/audioset2M_librispeech960.json'
    N = 390

tilde_N_masked = 2 * N
tilde_N_unmasked = L - tilde_N_masked

train_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(
        dataset_json_file=dataset_json_file,
        audio_conf={
            'num_mel_bins': 128,
            'target_length': L,
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
    batch_size=500,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    drop_last=True
)

mse_list = []

# Processing each batch with tqdm for progress bar
for i, (audio_input, labels) in enumerate(tqdm(train_loader, desc='Processing batches')):
    
    # Randomly select tilde_N_unmasked frames from each spectrogram in the batch
    selected_indices = torch.randperm(L)[:tilde_N_unmasked].repeat(audio_input.size(0), 1)
    selected_frames = torch.gather(audio_input, 1, selected_indices.unsqueeze(2).expand(-1, -1, 128))
    
    # Calculate the mean vector of the selected frames
    mean_vector = selected_frames.mean(dim=1)

    # Calculate MSE for the remaining frames
    remaining_indices = torch.tensor([i for i in range(L) if i not in selected_indices[0].tolist()]).repeat(audio_input.size(0), 1)
    remaining_frames = torch.gather(audio_input, 1, remaining_indices.unsqueeze(2).expand(-1, -1, 128))
    
    mse = ((remaining_frames - mean_vector.unsqueeze(1)) ** 2).mean(dim=[1, 2])

    mse_list.extend(mse.tolist())

# Calculate the mean of MSEs
overall_mean_mse = np.mean(mse_list)

# Calculate expected InfoNCE
expected_infoNCE = np.log(N)

print(f'Expected MSE: {overall_mean_mse}')
print(f'Expected InfoNCE: {expected_infoNCE}')