#  Original Authors: Yuan Gong, 2021, Massachusetts Institute of Technology
#  Edited by: Andrin Fassbind, Fabian Bosshard, 2024, Zurich University of Applied Sciences

import torch
import numpy as np
import os, sys


sys.path.append(os.getcwd()) # Add the parent directory to sys.path
import dataloader


# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 998, 'freqm': 0, 'timem': 0, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset_librispeech'}

train_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset('data/finetuning/augmentation.json', 
                                   label_csv='data/finetuning/augmentation.csv',
                                   audio_conf=audio_conf), batch_size=1000, shuffle=False, num_workers=32, pin_memory=True)
mean=[]
std=[]
for i, (audio_input, labels) in enumerate(train_loader):
    cur_mean = torch.mean(audio_input)
    cur_std = torch.std(audio_input)
    mean.append(cur_mean)
    std.append(cur_std)
    print(cur_mean, cur_std)
print(np.mean(mean), np.mean(std))