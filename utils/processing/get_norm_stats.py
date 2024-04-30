# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

import dataloader_ast

# set skip_norm as True only when you are computing the normalization stats
audio_conf = {'num_mel_bins': 128, 'target_length': 998, 'freqm': 0, 'timem': 0, 'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'audioset_librispeech'}

train_loader = torch.utils.data.DataLoader(
    dataloader_ast.AudiosetDataset('/home/bosfab01/SpeakerVerificationBA/data/audioset2M_librispeech960.json', 
                                   label_csv='/home/bosfab01/SpeakerVerificationBA/data/label_information.csv',
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