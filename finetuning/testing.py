import os, sys

sys.path.append(os.getcwd())
sys.path.append('/home/fassband/ba/SpeakerVerificationBA')

import dataloader
import torch
import torch.nn.functional as F
from ssast_model import ASTModel

import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

task='finetuning_avg'
mask_patch=390 # 390 would be more similar to original paper (because we habe 998 instead of 1024 targetlength)

dataset='asli' # audioset and librispeech
tr_data='/home/fassband/ba/SpeakerVerificationBA/data/eer2/augmentation.json'
label_csv='/home/fassband/ba/SpeakerVerificationBA/data/eer2/augmentation.csv'
dataset_mean=-6.5975285
dataset_std=3.6734943
target_length=998 # (10000ms - (25ms - 10ms)) // 10ms = 998
num_mel_bins=128

model_size='base'
# no patch split overlap
fshape=128
tshape=2
fstride=fshape
tstride=tshape
# no class balancing as it implicitly uses label information
bal='none'
lr=1e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0

epoch=10
batch_size=48

# shuffle frames in the spectrogram in random order
shuffle_frames="False"
# how often should model be evaluated on the validation set and saved
epoch_iter=1000
# how often should loss and statistics be printed
n_print_steps=100

# set pretrained model
#pretrained_model='/home/fassband/ba/SpeakerVerificationBA/pretraining/exp/pretrained-20240501-162648-original-base-f128-t2-b48-lr1e-4-m390-pretrain_joint-asli/models/best_audio_model.pth'
pretrained_model='/home/fassband/ba/SpeakerVerificationBA/finetuning/exp/finetuned-20240514-004009-original-base-f128-t2-b48-lr1e-4-m390-finetuning_avg-asli/models/best_audio_model.pth'

num_workers = 16

n_class = 527 # embedding size

audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mixup': mixup, 'dataset': dataset,
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':None, 'shuffle_frames':shuffle_frames, 'finetuning':True}


# Init Dataloader
dataset = dataloader.AudioDataset(tr_data, label_csv=label_csv, audio_conf=audio_conf)


# Init model
audio_model = ASTModel(label_dim=n_class, fshape=fshape, tshape=tshape, fstride=fstride, tstride=tstride,
                input_fdim=num_mel_bins, input_tdim=target_length, model_size=model_size, pretrain_stage=False,
                load_pretrained_mdl_path=pretrained_model).to(device=DEVICE)

cluster = (num_mel_bins != fshape)

inp1, inp1_2, lab1 = dataset.__getitem__(0)
inp2, inp2_2, lab2 = dataset.__getitem__(30)
inp3, inp3_2, lab3 = dataset.__getitem__(149)

inp1 = inp1.unsqueeze(0)

inp2 = inp2.unsqueeze(0)

inp3 = inp3.unsqueeze(0)


o1 = audio_model(inp1, 'finetuning_avg', mask_patch=mask_patch, cluster=cluster)
o2 = audio_model(inp2, 'finetuning_avg', mask_patch=mask_patch, cluster=cluster)
o3 = audio_model(inp3, 'finetuning_avg', mask_patch=mask_patch, cluster=cluster)

def calc_distance(v1, v2):
    dist = F.cosine_similarity(v1,v2).item()
    return dist

print(calc_distance(o1,o2))
print(calc_distance(o1,o3))