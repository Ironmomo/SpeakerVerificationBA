import os, sys

sys.path.append(os.getcwd())
sys.path.append('/home/fassband/ba/SpeakerVerificationBA')

import dataloader
import torch
import torch.nn.functional as F
from ssast_model import ASTModel

import matplotlib.pyplot as plt
import numpy as np
from finetuning.utilities import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

task='finetuning_avg_v1'
mask_patch=390 # 390 would be more similar to original paper (because we habe 998 instead of 1024 targetlength)

dataset='asli' # audioset and librispeech
tr_data='/home/fassband/ba/SpeakerVerificationBA/data/eer3/augmentation.json'
label_csv='/home/fassband/ba/SpeakerVerificationBA/data/eer3/augmentation.csv'
dataset_mean=-7.1876974
dataset_std=4.2474914
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
shuffle_frames=True
# how often should model be evaluated on the validation set and saved
epoch_iter=1000
# how often should loss and statistics be printed
n_print_steps=100

# set pretrained model
pretrained_model='/home/fassband/ba/SpeakerVerificationBA/finetuning/exp/finetuned-20240529-174239-original-base-f128-t2-b128-lr1e-4-m390-finetuning_avg_v1-asli/models/best_audio_model.pth'

num_workers = 16

n_class = 527 # embedding size

audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mixup': mixup, 'dataset': dataset,
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':None, 'shuffle_frames':shuffle_frames, 'finetuning':True}


# Init Dataloader
dataset = dataloader.AudioDataset(tr_data, label_csv=label_csv, audio_conf=audio_conf)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True)


# Init model
audio_model = ASTModel(label_dim=n_class, fshape=fshape, tshape=tshape, fstride=fstride, tstride=tstride,
                input_fdim=num_mel_bins, input_tdim=target_length, model_size=model_size, pretrain_stage=False,
                load_pretrained_mdl_path=pretrained_model).to(device=DEVICE)

cluster = (num_mel_bins != fshape)

positiv_similarity = AverageMeter()
negativ_similarity = AverageMeter()
size = len(test_loader)


def calc_distance(v1, v2):
    dist = F.cosine_similarity(v1,v2)
    return dist


for i, (audio_input, audio_input_two, label) in enumerate(test_loader):
        batch_size = len(audio_input)
        print(f"[{i*batch_size}/{size*batch_size}]")

        output = audio_model(audio_input, task, mask_patch=mask_patch, cluster=cluster)
        output_two = audio_model(audio_input_two, task, mask_patch=mask_patch, cluster=cluster)
        target_pos = torch.ones(batch_size)

        pos_cos = calc_distance(output, output_two)

         # shuffle two
        perm = torch.randperm(batch_size)
        output_two_shuffle = output_two[perm]
        
        label_shuffle = label[perm]
        # Check if tensors are equal element-wise
        equal_mask = torch.eq(label, label_shuffle)
    
        # Convert boolean mask to 1s and 0s
        equal_mask = equal_mask.float()

        # Sum along the second dimension to get a [B, 1] mask
        equal_mask = torch.sum(equal_mask, dim=1, keepdim=True)
        target_neg = torch.where(equal_mask == label.size(1), torch.ones_like(equal_mask), -torch.ones_like(equal_mask))
        target_neg = torch.squeeze(target_neg)
        
        neg_cos = calc_distance(output[target_neg == -1], output_two_shuffle[target_neg == -1])
        
        pos_cos = torch.mean(pos_cos)
        neg_cos = torch.mean(neg_cos)
        
        print(f"Positive Similarity: {pos_cos}")
        print(f"Negative Similarity: {neg_cos}")
        
        positiv_similarity.update(pos_cos)
        negativ_similarity.update(neg_cos)


print(f"Positive Similarity: {positiv_similarity.avg}")
print(f"Negative Similarity: {negativ_similarity.avg}")


