#!/bin/bash

set -x
export TORCH_HOME=../../pretrained_models
mkdir -p exp
mkdir -p slurm_log

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

task=pretrain_joint
mask_patch=400

# audioset and librispeech
dataset=asli
tr_data=/home/bosfab01/SpeakerVerificationBA/data/audioset2M_librispeech960.json
te_data=/home/bosfab01/SpeakerVerificationBA/data/audioset_eval.json
dataset_mean=-3.6925695
dataset_std=4.020388
target_length=1024
num_mel_bins=128

model_size=base
# no patch split overlap
fshape=128
tshape=2
fstride=${fshape}
tstride=${tshape}
# no class balancing as it implicitly uses label information
bal=none
batch_size=24
lr=1e-4
# learning rate decreases if the pretext task performance does not improve on the validation set
lr_patience=2
epoch=9
# no spectrogram masking
freqm=0
timem=0
# no mixup training
mixup=0

exp_dir=./exp/mask01-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}

CUDA_CACHE_DISABLE=1 python3 -W ignore run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv /home/bosfab01/SpeakerVerificationBA/data/label_information.csv \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${task} --lr_patience ${lr_patience} --epoch_iter 4000