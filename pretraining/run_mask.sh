#!/bin/bash

#  Original Author: Yuan Gong, 2021, Massachusetts Institute of Technology
#  Edited by: Andrin Fassbind, Fabian Bosshard, 2024, Zurich University of Applied Sciences

set -x
SCRIPT_DIR=$(cd $(dirname "$0") && pwd) # Get the directory where the script is located
export TORCH_HOME=$SCRIPT_DIR/../pretrained_models # Set TORCH_HOME relative to the script location
mkdir -p exp
mkdir -p slurm_log

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2

task=pretrain_joint
mask_patch=390 # 390 would be more similar to original paper (because we habe 998 instead of 1024 targetlength)

dataset=asli # audioset and librispeech
tr_data=$SCRIPT_DIR/../data/audioset2M_librispeech960.json
te_data=$SCRIPT_DIR/../data/audioset_eval.json
dataset_mean=-5.0716844 
dataset_std=4.386603
target_length=998 # (10000ms - (25ms - 10ms)) // 10ms = 998
num_mel_bins=128

model_size=base
# no patch split overlap
fshape=128
tshape=2
fstride=${fshape}
tstride=${tshape}
# no class balancing as it implicitly uses label information
bal=none
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
epoch_iter=4000
# how often should loss and statistics be printed
n_print_steps=100


# Create the experiment directory
current_time=$(date "+%Y%m%d-%H%M%S")
if [[ $shuffle_frames == "True" ]]; then
    exp_dir=$SCRIPT_DIR/exp/pretrained-${current_time}-shuffled-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
else
    exp_dir=$SCRIPT_DIR/exp/pretrained-${current_time}-original-${model_size}-f${fshape}-t${tshape}-b$batch_size-lr${lr}-m${mask_patch}-${task}-${dataset}
fi

CUDA_CACHE_DISABLE=1 python3 -W ignore run.py --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv $SCRIPT_DIR/../data/label_information.csv \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps ${n_print_steps} \
--task ${task} --lr_patience ${lr_patience} --epoch_iter ${epoch_iter} --shuffle_frames ${shuffle_frames} \