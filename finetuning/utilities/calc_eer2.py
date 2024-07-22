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

task='finetuning_avg_v1'
mask_patch=390 # 390 would be more similar to original paper (because we habe 998 instead of 1024 targetlength)

dataset='asli' # audioset and librispeech
tr_data='/home/fassband/ba/SpeakerVerificationBA/data/eer3/augmentation.json'
label_csv='/home/fassband/ba/SpeakerVerificationBA/data/eer3/augmentation.csv'
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
shuffle_frames=False
# how often should model be evaluated on the validation set and saved
epoch_iter=1000
# how often should loss and statistics be printed
n_print_steps=100

# set pretrained model
pretrained_model='/home/fassband/ba/SpeakerVerificationBA/finetuning/exp/finetuned-20240514-111049-original-base-f128-t2-b128-lr1e-4-m390-finetuning_avg-asli/models/best_audio_model.pth'

num_workers = 16

n_class = 527 # embedding size

audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mixup': mixup, 'dataset': dataset,
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':None, 'shuffle_frames':shuffle_frames, 'finetuning':True}


# Init Dataloader
dataset = dataloader.AudioDataset(tr_data, label_csv=label_csv, audio_conf=audio_conf)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

print(f"Size dataset: {len(test_loader)}")

# Init model
audio_model = ASTModel(label_dim=n_class, fshape=fshape, tshape=tshape, fstride=fstride, tstride=tstride,
                input_fdim=num_mel_bins, input_tdim=target_length, model_size=model_size, pretrain_stage=False,
                load_pretrained_mdl_path=pretrained_model).to(device=DEVICE)

#if not isinstance(audio_model, torch.nn.DataParallel):
#    audio_model = torch.nn.DataParallel(audio_model)
 
# Configure Plot
# Setting global parameters for Matplotlib to use LaTeX rendering
plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX rendering
    'text.latex.preamble': r'\usepackage{lmodern}\usepackage{amsmath}',  # Use Latin Modern font and include amsmath
    'font.family': 'serif',  # Use serif font for consistency with LaTeX document
    'font.serif': ['Latin Modern Roman'],  # Specify Latin Modern Roman
    'pdf.fonttype': 42,  # Ensures fonts are embedded as TrueType
    'savefig.dpi': 400,  # Lower DPI setting for non-text elements
    'font.size': 11,  # Adjust font size to match document (you may need to tweak this)
    'axes.labelsize': 9.0,  # Size of the x and y labels
    'axes.titlesize': 11,  # Size of the plot title
    'xtick.labelsize': 7.5,  # Size of the x-axis tick labels
    'ytick.labelsize': 7.5,  # Size of the y-axis tick labels
    'legend.fontsize': 9,  # Size of the legend font
    'figure.titlesize': 12.0  # Size of the figure's main title if any
})
    
def main(model, dataloader):
    cluster = (num_mel_bins != fshape)
    audio_model.eval()
    size = len(dataloader)
    
    data = []
    
    for i, (audio_input, audio_input_two, labels) in enumerate(dataloader):
        print(f"[{i*batch_size}/{size*batch_size}]")
        # Calc embeddings
        output = model(audio_input, 'finetuning_avg', mask_patch=mask_patch, cluster=cluster)
        output_two = model(audio_input_two, 'finetuning_avg', mask_patch=mask_patch, cluster=cluster)
        
        # Make Classification
        for t in np.arange(0.0, 1.02, 0.01):
            tp, tn, fp, fn = get_classification(output, output_two, labels, t)
            data.append([tp,tn,fp,fn,t])
            
    # Calc Performance
    data_t = torch.tensor(data)
    
    acc_t = calc_acc(data_t)
    
    fp_t = get_mean_val_by_treshold(data_t, 2)
    fn_t = get_mean_val_by_treshold(data_t, 3)
    
    # Extract the labels (last column)
    t = data_t[:, -1]
    unique_t = t.unique()
    
    eer_plot(acc_t, fp_t, fn_t, unique_t)
    
    
       
def calc_acc(data):
    # Extract the labels (last column)
    t = data[:, -1]

    # Find unique labels
    unique_t = t.unique()

    # Create a tensor to store the result for each unique t
    acc_t = torch.zeros_like(unique_t, dtype=torch.float32)

    # Iterate over each unique t, perform the calculation, and store the result
    for i, label in enumerate(unique_t):
        # Select rows where the label matches the current label and columns 0-3
        d = data[t == label, :4]
        
        # Perform the calculation
        acc = (d[:, 0] + d[:, 1]) / (d[:, 0] + d[:, 1] + d[:, 2] + d[:, 3])
        # Store the mean result in acc_t
        acc_t[i] = acc.mean()
        
    return acc_t


def get_mean_val_by_treshold(data, col):
    # Extract the labels (last column)
    t = data[:, -1]

    # Find unique labels
    unique_t = t.unique()

    # Create a tensor to store the result for each unique t
    val_t = torch.zeros_like(unique_t, dtype=torch.float32)

    # Iterate over each unique t, perform the calculation, and store the result
    for i, label in enumerate(unique_t):
        # Select rows where the label matches the current label and columns 0-3
        d = data[t == label, col]

        # Store the mean result in acc_t
        val_t[i] = d.mean()
    
    return val_t
    

def calc_distance(v1, v2):
    dist = F.cosine_similarity(v1, v2, dim=0).item()
    return dist


def get_classification(embeddings, embeddings_two, labels, treshold):
    true_positive = 0
    true_negative = 0
    
    false_positive = 0
    false_negative = 0
    
    for idx in range(len(embeddings) - 1):
        v1_1 = embeddings[idx]
        v1_2 = embeddings_two[idx]
        v2_1 = embeddings[idx+1]
        
        tp1, tn1, fp1, fn1 = calc_class(v1_1, v1_2, True, treshold)
        tp2, tn2, fp2, fn2 = calc_class(v1_1, v2_1, torch.equal(labels[idx], labels[idx + 1]), treshold)

        true_positive += tp1
        true_positive += tp2
        true_negative += tn1
        true_negative += tn2
        false_positive += fp1
        false_positive += fp2
        false_negative += fn1
        false_negative += fn2       
                
    
    return true_positive, true_negative, false_positive, false_negative


def calc_class(v1, v2, same, treshold):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    # Model classify as negative
    if calc_distance(v1, v2) < treshold:
        # Embeddings are positive
        if same:
            fn += 1
        # Embeddings are negative
        else:
            tn += 1
            
    # Model classify as positive
    else:
        # Embeddings are negative
        if same:
            fp += 1
        # Embeddings are negative
        else:
            tp += 1
            
    return tp, tn, fp, fn
 
                        
def eer_plot(acc, fp, fn, unique_t):
    best_t_idx = abs(fp - fn).argmin()
    best_t = unique_t[best_t_idx]
    eer = (fp[best_t_idx] + fn[best_t_idx]) / 2
    
    print(f'Best Treshold: {best_t}')
    print(f'EER: {eer}')
    print(f"Best Accuracy: {acc.max()}")
        
    plt.figure()
    plt.plot(unique_t, fp, label='False Positive')
    plt.plot(unique_t, fn, label='False Negative')
    plt.xlabel('Treshold')
    plt.ylabel('Error Rate in %')
    plt.title('EER - Equal Error Rate')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(unique_t, acc)
    plt.xlabel('Cosine Similarity - Treshold')
    plt.ylabel('Accuracy Score in %')
    plt.title('Accuracy Score')
    plt.show()

main(audio_model, test_loader)
