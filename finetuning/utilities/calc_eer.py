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
dataset_mean=-6.4959574
dataset_std=3.421145
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
    batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

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
    
def get_embeddings(model, dataloader):
    embeddings = []
    label_embeddings = []
    cluster = (num_mel_bins != fshape)
    audio_model.eval()
    size = len(dataloader)
    for i, (audio_input, audio_input_two, labels) in enumerate(dataloader):
        print(f"[{i*batch_size}/{size*batch_size}]")
        #inp = torch.cat((audio_input, audio_input_two), dim=0).to(DEVICE)
        #lab = torch.cat((labels, labels), dim=0).to(DEVICE)
        
        output = model(audio_input, task, mask_patch=mask_patch, cluster=cluster)
        
        embeddings.extend(output)
        label_embeddings.extend(labels)
        
    return embeddings, label_embeddings

def calc_distance(v1, v2):

    #dist = torch.norm(v1 - v2).item()
    dist = F.cosine_similarity(v1, v2, dim=0).item()
    return dist

def get_classification(embeddings, labels, treshold):
    true_positive = 0
    true_negative = 0
    
    false_positive = 0
    false_negative = 0
    
    for v1_idx in range(len(embeddings)):
        for v2_idx in range(len(embeddings)):
            if v1_idx != v2_idx:
                # Model classify as negative
                if calc_distance(embeddings[v1_idx], embeddings[v2_idx]) < treshold:
                    # Embeddings are positive
                    if torch.equal(labels[v1_idx], labels[v2_idx]):
                        false_negative += 1
                    # Embeddings are negative
                    else:
                        true_negative += 1
                        
                # Model classify as positive
                else:
                    # Embeddings are negative
                    if not torch.equal(labels[v1_idx], labels[v2_idx]):
                        false_positive += 1
                    # Embeddings are negative
                    else:
                        true_positive += 1
    
    return true_positive, true_negative, false_positive, false_negative
                        
def eer_plot(model, dataloader):
    print("Calc embeddings")
    embeddings, labels = get_embeddings(model, dataloader)
    fp_list = []
    fn_list = []
    t_list  = []
    ac_list = []
    best_t = 0
    val_best_t = 100
    closest = float('inf')
    
    print("Check treshold")
    for t in np.arange(0.0, 1.0, 0.1):
        tp, tn, fp, fn = get_classification(embeddings, labels, t)
        fp_list.append((fp / (tp + tn + fp + fn)) * 100)
        fn_list.append((fn / (tp + tn + fp + fn)) * 100)
        t_list.append(t)

        if abs(fp - fn) < closest:
            best_t = t
            val_best_t = ((fp + fn)/2) / (tp + tn + fp + fn) * 100
            closest = abs(fp - fn)
        acc = (tp + tn) / (tp + tn + fp + fn) * 100
        ac_list.append(acc)
    
    print(f'Best Treshold: {best_t}')
    print(f'Best Treshold EER: {val_best_t}')
    print(f"Best Accuracy: {max(ac_list)}")
        
    plt.figure()
    plt.plot(t_list, fp_list, label='False Positive')
    plt.plot(t_list, fn_list, label='False Negative')
    plt.xlabel('Treshold')
    plt.ylabel('Error Rate in %')
    plt.title('EER - Equal Error Rate')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(t_list, ac_list)
    plt.xlabel('Cosine Similarity - Treshold')
    plt.ylabel('Accuracy Score in %')
    plt.title('Accuracy Score')
    plt.show()

eer_plot(audio_model, test_loader)

