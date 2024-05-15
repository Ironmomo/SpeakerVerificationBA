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
pretrained_model='/home/fassband/ba/SpeakerVerificationBA/finetuning/exp/finetuned-20240514-111049-original-base-f128-t2-b128-lr1e-4-m390-finetuning_avg-asli/models/best_audio_model.pth'

num_workers = 16

n_class = 527 # embedding size

audio_conf = {'num_mel_bins': num_mel_bins, 'target_length': target_length, 'freqm': freqm, 'timem': timem, 'mixup': mixup, 'dataset': dataset,
              'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':None, 'shuffle_frames':shuffle_frames, 'finetuning':True}


# Init Dataloader
dataset = dataloader.AudioDataset(tr_data, label_csv=label_csv, audio_conf=audio_conf)

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

print(f"Size dataset: {len(test_loader)}")

# Init model
audio_model = ASTModel(label_dim=n_class, fshape=fshape, tshape=tshape, fstride=fstride, tstride=tstride,
                input_fdim=num_mel_bins, input_tdim=target_length, model_size=model_size, pretrain_stage=False,
                load_pretrained_mdl_path=pretrained_model).to(device=DEVICE)

#if not isinstance(audio_model, torch.nn.DataParallel):
#    audio_model = torch.nn.DataParallel(audio_model)
 
    
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
        
        output = model(audio_input, 'finetuning_avg', mask_patch=mask_patch, cluster=cluster)
        
        embeddings.extend(output)
        label_embeddings.extend(labels)
        
    return embeddings, label_embeddings

def calc_distance(v1, v2):

    #dist = torch.norm(v1 - v2).item()
    dist = F.cosine_similarity(v1,v2, dim=0).item()
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
    acc_list = []
    best_t = 0
    val_best_t = 100
    closest = float('inf')
    
    print("Check treshold")
    for t in np.arange(0.0, 1.01, 0.01):
        tp, tn, fp, fn = get_classification(embeddings, labels, t)
        fp_list.append((fp / (tp + tn + fp + fn)) * 100)
        fn_list.append((fn / (tp + tn + fp + fn)) * 100)
        t_list.append(t)

        if abs(fp - fn) < closest:
            best_t = t
            val_best_t = min(fp,fn)
            closest = abs(fp - fn)
        acc = (tp + tn) / (tp + tn + fp + fn) * 100
        acc_list.append(acc)
    
    print(f'Best Treshold: {best_t}')
    print(f'Best Treshold EER: {val_best_t}')
    print(f"Best Accuracy: {max(acc_list)}")
        
    plt.figure()
    plt.plot(t_list, fp_list, label='False Positive')
    plt.plot(t_list, fn_list, label='False Negative')
    plt.xlabel('Treshold')
    plt.ylabel('Error Rate in %')
    plt.title('EER - Equal Error Rate')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(t_list, acc_list)
    plt.xlabel('Cosine Similarity - Treshold')
    plt.ylabel('Accuracy in %')
    plt.title('Accuracy')
    plt.show()

eer_plot(audio_model, test_loader)
