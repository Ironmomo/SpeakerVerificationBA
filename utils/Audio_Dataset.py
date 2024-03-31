import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
import pandas as pd
import librosa
import numpy as np
import os


class AudioDataset(Dataset):
    """Audio dataset."""

    def __init__(self, csv_file, sampling_rate: int = 16000, n_mels: int = 128, fmax: int = 8000):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            sampling_rate: of the audio data
            n_mels: filter banks for the mel spectogram
            fmax: maximum frequency for mel spectogram
            
        """
        self.annotations = pd.read_csv(csv_file)
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.fmax = fmax
        
        self.label_dim = self.annotations['label'].nunique()
        self.input_tdim = self.__getitem__(0)[0].size()[1]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_file = self.annotations.iloc[idx, 0]
        
        array, _ = librosa.load(audio_file)
        
        S = librosa.feature.melspectrogram(y=array, sr=self.sampling_rate , n_mels=self.n_mels, fmax=self.fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        if not torch.is_tensor(S_dB):
            S_dB = torch.tensor(S_dB)
        
        speaker_id = self.annotations.iloc[idx, 1:]
        
        if not torch.is_tensor(speaker_id):
            speaker_id = torch.tensor(speaker_id)
            if speaker_id.dim() == 0:
                speaker_id = speaker_id.squeeze(0)

        return (S_dB, speaker_id)
   

"""
#    Example Usage   
        
root_path = os.path.join(os.getcwd(), 'data','LibriSpeech')

new_data_path = os.path.join(root_path, 'preprocessed')

AUGMENTATION_FILE = os.path.join(new_data_path, 'augmentation.csv')
    
dataset = AudioDataset(AUGMENTATION_FILE)

train_data, test_data = random_split(dataset, (0.8, 0.2))

train_sampler = RandomSampler(train_data)

dataloader = DataLoader(
        train_data,
        batch_size=10,
        sampler=train_sampler
    )



current = 0
size = len(dataloader.dataset)
for batch, (audio, label) in enumerate(dataloader):
    label = label.squeeze(dim = 1)
    current = (batch + 1) * len(audio)
    print(f"{current}/{size} Audio Shape: {audio.size()} Label Shape: {label.size()}")
"""