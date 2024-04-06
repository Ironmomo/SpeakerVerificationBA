# SpeakerVerificationBA

## Introduction

This GitHub repository is dedicated to exploring the application of transformer networks in the domain of Automatic Speaker Recognition (ASR). The project focuses on investigating whether transformer architectures can effectively capture and learn voice dynamics for speaker identification tasks.

## Getting Started
### Init
1. Clone the Repo:
```
git clone https://github.com/Ironmomo/SpeakerVerificationBA.git
git lfs fetch
```
2. It is recommended to use a venv:
```
python3 -m venv myEnv
source myEnv/bin/activate
```
3. Install requirements:
```
pip install -r requirements.txt
```

## Data Handling
### Data Loading
We provide a script to download the librispeech dataset or the voxceleb 1&2 dataset. You can use the same script just specify *voxceleb* or *librispeech* as argument.\
e.g.  ```./data/load_data.sh voxceleb # Downloads voxceleb dataset 1&2```\
The dataset will be stored at *data/<dataset>/* where every subdirectory contains audiofiles of one specific speaker.

### Data Preprocessing
We provide you with a script to preprocess the data for the model. As suggested in the SSAST paper the data will be cut/padded to .flac Audiosamples of 10 sec with a sampling rate of 16000. Beside that an *Augmentation.csv* gets created which keeps the path of the preprocessed audiofile and the speaker_id.
The script allows you to specify the path of the dataset relativ to *data/*. Keep in mind that the path provided should be the directory containing the speaker directories. If you are using our Data loading scrip it is straight forward.\
You have to set the destination directory for the preprocessed data. This allows you to merge different datasets into one directory which can then be further used to create an instance of Audio_Dataset.
e.g. 
```
# Download librispeech
./data/load_data.sh librispeech
# Preprocess librispeech and save preprocessed data to data/preprocessed
python3 utils/Data_preprocess -d librispeech -n preprocessed
```

### Load Data for Training and Evaluation
We are using Pytorch. Therefore we created a class called Audio_Dataset which inherits from Dataset. Create an instance of Audio_Dataset to provide the model with data. The class has been implemented in *utils/Audio_Dataset.py*
To create an instance provide the augmentation file which has been created when preprocessing the data.
