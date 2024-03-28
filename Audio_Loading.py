import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from IPython.display import Audio

# Load audio file
# array, sampling_rate = librosa.load(librosa.ex("trumpet"))
audio_file = '/teamspace/studios/this_studio/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac'
array, sampling_rate = librosa.load(audio_file)


#################
#     Wave      #
#################
plt.figure().set_figwidth(12)
librosa.display.waveshow(array, sr=sampling_rate, color="blue")

#################
#  Spectogramm  #
#################

# Short time fourier transformation
D = librosa.stft(array)
# Convert amplitude spectogram to db scale
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Plot 
plt.figure().set_figwidth(12)
librosa.display.specshow(S_db, x_axis="time", y_axis="hz")
plt.colorbar()


#####################
#  Mel-Spectogramm  #
#####################

# Transform Mel Spectogramm
S = librosa.feature.melspectrogram(y=array, sr=sampling_rate, n_mels=128, fmax=8000)
S_dB = librosa.power_to_db(S, ref=np.max)

# Plot
plt.figure().set_figwidth(12)
librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sampling_rate, fmax=8000)
plt.colorbar()

#####################
#  Play Audio  #
#####################

def play_audio(audio_file):
    try:
        # Display the audio player
        return Audio(audio_file)
    except Exception as e:
        print("Error playing audio:", e)
        return None

play_audio(audio_file)
