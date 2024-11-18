import librosa
import librosa.display as ld
import numpy as np
import scipy.signal as signal
import os
import time
import matplotlib.pyplot as plt
import math
plt.rcParams['figure.figsize'] = (14,6)

arquivo_entrada = (r'E:\AI Python\Audio\wav\ticket_audio_bruto_cortado_2minutos.wav')

#arquivo_entrada, sr = librosa.load(librosa.ex('trumpet'))

audio, sr = librosa.load(data = arquivo_entrada, sr = 16000)

onset_env = librosa.onset.onset_strength(y = audio, sr = sr, max_size=5)

onset_env.shape, type(onset_env)


times = librosa.times_like(onset_env, sr = sr)
times.shape, type(times)

onset_frames = librosa.onset.onset_detect(onset_envelope= onset_env, sr = sr)
#print (onset_frames)

onset_times = librosa.onset.onset_detect(onset_envelope= onset_env, sr= sr, units= 'time')

#plt.plot(times, onset_env, label = 'Onset strength')
#plt.vlines(times[onset_frames], 0, onset_env.max(), color = 'r', linestyle = '--', label = 'Onsets')
#plt.legend()


y_clicks = librosa.clicks(times=onset_times, length= len(audio), sr = sr)
audio(data = audio + y_clicks, rate = sr)
'''
# Extraindo mfcc
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
mfccs_db = librosa.amplitude_to_db(abs(mfccs))

# Fazendo a normalização de mfcc
mfccs_norm = np.mean(mfccs_db.T, axis=0)

#print(mfccs.shape, type(mfccs))
#ld.specshow(mfccs_norm, sr = sr, x_axis= 'time', cmap = 'Spectral')
plt.plot(mfccs_norm)
plt.show()
'''