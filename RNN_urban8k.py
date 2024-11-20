import os
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display as ld
from IPython.display import Audio
from tqdm import tqdm
import tensorflow
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from keras._tf_keras.keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()

print('Versão tensorflow: ', tensorflow.__version__)

metadata = pd.read_csv(r'C:\modelCNN-main\model\metadata\UrbanSound8K.csv')

fsID = []
classID = []
occurID = []
sliceID = []
full_path = []

for root, dirs, files in tqdm(os.walk('C:\modelCNN-main\model\audio\fold1')):
    for file in files:
        try:
            fs = int(file.split('-')[0])
            class_ =  int(file.split('-')[1])
            occur = int(file.split('-')[2])
            slice_ = file.split('-')[3]
            slice_ = int(file.split('.')[0])
            
            fsID.append(fs)
            classID.append(class_)
            occurID.append(occur)
            sliceID.append(slice_)
            
            full_path.append((root, file))
        except ValueError:
            continue    

sound_list = ['ar_condicionado','buzina_de_carro','crianca_brincado','latido_de_cachorro','perfuracao','motor_em_marcha','']
sound_dict = {em[0]:em[1] for em in enumerate(sound_list)}

df = pd.DataFrame([fsID, classID, occurID, sliceID, full_path]).T
print(df)

df.columns = ['fsID', 'classID', 'occurID', 'sliceID', 'full_path']
df['classID'] = df['classID'].map(sound_dict)
print(df['classID'])

df['path'] = df['full_path'].apply(lambda x: x[0] + '/' + x[1])

print(df['path'])

df['classID'].describe()
df['classID'].value_counts()

plt.figure(figsize=(18,7))
sns.countplot(df['classID'])
