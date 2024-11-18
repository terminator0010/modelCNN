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

