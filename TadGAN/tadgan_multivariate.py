# -*- coding: utf-8 -*-
"""TadGAN_multivariate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ISPbyx301vQy6kOnM4GUaJESMtcmD5LY
"""

! pip install orion-ml
! pip install 'urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1'


! git clone https://github.com/signals-dev/Orion.git
! mv Orion/tutorials/tulog/* .

# general imports 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#importing sklearn module
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
#utils.py contains all the plot function.

from utils2 import plot, plot_ts, plot_rws, plot_error, unroll_ts

signal = pd.read_csv('Mining Process mean EX1, EX2, EX3 Final.csv')

signal

signal['date'] = pd.to_datetime(signal['date'])

signal['ts'] = signal.date.values.astype(np.int64) // 10 ** 9

"""multi variate"""

signal.columns = ['date', '0', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','ts']

signal

signal[['ts','0', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','date']]

signal = signal.drop(['date'], axis=1)
signal

signal = signal[['ts', '0', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']]

signal.rename(columns={'ts':'timestamp'}, inplace=True)
signal

signal.rename(columns={'ts':'timestamp'}, inplace=True)
signal.rename(columns={'Silica Concentrate':'value'}, inplace=True)
signal

df=signal
df.to_csv("TadGAN.csv", index=False, encoding="utf-8-sig")

"""LSTM"""

from orion import Orion
filename = "TadGAN.csv"
from orion.data import load_signal, load_anomalies
data21 = load_signal(filename)

hyperparameters = {
    "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences#1": {
        'target_column': 0 
    },
    'keras.Sequential.LSTMTimeSeriesRegressor#1': {
        'epochs': 5,
        'verbose': True
    }
}

orion = Orion(
    pipeline='lstm_dynamic_threshold',
    hyperparameters=hyperparameters
)

orion.fit(data21)

orion.detect(data21)

"""LSTM_AutoEncoder"""

hyperparameters = {
    "mlprimitives.custom.timeseries_preprocessing.rolling_window_sequences#1": {
        "window_size": 100,
        "target_column": 0 
    },
    'keras.Sequential.LSTMSeq2Seq#1': {
        'epochs': 5,
        'verbose': True,
        'input_shape': [100, 26],
        'target_shape': [100, 1],
    }
}

orion = Orion(
    pipeline='lstm_autoencoder',
    hyperparameters=hyperparameters
)

orion.fit(data21)

orion.detect(data21)

"""TadGAN"""

hyperparameters = {
    "mlprimitives.custom.timeseries_preprocessing.time_segments_aggregate#1": {
        "interval": 3600
    },

    'orion.primitives.tadgan.TadGAN#1': {
        'epochs': 5,
        'verbose': True,
        'input_shape': [100, 26]
    }
}
orion = Orion(
    pipeline='tadgan.json',
    hyperparameters=hyperparameters
)
orion.fit(data21)

orion.detect(data21)
