import geopandas as gpd
import numpy as np
import itertools
from PIL import Image
import io
import matplotlib.pyplot as plt

# fig2img
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    #return buf
    img = Image.open(buf)
    return img

def render(frames, image_file = 'FileName', duration = 100):
    images = frames
    # loop=0: loop forever, duration=1: play each frame for 1ms
    images[0].save(
        f"{image_file}.gif", format = 'GIF', save_all=True, append_images=images[1:], loop=0, duration=duration, transparency=0, disposal = 2)

#%%
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.ma as ma
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow.keras.preprocessing as prep
from sklearn.model_selection import train_test_split
from model import AugementedConvLSTM
import configparser
import argparse
import h5py
import itertools

#%%
def load_dataset(model = None, mon = False):
    if mon:
        Y = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/Y_mon.npy")
        try:
            X = np.load(rf"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/{model}_mon.npy")
        except:
            print("Models available: MIROC-ESM, CanESM2, HadGEM2-ES, GFDL-CM3")
            return None, None
    else:
        if model == 'MIROC-ESM':
            X = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/MIROC-ESM.npy")
            Y = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/Y.npy")
        elif model == 'HadGEM2-ES':
            X = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/HadGEM2-ES.npy")
            Y = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/Y_360.npy")
        elif model == 'CanESM2':
            X = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/CanESM2.npy")
            Y = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/Y.npy")
        elif model == 'GFDL-CM3':
            X = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/GFDL-CM3.npy")
            Y = np.load(r"/home/uditbhatia/Documents/Sarth/Downscaling_AugmentedConvLSTM/03_Preprocess_Data/Stacked/Y.npy")
        else:
            print("Models available: MIROC-ESM, CanESM2, HadGEM2-ES, GFDL-CM3")
            return None, None
    return X,Y

def normalize(data):
    data = data - data.mean()
    data = data / data.std()
    return data
#%%
def set_data(X, Y):
    X_normalized = np.zeros((7, np.max(X.shape), 129, 135))
    for i in range(7):
        X_normalized[i,] = normalize(X[i,])

    Y_normalized = normalize(Y)

    print("Mean of GCM Data: ",X[0,].mean())
    print("Variance of GCM Data: ",X[0,].std(),end="\n")

    print("Mean of Obseved Data: ",Y.mean())
    print("Variance of Obseved Data: ",Y.std(),end="\n")

    std_observed = Y.std()
    X = X_normalized.transpose(1,2,3,0)
    Y = Y_normalized.reshape(-1,129, 135, 1)
    return X, Y, std_observed

#%%
def data_generator(X,Y):
    time_steps = 4
    batch_size1 = 1
    generator = prep.sequence.TimeseriesGenerator(
        X, 
        Y.reshape(-1, 129, 135, 1),
        length=time_steps, 
        batch_size=batch_size1
        )
    return generator

#%%
