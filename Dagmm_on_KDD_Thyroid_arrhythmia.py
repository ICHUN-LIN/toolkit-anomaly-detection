import keras
from keras.layers import Activation, Dense, Input, Dropout
from keras.layers import Conv2D, Flatten ,MaxPooling2D, BatchNormalization
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.models import Model
from toolkit_prototype import toolkit
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import math
from keras.datasets import mnist, cifar10
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn
from Dagmm.dataset import gen_KDD_train_valid_data, gen_Thyroid_train_valid_data, gen_arrhythmia_train_valid_data, gen_mist_train_valid_data, gen_cifar10_train_valid_data

def test_KDD_on_Dagmm():
    tool = toolkit()
    #set options
    options = tool.setAlgorthem("DAGMM")
    options.datasetname = "kdd"
    X_train, X_test, y_train, y_test = gen_KDD_train_valid_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    minmaxscaler = sklearn.preprocessing.MinMaxScaler().fit(X_train)
    X_train = minmaxscaler.transform(X_train)

    minmaxscaler = sklearn.preprocessing.MinMaxScaler().fit(X_test)
    X_test = minmaxscaler.transform(X_test)

    options.X_test = X_test
    options.X_train = X_train
    options.y_train = y_train
    options.y_test = y_test
    options.comp_hidden = [60, 30, 10, 1, 10, 30, 60]
    options.mid_layer = 3
    options.epoch = 80
    options.batch_size = 4096
    tool.train(options)
    tool.test()

def test_Thyroid_on_Dagmm():
    tool = toolkit()
    #set options
    options = tool.setAlgorthem("DAGMM")
    options.datasetname = "Thyroid"
    X_train, X_test, y_train, y_test = gen_Thyroid_train_valid_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    options.X_test = X_test
    options.X_train = X_train
    options.y_train = y_train
    options.y_test = y_test
    options.comp_hidden = [14,10,14]
    options.mid_layer = 1
    options.mix_components = 2
    options.epoch = 100
    options.normal_portion = 97.5
    options.batch_size = 1024
    tool.train(options)
    tool.test()

def test_arrhythmia_on_Dagmm():
    tool = toolkit()
    #set options
    options = tool.setAlgorthem("DAGMM")
    options.datasetname = "Thyroid"
    X_train, X_test, y_train, y_test = gen_arrhythmia_train_valid_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    options.X_test = X_test
    options.X_train = X_train
    options.y_train = y_train
    options.y_test = y_test
    options.comp_hidden = [10,2,10]
    options.mid_layer = 1
    options.mix_components = 2
    options.epoch = 15000
    options.batch_size = 500
    options.normal_portion = 85
    tool.train(options)
    tool.test()

def test_Thyroid_on_Dagmm_with_customed_net_structure():
    tool = toolkit()
    #set options
    options = tool.setAlgorthem("DAGMM")
    options.datasetname = "Thyroid"
    X_train, X_test, y_train, y_test = gen_Thyroid_train_valid_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    options.X_test = X_test
    options.X_train = X_train
    options.y_train = y_train
    options.y_test = y_test

    # set compression model
    input_shape = (6,)
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(14, activation=None)(inputs)
    x = Activation('tanh')(x)
    mid_presentation = Dense(10, activation=None)(x)
    x = Dense(14, activation=None)(mid_presentation)
    x = Activation('tanh')(x)
    y = Dense(6, name='compression_out_layer',activation=None)(x)

    options.compress_net_inputs = inputs
    options.compress_net_mid_presentation = mid_presentation
    options.compress_net_outputs = y
    
    # set estimated model
    z = options.create_estimate_net_z_layer(inputs,mid_presentation,y)
    layer = Dense(10, activation=None)(z)
    layer = Activation('tanh')(layer)
    layer = Dropout(0.5, noise_shape=None, seed=None)(layer)
    gamma = Dense(2, activation='softmax', name='gamma')(layer)
    
    options.z_layer = z
    options.gamma_layer = gamma

    options.epoch = 100
    options.normal_portion = 97.5
    options.batch_size = 1024
    tool.train(options)
    tool.test()
    
test_KDD_on_Dagmm()
#test_Thyroid_on_Dagmm()
#test_arrhythmia_on_Dagmm()
#test_Thyroid_on_Dagmm_with_customed_net_structure()
