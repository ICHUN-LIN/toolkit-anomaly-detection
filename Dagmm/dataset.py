#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import math
from keras.datasets import mnist, cifar10
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def gen_mist_train_valid_data(classnum:int):
        (x_train, train_label), (x_test, test_label) = mnist.load_data()

        #abnormal:1 , normal:0
        i =0
        for t in test_label:
            if t == classnum:
               test_label[i] = 1
            else:
               test_label[i] = 0
            i =i+1
        
        #abnormal:1 , normal:0
        i =0
        for t in train_label:
            if t == classnum:
               train_label[i] = 1
            else:
               train_label[i] = 0
            i =i+1
        
        Image_size = x_train.shape[1]
        reshape_x_train = np.reshape(x_train, [-1, Image_size*Image_size])
        reshape_x_test = np.reshape(x_test, [-1, Image_size*Image_size])
        reshape_x_train = reshape_x_train[train_label==0]
        reshape_x_test = reshape_x_test/255
        reshape_x_train = reshape_x_train/255
        return reshape_x_train, reshape_x_test, train_label, test_label

def gen_cifar10_train_valid_data(classnum:int):
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
    
        #abnormal:1 , normal:0
        test_label = np.reshape(test_label, [-1])
        i =0
        for t in test_label:
            if t == classnum:
               test_label[i] = 1
            else:
               test_label[i] = 0
            i =i+1
        
        #abnormal:1 , normal:0
        train_label = np.reshape(train_label, [-1])
        i =0
        for t in train_label:
            if t == classnum:
               train_label[i] = 1
            else:
               train_label[i] = 0
            i =i+1

        Image_size = x_train.shape[1]
        reshape_x_train = np.reshape(x_train, [-1, Image_size*Image_size*3])
        reshape_x_test = np.reshape(x_test, [-1, Image_size*Image_size*3])
        reshape_x_train = reshape_x_train[train_label==0]
        reshape_x_test = reshape_x_test/255
        reshape_x_train = reshape_x_train/255
        print(reshape_x_test.shape)
        return reshape_x_train, reshape_x_test, train_label, test_label

def gen_KDD_train_valid_data():
    
    URL_BASE = "http://kdd.ics.uci.edu/databases/kddcup99"
    KDD_10_PERCENT_URL = URL_BASE + '/' + 'kddcup.data_10_percent.gz'
    KDD_COLNAMES_URL = URL_BASE + '/' + 'kddcup.names'

    #test download dataset
    dt = pd.read_csv(KDD_COLNAMES_URL, skiprows=1, sep=':', names=['f_names', 'f_types'])
    dt.to_csv('d:/Result1.csv')

    df_colnames = pd.read_csv(KDD_COLNAMES_URL, skiprows=1, sep=':', names=['f_names', 'f_types'])
    df_colnames.loc[df_colnames.shape[0]] = ['status', ' symbolic.']

    df = pd.read_csv(KDD_10_PERCENT_URL, header=None, names=df_colnames['f_names'].values)

    df_symbolic = df_colnames[df_colnames['f_types'].str.contains('symbolic.')]

    # one-hot encoding
    X = pd.get_dummies(df.iloc[:, :-1], columns=df_symbolic['f_names'][:-1])  # except status

    # abnormal: 1, normal: 0
    y = np.where(df['status'] == 'normal.', 1, 0)
    
    # generate train/test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123) #stratify 

     # only train with normal data
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]
    return X_train, X_test, y_train, y_test

def gen_Thyroid_train_valid_data():
    path = r'thyroid.mat'
    matdata = sio.loadmat(path)
    label_y = matdata['y']

    #abnormal:1 , normal:0
    y_label = np.where(label_y ==[0] , 0, 1)
    y_label = y_label.reshape(-1)
    X = matdata['X']

    X = pd.DataFrame(X).fillna(0)
    minmaxscaler = MinMaxScaler().fit(X)
    X = minmaxscaler.transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y_label, test_size=0.5, random_state=123)
    #abnormal:1 , normal:0
    # only train with normal data
    X_train = X_train[Y_train == 0]
    Y_train = Y_train[Y_train == 0]

    return X_train, X_test, Y_train, Y_test

    X_train = X[y_label==0]
    y_train = y_label[y_label==0]
    X_test = X
    y_test = y_label

    return X_train, X_test, y_train, y_test

def gen_arrhythmia_train_valid_data():
    path = r'arrhythmia.mat'
    matdata = sio.loadmat(path)
    X = matdata['X']
    y = matdata['y']
    y = np.where(y ==[0] , 0, 1)
    y = y.reshape(-1)

    X = pd.DataFrame(X).fillna(0)
    minmaxscaler = MinMaxScaler().fit(X)
    X = minmaxscaler.transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train = X[y==0]
    Y_train = y[y==0]
    X_test = X
    Y_test = y
    
    return X_train, X_test, Y_train, Y_test