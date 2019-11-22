import keras
from keras.layers import Activation, Dense, Input
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

def gen_KDD_train_valid_data_for_DeepSVDD():
    
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

    y_train = np.reshape(y_train, [-1])
    y_test = np.reshape(y_test, [-1])

    # only train with normal data
    X_train = X_train[y_train == 0]
    y_train = y_train[y_train == 0]

    return X_train, y_train, X_test, y_test

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

    return X_train, Y_train,X_test, Y_test

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

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=123)
    
    X_train = X_train[Y_train == 0]
    Y_train = Y_train[Y_train == 0]

    return X_train, Y_train, X_test, Y_test

def test_kdd_in_DeepSVDD():

    input_shape = (121,)
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(60, activation='tanh')(inputs)
    out = Dense(30, activation='tanh')(x)
    Encoder = Model(inputs,out)

    i = Input(shape=(30,))
    x = Dense(60, activation='tanh')(i)
    y = Dense(121, name='decoder_out', activation=None)(x)
    Decoder = Model(i,y)

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "kdd"
    options.class_num = 0
    options.pre_train_epochs = 10
    options.train_epochs = 20
    options.encoder_model = Encoder
    options.decoder_model = Decoder
    options.inputs = inputs
    options.x_train, options.train_label, options.x_test, options.test_label = gen_KDD_train_valid_data_for_DeepSVDD()
    scaler = StandardScaler()
    options.x_train = scaler.fit_transform(options.x_train)
    options.x_test = scaler.transform(options.x_test)
    tool.train(options)
    tool.test()

def test_Thyroid_in_DeepSVDD():
    
    input_shape = (6,)
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(14, activation='tanh')(inputs)
    out = Dense(5, activation='tanh')(x)
    Encoder = Model(inputs,out)

    i = Input(shape=(5,))
    x = Dense(14, activation='tanh')(i)
    y = Dense(6, name='decoder_out', activation=None)(x)
    Decoder = Model(i,y)

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "Thyroid"
    options.class_num = 0
    options.pre_train_epochs = 10
    options.train_epochs = 20
    options.encoder_model = Encoder
    options.decoder_model = Decoder
    options.inputs = inputs
    options.x_train, options.train_label, options.x_test, options.test_label = gen_Thyroid_train_valid_data()
    scaler = StandardScaler()
    options.x_train = scaler.fit_transform(options.x_train)
    options.x_test = scaler.transform(options.x_test)
    tool.train(options)
    tool.test()

def test_arrhythmia_in_DeepSVDD():    

    input_shape = (274,)
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(125, activation='tanh')(inputs)
    out = Dense(60, activation='tanh')(x)
    Encoder = Model(inputs,out)

    i = Input(shape=(60,))
    x = Dense(125, activation='tanh')(i)
    y = Dense(274, name='decoder_out', activation=None)(x)
    Decoder = Model(i,y)

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "arrhythmia"
    options.class_num = 0
    options.pre_train_epochs = 10
    options.train_epochs = 50
    options.encoder_model = Encoder
    options.decoder_model = Decoder
    options.inputs = inputs
    options.x_train, options.train_label, options.x_test, options.test_label = gen_arrhythmia_train_valid_data()
    scaler = StandardScaler()
    options.x_train = scaler.fit_transform(options.x_train)
    options.x_test = scaler.transform(options.x_test)
    tool.train(options)
    tool.test()


#test_kdd_in_DeepSVDD()
test_Thyroid_in_DeepSVDD()
test_arrhythmia_in_DeepSVDD()