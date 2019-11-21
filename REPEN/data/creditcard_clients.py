import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


def get_train(*args):
    """Get training dataset for KDD 10 percent"""
    return _get_adapted_dataset("train")

def get_test(*args):
    """Get testing dataset for KDD 10 percent"""
    return _get_adapted_dataset("test")

def get_shape_input():
    """Get shape of the dataset for KDD 10 percent"""
    return (None, 23)

def get_shape_label():
    """Get shape of the labels in KDD 10 percent"""
    return (None,)

def _get_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 120)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 120)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    
    df = pd.read_excel('creditcard_clients.xls')
    # Normalize the purchase amount
    text_l = ['X_1','X_12', 'X_13','X_14','X_15','X_16', 'X_17','X_18','X_19','X_20', 'X_21', 'X_22','X_23']

    for name in text_l:
        df['normAmount'+ name] = StandardScaler().fit_transform(df[name].values.reshape(-1, 1))
        
    for name in text_l:
        df = df.drop([name],axis=1)
    #df = df.drop(['Amount'], axis=1)
    labels =df['Class']
    
    #df ['normAmount'] =

    
   

    df_train = df.sample(frac=0.7, random_state=42)
    
    df_test = df.loc[~df.index.isin(df_train.index)]

    x_train, y_train = _to_xy(df_train, target='Class')
    y_train = y_train.flatten().astype(int)
    x_test, y_test = _to_xy(df_test, target='Class')
    y_test = y_test.flatten().astype(int)

    
    #x_train_fraud = x_train[y_train==1]
    #y_train_fraud = y_train[y_train == 1]
    #x_test = np.concatenate([x_test, x_train_fraud])
    #y_test = np.concatenate([y_test, y_train_fraud])
    
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]
    
    '''
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)
    scaler.transform(x_test)
    '''
    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset

def _get_adapted_dataset(split):
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    dataset = _get_dataset()
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split != 'train':
        dataset[key_img], dataset[key_lbl] = _adapt(dataset[key_img],
                                                    dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])



def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)


def _adapt(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1   

    rng = np.random.RandomState(42) # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_test = inliersx.shape[0]
    out_size_test = int(size_test*rho/(1-rho))

    outestx = outliersx[:out_size_test]
    outesty = outliersy[:out_size_test]

    testx = np.concatenate((inliersx,outestx), axis=0)
    testy = np.concatenate((inliersy,outesty), axis=0)

    size_test = testx.shape[0]
    inds = rng.permutation(size_test)
    testx, testy = testx[inds], testy[inds]

    return testx, testy
'''
X_train, y_train = get_train()
#trainx_copy = trainx.copy()
X_test, y_test = get_test()

from scipy.stats import itemfreq
print(itemfreq(y_train))
print(itemfreq(y_test))
print(y_train.shape)
print(y_test.shape)

'''
