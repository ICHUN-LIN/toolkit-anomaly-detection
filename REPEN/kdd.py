import logging
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

logger = logging.getLogger(__name__)

path = str(Path().absolute())+"/REPEN"
def get_dataset():
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
    col_names = _col_names()
    df = pd.read_csv(path + "/data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    text_l = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login']

    for name in text_l:
        _encode_text_dummy(df, name)

    labels = df['label'].copy()
    labels[labels != 'normal.'] = 0
    labels[labels == 'normal.'] = 1

    df['label'] = labels


    x_train, y_train = _to_xy(df, target='label')
    y_train = y_train.flatten().astype(int)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    scaler.transform(x_train)

    dataset = {}
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)

    return x_train, y_train

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

def _encode_text_dummy(df, name):
    """Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1]
    for red,green,blue)
    """
    dummies = pd.get_dummies(df.loc[:,name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df.loc[:, dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def _to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().astype(np.float32)

def _col_names():
    """Column names of the dataframe"""
    return ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]


