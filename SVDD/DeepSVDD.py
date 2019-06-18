import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten ,MaxPooling2D, BatchNormalization
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist, cifar10
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import LeakyReLU
from keras.models import load_model
from .Mnist_model import mnist_model, show_mnist_image
from .DataSet import Loader
from .Cifar10_model import cifar10_model, plot_images_labels_prediction, cifar10_model2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from keras.callbacks import LambdaCallback, Callback 

def dist(xxx):
    return (xxx-inital_center) ** 2, K.sum(K.square(xxx - inital_center),axis=1)- Radius**2

inital_center = 0
Radius = 1
def my_loss_function(y_true, y_pred):
    result = K.mean(K.sum(K.square(y_pred - inital_center),axis=1))
    return result

def init_center_c(data,train_model:Model, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    output = train_model.predict(data)
    c = np.zeros(output.shape[0])
    c = np.sum(output, axis=0)
    c /= output.shape[0]
    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps
    return c

def train_autoencoder(x_train, x_test, autoencoder):
    autoencoder.summary()
    optimizer = keras.optimizers.adam(lr=0.001, decay=1e-06)
    autoencoder.compile(loss='mse', optimizer=optimizer)
    # Train the autoencoder
    batch_size = 128
    autoencoder.fit(x_train,
                    x_train,
                    epochs=20,
                    batch_size=batch_size)    
    return

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

def get_model(name:str):    
    if(name=="mnist"):
        return mnist_model()        
    if(name=="cifar10"):
        return cifar10_model2()    
    return

def show_img(name:str, dataset):    
    if(name=="mnist"):
        show_mnist_image(dataset)        
    if(name=="cifar10"):
        plot_images_labels_prediction(dataset,dataset,0, 25)
    return

def get_radius(dist, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def find_abnormal_case(encoder,datasetname,x_data,x_org,iverse):
    x_result = encoder.predict(x_data)
    i= 0
    error = np.zeros(x_result.shape[0])
    for x in x_result:
        sd = ((x-inital_center)**2).sum()
        error[i] = sd
        i=i+1

    x_list = x_org.tolist()
    cc = sorted(zip(error,x_list),reverse=iverse)
    (error,x_list) = zip(*cc)
    x_list = np.array(x_list)
    show_img(datasetname,x_list)

def test_auc(encoder,datasetname, test_label, x_data,label,iverse,x_org, p=False):
    x_result = encoder.predict(x_data)
    i= 0
    error = np.zeros(x_result.shape[0])
    for x in x_result:
        sd = ((x-inital_center)**2).sum()
        error[i] = sd
        i=i+1

    val = roc_auc_score(test_label, error)
    print('Test set AUC: {:.2f}%'.format(100. * val))
    if(p):
        x_list = x_org.tolist()
        cc = sorted(zip(error,x_list),reverse=iverse)
        (error,x_list) = zip(*cc)
        x_list = np.array(x_list)
        show_img(datasetname,x_list)

class DeepSVDD_Option:
    datasetname = ""
    class_num = 0
    train_x = 0
    encoder_model = None
    decoder_model = None
    #pretrain = True
    #loss_function = 
    #label = 

class DeepSVDD(object):
    
    def __init__(self,options:DeepSVDD_Option):
        self.datasetname = options.datasetname
        self.class_num = options.class_num
        self.x_train, self.train_label, self.x_test, self.test_label, self.channel, self.org_test, self.x_train_org = Loader.load_dataset(self.datasetname,self.class_num)
        model = get_model(self.datasetname)
        Image_size = self.x_train.shape[1]
        self.inputs = Input(shape=(Image_size, Image_size, self.channel), name='encoder_input')
        if(options.encoder_model is None):
            self.encoder = model.get_encoder_model(input_shape=(Image_size, Image_size, self.channel))
            self.decoder = model.get_decoder_model()
        else:
            self.encoder = options.encoder_model
            self.decoder = options.decoder_model

    def train(self):
        autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name='autoencoder')
        train_autoencoder(self.x_train, self.x_test, autoencoder)
        #self.encoder.save("encoder_model_1_7.h5")
        global inital_center
        inital_center = init_center_c(self.x_train, self.encoder)
        batch_size = 128
        self.encoder.summary()
        history = LossHistory()
        optimizer = keras.optimizers.adam(lr=0.001, decay=1e-06)
        self.encoder.compile(loss=my_loss_function, optimizer=optimizer)
        self.encoder.fit(self.x_train,self.train_label,                
                        epochs=20,
                        batch_size=batch_size,
                        callbacks=[history]
                        )
        #history.loss_plot('epoch')
        #test_auc(encoder, self.datasetname,test_label,x_test,test_label,False,org_test)
    
    def test(self):
        test_auc(self.encoder, self.datasetname,self.test_label,self.x_test,self.test_label,False,self.org_test)




