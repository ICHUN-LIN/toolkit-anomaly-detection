import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras import backend as K
from keras.layers import Input
from keras.callbacks import LambdaCallback, Callback

from sklearn.metrics import roc_auc_score
from .Mnist_model import mnist_model
from .utility_functions import show_img, show_most_abnormal_case, show_most_normal_case, LossHistory
from .DataSet import Loader
from .Cifar10_model import cifar10_model, cifar10_simple_model

inital_center = 0
def my_loss_function(y_true, y_pred):
    result = K.mean(K.sum(K.square(y_pred - inital_center),axis=1))
    return result

def get_model(name:str):    
    if(name=="mnist"):
        return mnist_model()        
    if(name=="cifar10"):
        return cifar10_simple_model()    
    return

class DeepSVDD_Option:
    datasetname_options = ['minst','cifar10','userinput']
    datasetname = ""
    class_num = 0
    pre_train = True
    encoder_model = None
    decoder_model = None
    pre_train_epochs = 20
    train_epochs = 40
    #loss_function =
    #train_x = None 
    #label = 

class DeepSVDD(object):
    
    def __init__(self,options:DeepSVDD_Option):
        self.datasetname = options.datasetname
        self.class_num = options.class_num
        self.x_train, self.train_label, self.x_test, self.test_label, self.channel, self.org_test, self.x_train_org = Loader.load_dataset(self.datasetname,self.class_num)
        model = get_model(self.datasetname)
        Image_size = self.x_train.shape[1]
        self.inputs = Input(shape=(Image_size, Image_size, self.channel), name='encoder_input')
        self.pretrain = options.pre_train
        self.pre_train_epochs = options.pre_train_epochs
        self.train_epochs = options.train_epochs
        if(options.encoder_model is None):
            self.encoder = model.get_encoder_model(input_shape=(Image_size, Image_size, self.channel))
            self.decoder = model.get_decoder_model()
        else:
            self.encoder = options.encoder_model
            self.decoder = options.decoder_model

    def train(self):
        
        if(self.pretrain is True):
            autoencoder = Model(self.inputs, self.decoder(self.encoder(self.inputs)), name='autoencoder')
            self.train_autoencoder(self.x_train, self.x_test, autoencoder, self.pre_train_epochs)
        
        global inital_center
        inital_center = self.init_center_c(self.x_train, self.encoder)
        batch_size = 128
        self.encoder.summary()
        history = LossHistory()
        optimizer = keras.optimizers.adam(lr=0.001, decay=1e-06)
        self.encoder.compile(loss=my_loss_function, optimizer=optimizer)
        self.encoder.fit(self.x_train,self.train_label,                
                        epochs=self.train_epochs,
                        batch_size=batch_size,
                        callbacks=[history]
                        )
        #history.loss_plot('epoch')
        #show_most_abnormal_case(self.encoder,inital_center,self.datasetname,self.x_train, self.x_train_org)
        #show_most_normal_case(self.encoder,inital_center,self.datasetname,self.x_train, self.x_train_org)

    def test(self):
        self.test_auc(self.encoder, self.datasetname,self.x_test,self.test_label)
    
    def train_autoencoder(self, x_train, x_test, autoencoder, epoches):
        autoencoder.summary()
        optimizer = keras.optimizers.adam(lr=0.001, decay=1e-06)
        autoencoder.compile(loss='mse', optimizer=optimizer)
        # Train the autoencoder
        batch_size = 128
        autoencoder.fit(x_train,
                        x_train,
                        epochs=epoches,
                        batch_size=batch_size)    
        return

    def init_center_c(self, data,train_model:Model, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        output = train_model.predict(data)
        c = np.zeros(output.shape[0])
        c = np.sum(output, axis=0)
        c /= output.shape[0]
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c
    
    def test_auc(self, encoder,datasetname,x_data,x_label):
        x_result = encoder.predict(x_data)
        i= 0
        error = np.zeros(x_result.shape[0])
        for x in x_result:
            sd = ((x-inital_center)**2).sum()
            error[i] = sd
            i=i+1
        val = roc_auc_score(x_label, error)
        print('Test set AUC: {:.2f}%'.format(100. * val))
        return





