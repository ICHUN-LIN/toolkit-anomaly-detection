from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten ,MaxPooling2D, BatchNormalization
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.layers import LeakyReLU
from keras.models import load_model
import tensorflow as tf

def plot_images_labels_prediction(images,labels,idx,num=10):#images爲圖像，labels爲標籤，prediction爲預測，idx爲要開始顯示的圖像的索引，num爲要顯示圖像的數量，默認是10，最多25
    fig=plt.gcf()
    fig.set_size_inches(12,14) #設置畫布大小爲12x14英寸
    if num>25:         #設置最多可以顯示25張圖片
        num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title=str(i) #label[i][0]即爲第i個圖像所屬的類別
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()

class cifar10_model(object):

    def get_encoder_model(self,input_shape):

        #32x32x3

        kernel_size = 5
        latent_dim = 128 #output features number
        inputs = Input(shape=input_shape, name='encoder_input')

        x = inputs
        x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(filters=64,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(filters=128,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

        #Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()
        return encoder

    def get_decoder_model(self):
        kernel_size = 5
        latent_inputs = Input(shape=(128,), name='decoder_input')
        x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
        x = Reshape((4, 4, 8))(x)
        x = LeakyReLU()(x)
        
        x = Conv2DTranspose(filters=128, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)        
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        x = Conv2DTranspose(filters=64, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        x = Conv2DTranspose(filters=32, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
        x = Conv2DTranspose(filters=3, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)

        outputs = Activation('sigmoid', name='decoder_output')(x)
        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        return decoder

class cifar10_model2(object):
    
   
    def get_encoder_model(self,input_shape):
        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(32, (3, 3), padding='same',use_bias= False)(input_img)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same',use_bias= False)(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        out = Flatten()(x)

        #Instantiate Encoder Model
        encoder = Model(input_img, out, name='encoder')
        encoder.summary()

        return encoder

    def get_decoder_model(self):
        latent_inputs = Input(shape=(8*8*32,), name='decoder_input')
        x = Reshape((8, 8, 32))(latent_inputs) 
        x = Conv2D(32, (3, 3), padding='same',use_bias= False)(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same',use_bias= False)(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same',use_bias= False)(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        decoded = Activation('sigmoid')(x)
        decoder = Model(latent_inputs, decoded)
        decoder.summary()
        return decoder    

    def autoencoder_model_test(self):
        input_img = Input(shape=(32, 32, 3))
        x = Conv2D(32, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)
        model = Model(input_img, decoded)
        return model
