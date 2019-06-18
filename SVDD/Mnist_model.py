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

def show_mnist_image(x_train,image_size=28):
    rows, cols = 5, 8 
    num = rows * cols
    imgs = x_train[:num]
    imgs = imgs.reshape((rows, cols, image_size, image_size))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    imgs = (imgs * 255).astype(np.uint8)
    plt.figure()
    plt.axis('off')
    plt.title('top')
    plt.imshow(imgs, interpolation='none', cmap='gray')
    plt.show()
    return

class mnist_model(object):
    
    def get_encoder_model(self,input_shape):
        kernel_size = 5
        latent_dim = 32 #output features number
        inputs = Input(shape=input_shape, name='encoder_input')

        x = inputs
        x = Conv2D(filters=8,kernel_size=kernel_size,use_bias= False
                    ,padding="same")(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(filters=4,kernel_size=kernel_size,use_bias= False
                    ,padding="same")(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Generate the latent vector
        x = Flatten()(x)
        latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')
        encoder.summary()
        return encoder

    def get_decoder_model(self):        
        kernel_size = 5
        latent_inputs = Input(shape=(32,), name='decoder_input')
        x = Dense(7*7*4,use_bias= False)(latent_inputs)
        x = Reshape((7, 7, 4))(x)
        x = BatchNormalization(epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        #x = ZeroPadding2D(padding=(3, 3), dim_ordering='default')(x)
        x = Conv2DTranspose(filters=8, kernel_size=kernel_size, padding='same',use_bias= False)(x)
        x = BatchNormalization( epsilon=1e-04, scale=False)(x)
        x = LeakyReLU()(x)
        x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)

        x = Conv2DTranspose(filters=1, kernel_size=kernel_size, padding='same',use_bias= False)(x)
        outputs = Activation('sigmoid', name='decoder_output')(x)

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')
        decoder.summary()
        return decoder