from toolkit_prototype import toolkit
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten ,MaxPooling2D, BatchNormalization
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.models import Model


def test_SVDD():
    i = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(i)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    Encoder = Model(i, x)

    i = Input(shape=(4, 4, 8)) # 8 conv2d features
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(i)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    Decoder = Model(i, x)

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "mnist"
    options.class_num = 4
    options.encoder_model = Encoder
    options.decoder_model = Decoder
    tool.train(options)
    tool.test()

    return


test_SVDD()

'''
tool = toolkit()
#set options
options = tool.setAlgorthem("deepsvdd")
options.datasetname = "mnist"
options.class_num = 4
tool.train(options)
tool.test()
'''