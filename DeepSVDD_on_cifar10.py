from toolkit_prototype import toolkit
import keras
from keras.layers import Activation, Dense, Input, LeakyReLU
from keras.layers import Conv2D, Flatten ,MaxPooling2D, BatchNormalization
from keras.layers import Reshape, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.models import Model

def test_B(n_class,epcho):
    #change kernal size to 3*3
    kernel_size = 3
    latent_dim = 128 #output features number
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('change kernal size to 3*3')
    return

def test_C(n_class,epcho):
    #set filter = 32
    kernel_size = 5
    latent_dim = 128 #output features number
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

    x = inputs
    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization(epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

    #Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')

    latent_inputs = Input(shape=(128,), name='decoder_input')
    x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
    x = Reshape((4, 4, 8))(x)
    x = LeakyReLU()(x)
        
    x = Conv2DTranspose(filters=32, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)        
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
        
    x = Conv2DTranspose(filters=32, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)
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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('set filter = 32')
    return

def test_D(n_class,epcho):
    #remove one layer
    kernel_size = 5
    latent_dim = 128 #output features number
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

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

    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

    #Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')

    latent_inputs = Input(shape=(128,), name='decoder_input')
    x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
    x = Reshape((8, 8, 2))(x)
    x = LeakyReLU()(x)

    '''    
    x = Conv2DTranspose(filters=128, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)        
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x)
    '''

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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('remove one Conv layer')
    return

def test_E(n_class,epcho):
    #remove_one_dense_layer
    kernel_size = 5
    latent_dim = 128 #output features number
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

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
    latent = Flatten()(x)
    #latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

    #Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')

    latent_inputs = Input(shape=(2048,), name='decoder_input')
    x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
    x = Reshape((4, 4, 128))(x)
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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('remove_one_dense_layer')
    return

def test_C_D(n_class,epcho):
    #remove one Conv layer+ remove one Dense layer
    kernel_size = 5
    latent_dim = 128 #output features number
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

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

    # Generate the latent vector
    latent = Flatten()(x)
    #latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

    #Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(8*8*64,), name='decoder_input')
    x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
    x = Reshape((8, 8, 64))(latent_inputs)
    x = LeakyReLU()(x)

        
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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('remove one Conv layer+ remove one Dense layer')
    return

def test_C_D_E(n_class,epcho):
    #remove one Conv layer+ remove one Dense layer+ set filter =32
    kernel_size = 5
    latent_dim = 128 #output features number
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

    x = inputs
    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization(epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    '''
    x = Conv2D(filters=128,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    '''

    # Generate the latent vector
    latent = Flatten()(x)
    #latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

    #Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(8*8*32,), name='decoder_input')
    x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
    x = Reshape((8, 8, 32))(latent_inputs)
    x = LeakyReLU()(x)

        
    x = Conv2DTranspose(filters=32, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x) 
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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('remove one Conv layer+ remove one Dense layer+set filters = 32')
    return

def test_B_C_D_E(n_class,epcho):
    #remove one Conv layer+ remove one Dense layer+ set filter =32
    #set kernal 3*3
    kernel_size = 3
    inputs = Input(shape=(32, 32, 3), name='encoder_input')

    x = inputs
    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization(epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=32,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    '''
    x = Conv2D(filters=128,kernel_size=kernel_size,use_bias= False
                    ,padding="same",kernel_initializer="glorot_normal")(x)
    x = BatchNormalization( epsilon=1e-04, scale=False)(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    '''

    # Generate the latent vector
    latent = Flatten()(x)
    #latent = Dense(latent_dim, name='latent_vector', use_bias=False)(x)

    #Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(8*8*32,), name='decoder_input')
    x = BatchNormalization( epsilon=1e-04, scale=False)(latent_inputs)
    x = Reshape((8, 8, 32))(latent_inputs)
    x = LeakyReLU()(x)

        
    x = Conv2DTranspose(filters=32, kernel_size=kernel_size, padding='same',use_bias= False,kernel_initializer="glorot_normal")(x) 
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

    tool = toolkit()
    #set options
    options = tool.setAlgorthem("deepsvdd")
    options.datasetname = "cifar10"
    options.class_num = n_class
    options.pre_train_epochs = 10
    options.train_epochs = epcho
    options.encoder_model = encoder
    options.decoder_model = decoder
    
    tool.train(options)
    tool.test()
    print('remove one Conv layer+ remove one Dense layer+set filters = 32+set kernal 3*3')
    return

n_class = 3
epcho = 60
test_B(n_class,epcho)
print('n_class='+str(n_class))
test_C(n_class,epcho)
print('n_class='+str(n_class))
test_D(n_class,epcho)
print('n_class='+str(n_class))
test_E(n_class,epcho)
print('n_class='+str(n_class))
test_C_D(n_class,epcho)
print('n_class='+str(n_class))
test_C_D_E(n_class,epcho)
print('n_class='+str(n_class))
test_B_C_D_E(n_class,epcho)
print('n_class='+str(n_class))
