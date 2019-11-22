from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from dataset import gen_KDD_train_valid_data, gen_Thyroid_train_valid_data, gen_arrhythmia_train_valid_data, gen_mist_train_valid_data, gen_cifar10_train_valid_data
from sklearn.preprocessing import StandardScaler
from collections import namedtuple
import os
import argparse
from keras.layers import Lambda, Input, Dense, Concatenate, Reshape, merge, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import tensorflow as tf
import numpy as np
from keras import layers
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support
    


def extract_rec_features(X, X_rec):
    def l2(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1, keepdims=True))

    X_l2 = l2(X)
    X_rec_l2 = l2(X_rec)
    res_l2 = l2(X - X_rec)
    relative_euclid = res_l2 / X_l2
    cos_similarity = tf.reduce_sum(X * X_rec, axis=1, keepdims=True) / (X_l2 * X_rec_l2)
    return relative_euclid, cos_similarity

def inference(gamma, z):
    # gamma: N*K; z: N*D
    gamma_sum = tf.reduce_sum(gamma, axis=0)  # K
    phi = tf.reduce_mean(gamma, axis=0)  # K
    mu = tf.reduce_sum(tf.expand_dims(gamma, axis=-1) * tf.expand_dims(z, axis=1),
                                     axis=0) / tf.expand_dims(gamma_sum, axis=-1)  # K*D / K*1 -> K*D
    z_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)  # N*1*D - 1*K*D -> N*K*D
    z_mu_mul = tf.expand_dims(z_mu, axis=-1) * tf.expand_dims(z_mu, axis=-2)  # N*K*D*1 * N*K*1*D -> N*K*D*D
    sigma = tf.reduce_sum(tf.expand_dims(tf.expand_dims(gamma, -1), -1) * z_mu_mul, axis=0)/ tf.expand_dims(tf.expand_dims(gamma_sum, -1), -1)  # K*D*D'''
    # sigma + epsilon
    eps = 1e-12
    diag_eps = tf.diag(tf.ones(z.shape[-1])) * eps  # D*D
    diag_eps = tf.expand_dims(diag_eps, axis=0)  # 1*D*D
    sigma_eps = sigma + diag_eps  # K*D*D
    return phi, mu, sigma_eps, sigma

def energy(z, phi, mu, sigma_eps):
    # z: N*D
    # phi: K
    # mu: K*D
    # sigma: K*D*D
    sigma_inverse = tf.matrix_inverse(sigma_eps)  # K*D*D
    z_mu = tf.expand_dims(z, axis=1) - tf.expand_dims(mu, axis=0)  # N*1*D - 1*K*D -> N*K*D
    exp_val_tmp = -0.5 * tf.reduce_sum(
        tf.reduce_sum(tf.expand_dims(z_mu, -1) * tf.expand_dims(sigma_inverse, 0), -2) * z_mu, -1)  # N*K
    det_sigma = tf.matrix_determinant(sigma_eps)  # K
    log_det_simga = tf.expand_dims(tf.log(tf.sqrt(2 * np.pi * det_sigma)), 0)  # K
    log_phi = tf.expand_dims(tf.log(phi), 0)  # 1*K
    exp_val = log_phi + exp_val_tmp - log_det_simga  # 1*K + N*k - 1*K -> N*K
    energies = -tf.reduce_logsumexp(exp_val, axis=1)  # N
    return energies

def gmm_loss(energies, sigma):
    energy_mean = tf.reduce_mean(energies)
    diag_loss = tf.reduce_sum(1 / tf.matrix_diag_part(sigma))  # reduce_sum(K*D)
    return energy_mean, diag_loss


class DAGMM_Options:
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    batch_size = 1024
    epoch = 80
    comp_hidden = [60, 30, 10, 1, 10, 30, 60]
    mid_layer = 3
    comp_activation = 'tanh'
    est_hidden = [10, ]
    est_activation = 'tanh'
    keep_prob = 0.5
    mix_components = 4
    lambda1 = 0.1
    lambda2 = 0.005
    lr = 0.0001
    normal_portion = 80

    #create model parameter
    compress_net_inputs = None
    compress_net_mid_presentation = None
    compress_net_outputs = None
    gamma_layer = None
    z_layer = None
    
    @staticmethod
    def create_estimate_net_z_layer(compress_net_inputs, compress_net_mid_presentation, compress_net_outputs):
        return Lambda(DAGMM.get_z, name='z')([compress_net_mid_presentation, compress_net_outputs, compress_net_inputs])

    
    
    '''
    def create_compress_net_org(self):
        input_shape = (self.X_train.shape[1],)
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(60, activation='tanh')(inputs)
        x = Dense(30, activation='tanh')(x)
        x = Dense(10, activation='tanh')(x)
        mid_presentation = x = Dense(1, name='mid_presentation', activation=None)(x)
        x = Dense(10, activation='tanh')(x)
        x = Dense(30, activation='tanh')(x)
        x = Dense(60, activation='tanh')(x)
        outputs = Dense(121, name='compression_out_layer', activation=None)(x)
        return inputs, mid_presentation, outputs
    '''

    '''
    def create_estimate_net_two(self):
        optimizer = keras.optimizers.adam(lr=0.0001)
        z = Lambda(DAGMM.sampling, name='z')([self.compress_net_mid_presentation, self.compress_net_outputs, self.compress_net_inputs])
        layer = Dense(10, activation='tanh')(z)
        layer = Dropout(0.5, noise_shape=None, seed=None)(layer)
        gamma = Dense(4, activation='softmax', name='gamma')(layer)
        my_loss_layer = Lambda(DAGMM.loss_layer, name='loss_layer')([self.compress_net_inputs, self.compress_net_outputs, gamma, z, self.option])
        final_model = Model(self.compress_net_inputs, my_loss_layer, name='encoder')
        final_model.compile(loss=DAGMM.my_loss_function, optimizer=optimizer)
        return final_model
    '''

class DAGMM(object):
    
    def __init__(self, options):
        self.X_train = options.X_train
        self.X_test = options.X_test
        self.y_train = options.y_train
        self.y_test = options.y_test
        self.option = options

        if options.compress_net_inputs!= None and options.compress_net_mid_presentation != None and options.compress_net_outputs!=None  and options.gamma_layer!=None: 
            self.compress_net_inputs= options.compress_net_inputs
            self.compress_net_mid_presentation = options.compress_net_mid_presentation
            self.compress_net_outputs = option.compress_net_outputs
            self.final_model = self.create_final_model(options.z_layer,options.gamma_layer)
        else:
            self.compress_net_inputs, self.compress_net_mid_presentation, self.compress_net_outputs = self.create_compress_net()
            self.final_model = self.create_estimate_net()       

    def create_compress_net(self):
        input_shape = (self.X_train.shape[1],)
        inputs = Input(shape=input_shape, name='encoder_input')
  
        i = 0
        for y in self.option.comp_hidden:            
            if i == 0 :
               x =  Dense(y, activation=self.option.comp_activation)(inputs)
            elif (i >0 and i != self.option.mid_layer):
               x =  Dense(y, activation=self.option.comp_activation)(x)
            elif (i> 0 and i == self.option.mid_layer):
               mid_presentation = x = Dense(y, name='mid_presentation', activation=None)(x)        
            i = i + 1
        
        outputs = Dense(self.X_train.shape[1], name='compression_out_layer', activation=None)(x)
        return inputs, mid_presentation, outputs

    def create_estimate_net(self):
        optimizer = keras.optimizers.adam(lr=self.option.lr)
        z = Lambda(DAGMM.get_z, name='z')([self.compress_net_mid_presentation, self.compress_net_outputs, self.compress_net_inputs])
        
        layer = Dense(self.option.est_hidden[0], activation='tanh')(z)
        layer = Dropout(self.option.keep_prob, noise_shape=None, seed=None)(layer)
        gamma = Dense(self.option.mix_components, activation='softmax', name='gamma')(layer)
        
        my_loss_layer = Lambda(DAGMM.loss_layer, name='loss_layer')([self.compress_net_inputs, self.compress_net_outputs, gamma, z])
        final_model = Model(self.compress_net_inputs, my_loss_layer, name='encoder')
        final_model.compile(loss=DAGMM.my_loss_function, optimizer=optimizer)
        return final_model

    def create_final_model(self,z,gamma):
        optimizer = keras.optimizers.adam(lr=self.option.lr)
        my_loss_layer = Lambda(DAGMM.loss_layer, name='loss_layer')([self.compress_net_inputs, self.compress_net_outputs, gamma, z])
        final_model = Model(self.compress_net_inputs, my_loss_layer, name='encoder')
        final_model.compile(loss=DAGMM.my_loss_function, optimizer=optimizer)
        return final_model

    def train(self):
        self.final_model.fit(self.X_train, self.X_train, 
        epochs=self.option.epoch,
        batch_size= self.option.batch_size
        )

    
    def train_by_design(self):
        batch_size = self.option.batch_size
        total_size = self.X_train.shape[0]
        epochs = self.option.epoch
        number = (int)(total_size/batch_size)+1
        print(number)
        for i in range(1, epochs):
            for j in range(1,number+1):
                test_train = X_train[(j-1)*batch_size:j*batch_size]
                self.final_model.train_on_batch(test_train, test_train)
                print(j)        
    
    def test(self):
        self.precise_recall_f1score()
        
    @staticmethod
    def my_loss_function(y_true, y_pred):
        return y_pred
    
    @staticmethod
    def loss_layer(args):
        #gamma: estimate_out_put_value, z:estimated_input_value
        #org_x: original_x, nex_x: compression_net: output
        org_x, new_x, gamma, z = args
        phi, mu, sigma_eps, sigma = inference(gamma, z)
        energies  = energy(z, phi, mu, sigma_eps)
        total_loss = DAGMM.loss(0.1, 0.005, energies, org_x, new_x, sigma_eps)
        return total_loss
    
    @staticmethod
    def loss(lambda1, lambda2, energies, X, X_rec, sigma):
        loss_euclid = tf.reduce_mean(tf.reduce_sum(tf.square(X - X_rec), axis=1))
        energy_mean, diag_loss = gmm_loss(energies, sigma)
        total_loss = loss_euclid + lambda1 * energy_mean + lambda2 * diag_loss
        return total_loss
    
    @staticmethod
    def get_z(args):
        x, y, i = args
        euclidean_dist, cosine_dist  = extract_rec_features(i, y)
        result = K.concatenate([x, euclidean_dist, cosine_dist])
        return result
    
    def precise_recall_f1score(self):
        gamma_layer = Model(inputs=self.final_model.input, outputs=self.final_model.get_layer('gamma').output)
        gamma_output = gamma_layer.predict(self.X_test)

        z_layer = Model(inputs=self.final_model.input, outputs=self.final_model.get_layer('z').output)
        z_output = z_layer.predict(self.X_test)

        phi, mu, sigma_eps, sigma = inference(gamma_output, z_output)
        energies  = energy(z_output, phi, mu, sigma_eps)

        with tf.Session() as sess:
            result = sess.run(energies)
            anomaly_energy_threshold = np.percentile(result, self.option.normal_portion)
            print("Test Energy threshold to detect anomaly :"+str(anomaly_energy_threshold))
            # abnormal: 1, normal: 0
            y_pred = np.where(result >= anomaly_energy_threshold, 1, 0)
            print(list(y_pred).count(0))
            print(list(y_pred).count(1))
            precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test, y_pred, average='binary')
            print(" Precision = {0:.3f}".format(precision))
            print(" Recall    = {0:.3f}".format(recall))
            print(" F1-Score  = {0:.3f}".format(fscore))

class DAGMM_WithOut_TwoFeature_of_midlayer(DAGMM):
    
    def __init__(self, options):
        super().__init__(options)

    def create_estimate_net(self):
        optimizer = keras.optimizers.adam(lr=self.option.lr)
        z = Lambda(DAGMM_WithOut_TwoFeature_of_midlayer.get_z, name='z')([self.compress_net_mid_presentation, self.compress_net_outputs, self.compress_net_inputs])
        layer = Dense(self.option.est_hidden[0], activation='tanh')(z)
        layer = Dropout(self.option.keep_prob, noise_shape=None, seed=None)(layer)
        gamma = Dense(self.option.mix_components, activation='softmax', name='gamma')(layer)
        my_loss_layer = Lambda(DAGMM.loss_layer, name='loss_layer')([self.compress_net_inputs, self.compress_net_outputs, gamma, z])
        final_model = Model(self.compress_net_inputs, my_loss_layer, name='encoder')
        final_model.compile(loss=DAGMM.my_loss_function, optimizer=optimizer)
        return final_model
    
    @staticmethod
    def get_z(args):
        x, y, i = args
        return x

class DAGMM_WithOut_compress_model(DAGMM):
    
    def __init__(self, options):
        super().__init__(options)

    def create_estimate_net(self):
        optimizer = keras.optimizers.adam(lr=self.option.lr)
        z = Lambda(DAGMM_WithOut_TwoFeature_of_midlayer.get_z, name='z')([self.compress_net_inputs, self.compress_net_inputs, self.compress_net_inputs])
        layer = Dense(self.option.est_hidden[0], activation='tanh')(z)
        layer = Dropout(self.option.keep_prob, noise_shape=None, seed=None)(layer)
        gamma = Dense(self.option.mix_components, activation='softmax', name='gamma')(layer)
        my_loss_layer = Lambda(DAGMM.loss_layer, name='loss_layer')([self.compress_net_inputs, self.compress_net_outputs, gamma, z])
        final_model = Model(self.compress_net_inputs, my_loss_layer, name='encoder')
        final_model.compile(loss=DAGMM.my_loss_function, optimizer=optimizer)
        return final_model
    

 
        
option = DAGMM_Options()
X_train, X_test, y_train, y_test = gen_KDD_train_valid_data()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
option.X_test = X_test
option.X_train = X_train
option.y_train = y_train
option.y_test = y_test
option.epoch = 80
option.batch_size = 1028
dagmm = DAGMM(option)
#dagmm.train()
#dagmm = DAGMM_WithOut_TwoFeature_of_midlayer(option)


option = DAGMM_Options()
X_train, X_test, y_train, y_test = gen_Thyroid_train_valid_data()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
option.X_test = X_test
option.X_train = X_train
option.y_train = y_train
option.y_test = y_test
option.comp_hidden = [14,5,14]
#option.comp_hidden = [12,4,3,4,12]
option.mid_layer = 1
option.mix_components = 2
option.epoch = 100
option.normal_portion = 97.5
option.batch_size = 1028
#dagmm = DAGMM(option)
dagmm = DAGMM_WithOut_TwoFeature_of_midlayer(option)
dagmm.train_by_design()
dagmm.test()

option = DAGMM_Options()
X_train, X_test, y_train, y_test = gen_arrhythmia_train_valid_data()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
option.X_test = X_test
option.X_train = X_train
option.y_train = y_train
option.y_test = y_test
option.comp_hidden = [10,2,10]
option.mid_layer = 1
option.mix_components = 2
option.epoch = 15000
option.batch_size = 500
option.normal_portion = 85
dagmm = DAGMM(option)
#dagmm.train()


option = DAGMM_Options()
X_train, X_test, y_train, y_test = gen_mist_train_valid_data(0) #32*32*1
option.X_test = X_test
option.X_train = X_train
option.y_train = y_train
option.y_test = y_test
#print(X_train.shape)
option.comp_hidden = [256, 64, 15, 64, 256]
option.mid_layer = 2
option.mix_components = 8
option.epoch = 30
option.normal_portion = 90
dagmm = DAGMM(option)
#dagmm.train()

gen_cifar10_train_valid_data(0) #32*32*3
option.X_test = X_test
option.X_train = X_train
option.y_train = y_train
option.y_test = y_test
option.comp_hidden = [1028, 256, 64, 15, 64, 256, 1028]
option.mid_layer = 3
option.mix_components = 8
option.epoch = 30
option.normal_portion = 90
dagmm = DAGMM(option)
#dagmm.train()