from keras.datasets import mnist, cifar10
import numpy as np

class Loader(object):

    @staticmethod    
    def load_dataset(name:str,classnum:int):    
        if(name=="mnist"):
            return Loader.load_data_from_mnist(classnum)
        
        if(name=="cifar10"):
            return Loader.load_data_from_cifar10(classnum)
        return

    @staticmethod
    def load_data_from_mnist(classnum: int):
        (x_train, train_label), (x_test, test_label) = mnist.load_data()
        aa = [x for x,y in zip(x_train, train_label) if y==classnum]
        bb = [x for x,y in zip(x_train, train_label) if y==classnum]
        cc = [x for x,y in zip(x_test, test_label) if y==classnum]
        dd = [y for x,y in zip(x_test, test_label) if y==classnum]

        i =0
        for t in test_label:
            if t == classnum:
               test_label[i] = 0
            else:
                test_label[i] =1
            i =i+1

        
        Image_size = x_train.shape[1]
        cc = np.reshape(aa, [-1, Image_size, Image_size, 1])
        x_test_2 = np.reshape(x_test, [-1, Image_size, Image_size, 1])
        cc = cc.astype('float32') / 255
        x_test_2 = x_test_2.astype('float32') / 255

        x_test =  np.reshape(x_test, [-1, Image_size, Image_size, 1])
        x_test = x_test.astype('float32') / 255

        aa = np.reshape(aa, [-1, Image_size, Image_size, 1])
        aa = aa.astype('float32') / 255
        return np.array(cc),np.array(bb),x_test_2,test_label,1,x_test,aa
    
    @staticmethod
    def load_data_from_cifar10(classnum: int):
        (x_train, train_label), (x_test, test_label) = cifar10.load_data()
        aa = [x for x,y in zip(x_train, train_label) if y==classnum]
        bb = [x for x,y in zip(x_train, train_label) if y==classnum]
        cc = [x for x,y in zip(x_test, test_label) if y==classnum]
        dd = [y for x,y in zip(x_test, test_label) if y==classnum]
        
        i =0
        for t in test_label:
            if t == classnum:
               test_label[i] = 0
            else:
                test_label[i] =1
            i =i+1

        Image_size = x_train.shape[1]
        aa = np.reshape(np.array(aa), [-1, Image_size, Image_size, 3])
        x_test = np.reshape(np.array(x_test), [-1, Image_size, Image_size, 3])
        
        cc, bb, x_test_2 = Loader.normalize_data(np.array(aa).astype('float32'),np.array(bb).astype('float32'),x_test.astype('float32'), scale=np.float32(255))  
        cc, bb, x_test_2 = Loader.global_contrast_normalization(cc,bb,x_test_2,scale="l1")
        cc, bb, x_test_2 = Loader.rescale_to_unit_interval(cc,bb,x_test_2)
        
        x_test = x_test.astype('float32') / 255
        aa = aa.astype('float32') / 255
        return np.array(cc),np.array(bb),x_test_2,test_label,3,x_test,aa

    @staticmethod
    def global_contrast_normalization(X_train, X_val, X_test, scale="std"):
        assert scale in ("std", "l1", "l2")
        
        na = np.newaxis

        X_train_mean = np.mean(X_train, axis=(1, 2, 3),
                            dtype=np.float32)[:, na, na, na]
        X_val_mean = np.mean(X_val, axis=(1, 2, 3),
                            dtype=np.float32)[:, na, na, na]
        X_test_mean = np.mean(X_test, axis=(1, 2, 3),
                            dtype=np.float32)[:, na, na, na]

        X_train -= X_train_mean
        X_val -= X_val_mean
        X_test -= X_test_mean

        if scale == "std":
            X_train_scale = np.std(X_train, axis=(1, 2, 3),
                                dtype=np.float32)[:, na, na, na]
            X_val_scale = np.std(X_val, axis=(1, 2, 3),
                                dtype=np.float32)[:, na, na, na]
            X_test_scale = np.std(X_test, axis=(1, 2, 3),
                                dtype=np.float32)[:, na, na, na]
        if scale == "l1":
            X_train_scale = np.sum(np.absolute(X_train), axis=(1, 2, 3),
                                dtype=np.float32)[:, na, na, na]
            X_val_scale = np.sum(np.absolute(X_val), axis=(1, 2, 3),
                                dtype=np.float32)[:, na, na, na]
            X_test_scale = np.sum(np.absolute(X_test), axis=(1, 2, 3),
                                dtype=np.float32)[:, na, na, na]
        if scale == "l2":
            # equivalent to "std" since mean is subtracted beforehand
            X_train_scale = np.sqrt(np.sum(X_train ** 2, axis=(1, 2, 3),
                                        dtype=np.float32))[:, na, na, na]
            X_val_scale = np.sqrt(np.sum(X_val ** 2, axis=(1, 2, 3),
                                        dtype=np.float32))[:, na, na, na]
            X_test_scale = np.sqrt(np.sum(X_test ** 2, axis=(1, 2, 3),
                                        dtype=np.float32))[:, na, na, na]

        X_train /= X_train_scale
        X_val /= X_val_scale
        X_test /= X_test_scale
        
        return X_train, X_val, X_test

    @staticmethod
    def rescale_to_unit_interval(X_train, X_val, X_test):
        X_train_min = np.min(X_train)
        X_train_max = np.max(X_train)

        X_train -= X_train_min
        X_val -= X_train_min
        X_test -= X_train_min

        X_train /= (X_train_max - X_train_min)
        X_val /= (X_train_max - X_train_min)
        X_test /= (X_train_max - X_train_min)
        return X_train, X_val, X_test
            
    @staticmethod
    def normalize_data(X_train, X_val, X_test, mode="per channel", scale=None):
        """ normalize images per channel, per pixel or with a fixed value
        """

        if scale is None:
            if mode == "per channel":
                n_channels = np.shape(X_train)[1]
                scale = np.std(X_train, axis=(0, 2, 3)).reshape(1, n_channels, 1, 1)
            elif mode == "per pixel":
                scale = np.std(X_train, 0)
            elif mode == "fixed value":
                scale = 255.
            else:
                raise ValueError("Specify mode of scaling (should be "
                                    "'per channel', 'per pixel' or 'fixed value')")

        X_train /= scale
        X_val /= scale
        X_test /= scale
        return X_train, X_val, X_test
                        
