import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import keras

def show_img(name:str, dataset):    
    if(name=="mnist"):
        show_mnist_image(dataset)        
    if(name=="cifar10"):
        plot_images_labels_prediction(dataset,dataset,0, 25)
    return

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

def show_most_abnormal_case(encoder, inital_center,datasetname, x_data, x_org):
    x_result = encoder.predict(x_data)
    i= 0
    error = np.zeros(x_result.shape[0])
    for x in x_result:
        sd = ((x-inital_center)**2).sum()
        error[i] = sd
        i=i+1

    x_list = x_org.tolist()
    cc = sorted(zip(error,x_list), reverse= True)
    (error,x_list) = zip(*cc)
    x_list = np.array(x_list)
    show_img(datasetname,x_list)
    return 

def show_most_normal_case(encoder, inital_center,datasetname, x_data, x_org):
    x_result = encoder.predict(x_data)
    i= 0
    error = np.zeros(x_result.shape[0])
    for x in x_result:
        sd = ((x-inital_center)**2).sum()
        error[i] = sd
        i=i+1

    x_list = x_org.tolist()
    cc = sorted(zip(error,x_list), reverse= False)
    (error,x_list) = zip(*cc)
    x_list = np.array(x_list)
    show_img(datasetname,x_list)
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
