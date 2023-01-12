import tensorflow as tf
from collections import OrderedDict
from models.Net import Net
import numpy as np
import torch
import keras
from keras.preprocessing import image as kImage
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score
from models.instance_normalization import InstanceNormalization
from keras.layers import BatchNormalization,concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model
from datasets import getfiles
import cv2
import gc
import os,sys


class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
            
        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)

            lossnew = self.compiled_loss(y, y_n, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(lossnew, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_n)
        # Return a dict mapping metric names to current value
        return lossnew

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)
def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)



class Client(object):
    def __init__(self,lr,i):
        self.model = Net(lr,(240,320,3))
        self.model = self.model.initModel()
        inp = self.model.input
        out = self.model.output
        self.createHead(out,i)
        self.index = i
        vision_model = CustomModel(inputs=inp, outputs=self.head, name='vision_model')
        opt = tf.keras.optimizers.RMSprop(lr = lr, rho=0.9, epsilon=1e-08, decay=0.)
        #c_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        c_loss = loss
        c_acc = acc
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        self.model = vision_model

    def setDataset(self,data,batch):
        self.dataset = data
        self.batch_size = batch
    def createHead(self,input,i):
        self.head= self.addHead(input,i)
        
    def addHead(self,input,i):
        d = self.UNetUp(input, [], 64,'d4Q'+str(i), 2, batch_norm=True,upsample=False,mix=False)
        out = Conv2D(1, (1,1), strides=1, padding='same', activation='sigmoid',name='d5_convQ'+str(i))(d)
        return out

    def UNetUp(self,x,skip,ch,block,num_rep,batch_norm=False,upsample=True,mix=True):
        
        if upsample:
            x = UpSampling2D(size=(2, 2))(x)
        if mix:
            x = concatenate([x,skip], axis=-1, name=block+'cat')
        x = Conv2D(ch, (3, 3), activation='relu', padding='same', name=block+'_convT')(x)
        if batch_norm:
            x = InstanceNormalization(name=block+'_instwT')(x)

        for k in range(num_rep):
            i = ord('a')
            i+=k
            c = chr(i)
            x = Conv2D(ch, (3, 3), activation='relu', padding='same', name=block+'_conv'+c+'T')(x)
            if batch_norm:
                x = InstanceNormalization(name=block+'_inst'+c+'T')(x)

        return x
    def getImgs(self,X_list,Y_list):

        # load training data
        num_imgs = len(X_list)
        X = np.zeros((num_imgs,240,320,3),dtype="float32")
        Y = np.zeros((num_imgs,240,320,1),dtype="float32")
        #Y = np.zeros((num_imgs,240,320,3),dtype="float32")
        for i in range(len(X_list)):
            x = kImage.load_img(X_list[i],target_size = [240,320,3])
            x = kImage.img_to_array(x)
            X[i,:,:,:] = x

            x = kImage.load_img(Y_list[i], grayscale = True,target_size = [240,320])
            x = kImage.img_to_array(x)
            x /= 255.0
            x = np.floor(x)
            Y[i,:,:,0] = np.reshape(x,(240,320))
        
        idx = list(range(X.shape[0]))
        np.random.shuffle(idx)
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]

    
        return X, Y

    def train(self,cycle):
        max_epoch = 5
        batch_size = self.batch_size
        X,Y =  getfiles(self.dataset)
        sub_dataset = int(len(X)/batch_size)
        
        #weights = np.load('../weights/client'+str(self.index)+'/weights_'+str(0)+'.npy', allow_pickle=True, encoding="latin1").tolist()
        #a = np.array(self.model.get_weights())
        #for k in range(len(weights)):
        #    a[k] = weights[k]
        #self.model.set_weights(a)

        f = open('../results/'+str(self.index)+'_client.txt','a')
        f.close()
        for epoch in range(max_epoch):
            print("\nStart of epoch %d" % (epoch,))
            i = 2
            # Iterate over the batches of the dataset.
            for step in range(batch_size):
            
                y = Y[step*sub_dataset:(step+1)*sub_dataset]
                x = X[step*sub_dataset:(step+1)*sub_dataset]
                cx,cy = self.getImgs(x,y)
                #self.model.fit(cx, cy,
                #    epochs=1, batch_size=1, verbose=2, shuffle = False)
                
                ccy = tf.convert_to_tensor(cy, dtype=tf.float32)

                for img in range(len(ccy)):
                    if img == 0:
                        lossx = self.model.train_step([cx[img:img+1],ccy[img:img+1]])
                    else:
                        lossx = lossx + self.model.train_step([cx[img:img+1],ccy[img:img+1]])
                #st = 
                tf.print('loss:',lossx,output_stream='file://C:/myCodes/paper10/results/'+str(self.index)+'_client.txt')
                tf.print('(',self.dataset,') step',step,' loss:',lossx,output_stream=sys.stdout)
                
                if i%1 == 0:
                    y = self.model.predict(cx)
                    img= y[0,:,:,:]*255
                    img = img.astype(np.uint8)
                    img1 = cx[0,:,:,:]
                    img1 = img1.astype(np.uint8)
                    img2 = cy[0,:,:,:3]*255
                    img2 = img2.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                    numpy_horizontal_concat = np.concatenate((img1, img,img2), axis=1)
                    #numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, img2), axis=1)
                    #cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
                    cv2.imwrite('../visresults/client/client_'+str(self.index)+'_'+str(epoch)+'_'+str(i)+'.png',numpy_horizontal_concat)
                    #cv2.imshow('frame',img)
                i = i+1
                del cx,cy
                gc.collect()

            weights = self.model.get_weights()
            a = np.array(weights)
        
            if not os.path.exists('../weights/client'+str(self.index)):
                os.mkdir('../weights/client'+str(self.index))
            np.save('../weights/client'+str(self.index)+'/weights_'+str(cycle)+'_'+str(epoch)+'.npy', a)
        f = open('../results/'+str(self.index)+'_client.txt','a')
        f.write('_:_')
        f.close()