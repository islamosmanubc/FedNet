import tensorflow as tf
from collections import OrderedDict
from models.Net import Net
import numpy as np
import keras
from keras.preprocessing import image as kImage
import tensorflow.keras.backend as K
from models.instance_normalization import InstanceNormalization
from keras.layers import concatenate
from keras.layers.convolutional import Conv2D, UpSampling2D
from models.client import Client
import cv2
from datasets import getfiles
import gc
import sys

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



class CustomModel(keras.Model):
    def train_step1(self, data):
        x, y = data
        mod = 0
        y_o = self(x, training=True)  # Forward pass
        clients = len(y_o)
        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossnew = self.compiled_loss(y, y_n[mod],regularization_losses = self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(lossnew, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossold = 0
            for i in range(clients):
                if i == mod:
                    continue
                lossold = lossold + self.compiled_loss(y_o[i], y_n[i], regularization_losses=self.losses)
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)
            #let the constant value based on the number of clients
            loss = lossnew+0.5*lossold

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_n[mod])
        return loss
    def train_step2(self, data):
        x, y = data
        mod=1

        y_o = self(x, training=True)  # Forward pass
        clients = len(y_o)
        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(lossnew, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossold = 0
            for i in range(clients):
                if i == mod:
                    continue
                lossold = lossold + self.compiled_loss(y_o[i], y_n[i], regularization_losses=self.losses)
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)
            loss = lossnew+0.5*lossold

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_n[mod])
        return loss
    def train_step3(self, data):
        x, y = data
        mod=2

        y_o = self(x, training=True)  # Forward pass
        clients = len(y_o)
        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(lossnew, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossold = 0
            for i in range(clients):
                if i == mod:
                    continue
                lossold = lossold + self.compiled_loss(y_o[i], y_n[i], regularization_losses=self.losses)
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)
            loss = lossnew+0.5*lossold

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_n[mod])
        return loss
    def train_step4(self, data):
        x, y = data
        mod = 3
        y_o = self(x, training=True)  # Forward pass
        clients = len(y_o)
        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossnew = self.compiled_loss(y, y_n[mod],regularization_losses = self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(lossnew, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossold = 0
            for i in range(clients):
                if i == mod:
                    continue
                lossold = lossold + self.compiled_loss(y_o[i], y_n[i], regularization_losses=self.losses)
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)
            loss = lossnew+0.5*lossold

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_n[mod])
        return loss
    def train_step5(self, data):
        x, y = data
        mod = 4
        y_o = self(x, training=True)  # Forward pass
        clients = len(y_o)
        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossnew = self.compiled_loss(y, y_n[mod],regularization_losses = self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(lossnew, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        with tf.GradientTape() as tape:
            y_n = self(x, training=True)  # Forward pass
            lossold = 0
            for i in range(clients):
                if i == mod:
                    continue
                lossold = lossold + self.compiled_loss(y_o[i], y_n[i], regularization_losses=self.losses)
            lossnew = self.compiled_loss(y, y_n[mod], regularization_losses=self.losses)
            loss = lossnew+0.5*lossold

        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_n[mod])
        return loss
    
class Server(object):
    def __init__(self,lr,nheads):
        self.model = Net(lr,(240,320,3))
        self.model = self.model.initModel()
        self.num_heads = nheads
        inp = self.model.input
        out = self.model.output
        self.createHeads(out)
        self.lr = lr

        vision_model = CustomModel(inputs=inp, outputs=self.heads, name='vision_model')
        opt = tf.keras.optimizers.RMSprop(lr = lr, rho=0.9, epsilon=1e-08, decay=0.)
        #c_loss = tf.keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
        c_loss = loss
        c_acc = acc
        vision_model.compile(loss=c_loss, optimizer=opt, metrics=[c_acc])
        self.model = vision_model

        self.names = [weight.name for layer in self.model.layers for weight in layer.weights]

    def createHeads(self,input):
        self.heads = []
        for i in range(self.num_heads):
            self.heads.append(self.addHead(input,i))
        
    def addHead(self,input,i):
        d = self.UNetUp(input, [], 64,'d4Q'+str(i), 2, batch_norm=True,upsample=False,mix=False)
        out = Conv2D(1, (1,1), strides=1, padding='same', activation='sigmoid',name='d5_convQ'+str(i))(d)
        return out

    def changeTrainabeLayers(self,mod):
        for layer in self.model.layers:
            for weight in layer.weights:
                if weight.name[0] == 'd':
                    if 'Q' in weight.name:
                        decoder_layer = int(weight.name.split('Q')[1][0])
                        if decoder_layer != mod:
                            layer.trainable = False
                        else:
                            layer.trainable = True
                        break
        
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

    def spawnClients(self,reset=False):
        self.weightsName = OrderedDict()
        self.weights = self.model.get_weights()
        for name, weight in zip(self.names, self.weights):
            self.weightsName.update({name: weight})
        
        if not reset:
            self.clients = []
            for i in range(self.num_heads):
                self.clients.append(Client(self.lr,i))
                self.initializeClient(self.clients[-1])
            self.clients[0].setDataset('CDNet',100)
            self.clients[1].setDataset('DAVIS16',50)
            self.clients[2].setDataset('SEGTRACK',5)
            self.clients[3].setDataset('DUTS',500)
            self.clients[4].setDataset('ECCID',50)
        else:
            for i in range(self.num_heads):
                self.initializeClient(self.clients[-1])
            
    def initializeClient(self,client):
        targetNames = [weight.name for layer in client.model.layers for weight in layer.weights]
        a = []
        for name in targetNames:
            a.append(self.weightsName[name])
        client.model.set_weights(a)

    def gatherClients(self):
        #retrieve clients from different PCs
        x=1

    def getImgs(self,X_list,Y_list):
        
        num_imgs = len(X_list)
        X = np.zeros((num_imgs,240,320,3),dtype="float32")
        for i in range(len(X_list)):
            x = kImage.load_img(X_list[i],target_size = [240,320,3])
            x = kImage.img_to_array(x)
            X[i,:,:,:3] = x
        idx = list(range(X.shape[0]))
        np.random.shuffle(idx)
        X = X[idx]
        return X

    def getGT(self,X,Y_list,mod):
        num_imgs = len(X)
        Y = np.zeros((num_imgs,240,320,mod+1),dtype="float32")
        for i in range(len(X)):
            Y[i,:,:,0] = Y_list[i,:,:,0]
        idx = list(range(X.shape[0]))
        np.random.shuffle(idx)
        np.random.shuffle(idx)
        X = X[idx]
        Y = Y[idx]
        return X, Y

    def trainServer(self,cycle):
        X,Y =  getfiles('YOVOS')
        max_epoch = 1
        batch_size = 1000
        sub_dataset = int(len(X)/batch_size)
        train_steps = []
        train_steps.append(self.model.train_step1)
        train_steps.append(self.model.train_step2)
        train_steps.append(self.model.train_step3)
        train_steps.append(self.model.train_step4)
        train_steps.append(self.model.train_step5)
        for epoch in range(max_epoch):
            print("\nStart of epoch %d" % (epoch,))
            i = 0
            # Iterate over the batches of the dataset.
            for step in range(batch_size):
            
                y = Y[step*sub_dataset:(step+1)*sub_dataset]
                x = X[step*sub_dataset:(step+1)*sub_dataset]
                mod = i%self.num_heads
                cx = self.getImgs(x,y)
                #print(self.model.summary())
                self.changeTrainabeLayers(mod)
                #print(self.model.summary())
                self.model.train_step = train_steps[mod]
                cy = self.clients[mod].model.predict(cx)
                #cx,cy = self.getGT(cx,cy,mod)
                
                ccy = tf.convert_to_tensor(cy, dtype=tf.float32)
                for img in range(len(ccy)):
                    printable = self.model.train_step([cx[img:img+1],ccy[img:img+1]])
                tf.print('step:',str(step),' loss:',printable, output_stream=sys.stdout)
                #self.model.fit(cx, cy,
                #    epochs=1, batch_size=1, verbose=2, shuffle = False)
                if i%1 == 0:
                    y = self.model.predict(cx)
                    img= y[mod][0,:,:,:]*255
                    img = img.astype(np.uint8)
                    img1 = cx[0,:,:,:]
                    img1 = img1.astype(np.uint8)
                    img2 = cy[0,:,:,:1]*255
                    img2 = img2.astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                    numpy_horizontal_concat = np.concatenate((img1, img,img2), axis=1)
                    #numpy_horizontal_concat = np.concatenate((numpy_horizontal_concat, img2), axis=1)
                    #cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
                    cv2.imwrite('../visresults/server/server_'+str(epoch)+'_'+str(i)+'.png',numpy_horizontal_concat)
                    #cv2.imshow('frame',img)
                i = i+1
                del cx,cy,ccy
                gc.collect()
                if step > 200:
                    break
            weights = self.model.get_weights()
            a = np.array(weights)
            np.save('../weights/server_weights_'+str(cycle)+'_'+str(epoch)+'.npy', a)
            
    def loadWeights(self):
        weights = np.load('../weights/server_weights_9_0.npy', allow_pickle=True, encoding="latin1").tolist()
        a = np.array(self.model.get_weights())
        for k in range(len(weights)):
            a[k] = weights[k]
        self.model.set_weights(a)

    def trainClients(self,cycle):
        for i in range(len(self.clients)):
            self.clients[i].train(cycle)

    def trainFedNet(self):
        max_epoch = 10
        for i in range(max_epoch):
            self.spawnClients(i!=0)
            #self.trainServer()
            self.trainClients(i)
            self.gatherClients()
            self.trainServer(i)