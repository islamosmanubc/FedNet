#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import random as rn
import os,sys


from sklearn.metrics import f1_score
from PIL import Image

# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results, from keras.io
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
from keras.preprocessing import image as kImage
from models.server import Server
import gc
from pathlib import Path
from datasets import getfiles


def getImgs(X_list,Y_list):

    # load training data
    num_imgs = len(X_list)
    X = np.zeros((num_imgs,240,320,3),dtype="float32")
    Y = np.zeros((num_imgs,240,320,1),dtype="float32")
    for i in range(len(X_list)):
        x = kImage.load_img(X_list[i],target_size = [240,320,3])
        x = kImage.img_to_array(x)
        X[i,:,:,:] = x
        
        x = kImage.load_img(Y_list[i], grayscale = True,target_size = [240,320])
        x = kImage.img_to_array(x)
        x /= 255.0
        x = np.floor(x)
        Y[i,:,:,0] = np.reshape(x,(240,320))
        
    return X, Y

### training function    
def test():
    lr = 1e-4
    batch_size = 10
    
    datasets = ['CDNet','DAVIS16','SEGTRACK','DUTS','ECCID']
    data_c =[2,3]
    server = Server(lr,len(datasets))
    server.loadWeights()
    data_counter = -1
    for data in datasets:
        data_counter = data_counter+1
        if data_counter not in data_c:
            continue
        data_name = data
        X,Y =  getfiles(data_name,'train')
    
        tasks = len(X)

        save_sample_path='../visresults/test'
        store_imgs = False
        ths = [0.2,0.4,0.6,0.8]
        epsilon = 1e-6
        iou_all = {}
        fm_all = {}
        for th in ths:
            iou_all.update({th:[]})
            fm_all.update({th:[]})

        f = open('../results/results_res_'+data_name+'_tr.txt','w')
        taskcounter = 0

        # Iterate over the batches of the dataset.
        for t in range(tasks):
    

            f = open('../results/results_res_'+data_name+'.txt','a')
            iou_t = {}
            fm_t = {}
            for th in ths:
                iou_t.update({th:[]})
                fm_t.update({th:[]})

            print("working on task %d"%(t,))
        
            Tx = X[t]
            Ty = Y[t]
            sub_dataset = int(len(Tx)/batch_size)
            
            taskcounter = taskcounter+1
            for step in range(batch_size+1):
                if step == batch_size:
                    x = Tx[step*sub_dataset:]
                    y = Ty[step*sub_dataset:]
             
                else:
                    x = Tx[step*sub_dataset:(step+1)*sub_dataset]
                    y = Ty[step*sub_dataset:(step+1)*sub_dataset]

                if len(x) == 0:
                    continue
                cx,cy = getImgs(x,y)
                yhat = np.zeros((len(cx),240,320,1))
                for i in range(len(cx)):
                    yhat[i,:,:,:] = server.model.predict(cx[i:i+1,:,:,:])[data_counter]
                actualy = cy
                max_iou = 0
                max_th = 0
                yp = np.zeros((len(yhat),240,320,1))

                for th in ths:
                    yp[yhat >= th] = 1
                    yp[yhat < th] = 0
            
                    pred = (yp == 1)
                    gt = (actualy == 1)
            
                    pred = np.reshape(pred,(len(yhat),240,320))
                    gt = np.reshape(gt,(len(yhat),240,320))

                    gtflat = gt.flatten()
                    predflat = pred.flatten()

                    fscore = f1_score(gtflat, predflat, average='binary')

                    intersection = np.sum((pred*gt),axis=1)
                    intersection = np.sum(intersection,axis=1)
                    union = np.sum(((pred+gt)>0),axis=1)
                    union = np.sum(union,axis=1)

                    iou = np.mean((intersection+epsilon)/(union+epsilon))
                    iou_t[th].append(iou)
                    fm_t[th].append(fscore)
                    iou_all[th].append(iou)
                    fm_all[th].append(fscore)
                    if iou > max_iou:
                        max_iou = iou
                        max_th = th


                if store_imgs:
                    for i in range(sub_dataset):
                        yd = yhat[i,:,:]
                        yd[yd>=max_th] = 1
                        yd[yd<max_th] = 0

                        yg = gt[i,:,:]
                        filepath = x[i]
                        p = Path(filepath).parts[-4:]
                        saveto = os.path.join(save_sample_path,data_name,p[0],p[1],p[2])
                        if not os.path.exists(saveto):
                            os.makedirs(saveto)
                        saveto = os.path.join(saveto,p[3])
    
                        d = cx[i,:,:,:]
                        yd = np.reshape(yd, (240,320))
                        yd = np.double(yd)
                        yg = np.double(yg)
                        d[:,:,0] = d[:,:,0]*0.4 + 0.6*(yd*255)
                        d[:,:,1] = d[:,:,1]*0.4 + 0.6*(yg*255)
                        d[:,:,2] = d[:,:,2]*0.4
                
                

                        d = np.uint8(d)
                        im = Image.fromarray(d)
                        im.save(saveto)

                del cx,cy,yp
            
            f.write('Task = ' +str(t) + '\n')
            fml = []
            ioul = []
            for th in ths:
                iou = np.mean(np.array(iou_t[th]))
                fm = np.mean(np.array(fm_t[th]))
                ioul.append(iou)
                fml.append(fm)
                f.write('Threshold = ' +str(th) + '\t')
                f.write('iou = ' +str(iou) + '\t')
                f.write('fm = ' +str(fm) + '\n')
            f.write('===================\n')
        
            print("evaluation of task %d"%(t,))
            print('Threshold = ' +str(ths))
            print('avg_iou = ' +str(ioul) )
            print('avg_fm = ' +str(fml) )
            f.close()
    
    
        f = open('../results/results_res_'+data_name+'.txt','a')
        f.write('===================\n')
        f.write('===================\n')
        f.write('===================\n')
        f.write('All dataset: \n')
        for th in ths:
            iou = np.mean(np.array(iou_all[th]))
            fm = np.mean(np.array(fm_all[th]))
            f.write('Threshold = ' +str(th) + '\t')
            f.write('iou = ' +str(iou) + '\t')
            f.write('fm = ' +str(fm) + '\n')
        f.close()


# =============================================================================
# Main func
# =============================================================================
test()
gc.collect()