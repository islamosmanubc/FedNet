
import os,sys
import numpy as np
import keras, glob

def getfiles(dataset,split = 'train'):
       
    Y_list = []
    X_list = []
    basepath = 'C:/myCodes/DynamicallyGrowingTree/'

    if dataset == 'YOVOS':
        if split == 'train':
            pathf = basepath+'datasets/train/JPEGImages/base_foreground_segmentation/yovos'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/train/'
        if split == 'test':
            pathf = basepath+'datasets/test/JPEGImages/base_foreground_segmentation/yovos'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/test/'
        for task in tasks:
            Y_list.append(glob.glob(os.path.join(pathf,'Annotations/base_foreground_segmentation/yovos', task, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/base_foreground_segmentation/yovos', task,'*.jpg')))

    elif dataset == 'SEGTRACK':
        if split == 'train':
            pathf = basepath+'datasets/segtrackv212/JPEGImages'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/segtrackv212'
            for task in tasks:
                Y_list.append(glob.glob(os.path.join(pathf,'GroundTruth',task,'*.png')))
                X_list.append(glob.glob(os.path.join(pathf,'JPEGImages',task, 'input','*.png')))
        
        if split == 'test':
            pathf = basepath+'datasets/segtrackv212/segrestest/t0'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/segtrackv212'
            for task in tasks:
                Y_list.append(glob.glob(os.path.join(pathf,'GroundTruth',task,'*.png')))
                X_list.append(glob.glob(os.path.join(pathf,'full',task, 'input','*.png')))
        
        for k in range(len(Y_list)):
           Y_list_temp = []
           for i in range(len(X_list[k])):
               X_name = os.path.basename(X_list[k][i])
               X_name = X_name.split('.')[0]
               for j in range(len(Y_list[k])):
                   Y_name = os.path.basename(Y_list[k][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][j])
                       break
           Y_list[k] = Y_list_temp
    elif dataset == 'DAVIS16':
        if split == 'train':
            pathf = basepath+'datasets/DAVIS/480p'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/DAVIS'
            for task in tasks:
                Y_list.append(glob.glob(os.path.join(pathf,'480pY',task, '*.png')))
                X_list.append(glob.glob(os.path.join(pathf,'480p', task,'input','*.jpg')))
        if split == 'test':
            pathf = basepath+'datasets/DAVIS/480ptest'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/DAVIS'
            for task in tasks:
                Y_list.append(glob.glob(os.path.join(pathf,'480pY',task, '*.png')))
                X_list.append(glob.glob(os.path.join(pathf,'480ptest', task,'input','*.jpg')))
        for k in range(len(Y_list)):
           Y_list_temp = []
           for i in range(len(X_list[k])):
               X_name = os.path.basename(X_list[k][i])
               X_name = X_name.split('.')[0]
               
               for j in range(len(Y_list[k])):
                   Y_name = os.path.basename(Y_list[k][j])
                   Y_name = Y_name.split('.')[0]
                   if (Y_name == X_name):
                       Y_list_temp.append(Y_list[k][j])
                       break
           Y_list[k] = Y_list_temp
    if dataset == 'CDNet':
        if split == 'train':
            pathf = basepath+'datasets/train/JPEGImages/continual_foreground_segmentation/cdnet'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/train/'
        else:
            pathf = basepath+'datasets/test/JPEGImages/continual_foreground_segmentation/cdnet'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/test/'
        for task in tasks:
            Y_list.append(glob.glob(os.path.join(pathf,'Annotations/continual_foreground_segmentation/cdnet', task, '*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/continual_foreground_segmentation/cdnet', task,'*.jpg')))
    if dataset == 'DAVIS17':
        if split == 'train':
            pathf = basepath+'datasets/train/JPEGImages/fewshot_foreground_segmentation/davis'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/train/'
        else:
            pathf = basepath+'datasets/test/JPEGImages/fewshot_foreground_segmentation/davis'
            tasks = os.listdir(pathf)
            pathf = basepath+'datasets/test/'
        for task in tasks:
            if split == 'train':
                Y_list.append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis_5shot', task, '*.png')))
                X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/fewshot_foreground_segmentation/davis_5shot', task,'*.jpg')))
            if split == 'test':
                Y_list.append(glob.glob(os.path.join(pathf,'Annotations/fewshot_foreground_segmentation/davis', task, '*.png')))
                X_list.append(glob.glob(os.path.join(pathf,'JPEGImages/fewshot_foreground_segmentation/davis', task,'*.jpg')))

    if dataset == 'DUTS':
        if split == 'train':
            pathf = basepath+'datasets/duts/DUTS-TR'
            Y_list.append(glob.glob(os.path.join(pathf,'DUTS-TR-Mask','*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'DUTS-TR-Image','*.jpg')))
        else:
            pathf = basepath+'datasets/duts/DUTS-TE'
            Y_list.append(glob.glob(os.path.join(pathf,'DUTS-TE-Mask','*.png')))
            X_list.append(glob.glob(os.path.join(pathf,'DUTS-TE-Image','*.jpg')))
    if dataset == 'ECCID':
        pathf = basepath+'datasets/eccid'
        Y_list.append(glob.glob(os.path.join(pathf,'ground_truth_mask','*.png')))
        X_list.append(glob.glob(os.path.join(pathf,'images','*.jpg')))
    
    xlist = []
    ylist = []

    if split == 'test':
        for k in range(len(X_list)):
            for i in range(len(Y_list[k])):
                xlist.append(X_list[k][i])
                ylist.append(Y_list[k][i])
            
        X_list = xlist
        Y_list = ylist
    
        X_list = np.array(X_list)
        Y_list = np.array(Y_list)
        idx = list(range(X_list.shape[0]))
        np.random.shuffle(idx)
        np.random.shuffle(idx)
        X_list = X_list[idx]
        Y_list = Y_list[idx]

    else:
        for k in range(len(X_list)):
            xlist.append([])
            ylist.append([])
            for i in range(len(Y_list[k])):
                xlist[-1].append(X_list[k][i])
                ylist[-1].append(Y_list[k][i])
            
        X_list = xlist
        Y_list = ylist
        X_list = np.array(X_list)
        Y_list = np.array(Y_list)
    
    return X_list,Y_list