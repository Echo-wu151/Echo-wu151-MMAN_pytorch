import numpy as np
import nibabel as nib
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import pdb
import itertools
import os
from PIL import Image




#读取每个nii文件，并将其切割为一张张图片
def build_set(imageData, numModalities) :
    num_classes = 4
    # Extract patches from input volumes and ground truth
    imageData_1 = np.squeeze(imageData[0,:,:,:])
    imageData_2 = np.squeeze(imageData[1,:,:,:])
    if (numModalities==3):
        imageData_3 = np.squeeze(imageData[2,:,:,:])
        imageData_g = np.squeeze(imageData[3,:,:,:])
    if (numModalities == 2):
        imageData_g = np.squeeze(imageData[2, :, :, :])#将所有的一维移除掉，比如这里就变成240，240，48（从四维变成了三维）
    patchesList_1 = []
    for i in range(imageData_1.shape[2]):
        patchesList_1.append(imageData_1[:,:,i])
    patchesList_2 = []
    for i in range(imageData_2.shape[2]):
        patchesList_2.append(imageData_2[:,:,i])
    patchesList_3 = []
    if (numModalities==3):
        for i in range(imageData_3.shape[2]):
            patchesList_3.append(imageData_3[:,:,i])
    patchesList_g = []
    for i in range(imageData_2.shape[2]):
        patchesList_g.append(imageData_g[:,:,i])
    if (numModalities==3):

        return np.array(patchesList_1),np.array(patchesList_2),np.array(patchesList_3),np.array(patchesList_g)
    if (numModalities==2):

        return np.array(patchesList_1),np.array(patchesList_2),np.array(patchesList_g)


##读取训练数据
def load_data_trainG(paths, pathg, imageNames, numSamples, numModalities):
    samplesPerImage = int(numSamples / len(imageNames))
    X1_train = []
    X2_train = []
    X3_train = []
    Y_train = []
    for num in range(len(imageNames)):
        imageData_1 = nib.load(paths[0] + os.sep + imageNames[num]).get_data()
        imageData_2 = nib.load(paths[1] + os.sep  + imageNames[num]).get_data()
        if (numModalities==3):
            imageData_3 = nib.load(paths[2] +os.sep  + imageNames[num]).get_data()
        imageData_g = nib.load(pathg + os.sep  + imageNames[num]).get_data()

        # #CLAHE
        # imageData_1 = nib.load(paths[0] + os.sep + imageNames[num]).get_data()[:,:,:,0]
        # imageData_2 = nib.load(paths[1] + os.sep  + imageNames[num]).get_data()[:,:,:,0]
        # if (numModalities==3):
        #     imageData_3 = nib.load(paths[2] +os.sep  + imageNames[num]).get_data()[:,:,:,0]
        # imageData_g = nib.load(pathg + os.sep  + imageNames[num]).get_data()

        num_classes = len(np.unique(imageData_g))

        if (numModalities == 2):
            imageData = np.stack((imageData_1, imageData_2, imageData_g))
            patchesList_1,patchesList_2,patchesList_g = build_set(imageData, numModalities)
        if (numModalities == 3):
            imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
            patchesList_1,patchesList_2,patchesList_3,patchesList_g = build_set(imageData, numModalities)

        img_shape = imageData.shape
        #接下来开始打乱顺序
        idx = np.arange(patchesList_1.shape[0])
        np.random.shuffle(idx)
        patchesList_1=patchesList_1[idx,]
        patchesList_2=patchesList_2[idx,]            
        X1_train.append(patchesList_1)
        X2_train.append(patchesList_2)

        if (numModalities == 3):
            patchesList_3=patchesList_3[idx,]
            X3_train.append(patchesList_3)
        patchesList_g=patchesList_g[idx,]
        Y_train.append(patchesList_g)
        
        del patchesList_1
        del patchesList_2
        del patchesList_3
        del patchesList_g
    X1_train=np.array(X1_train)
    X2_train=np.array(X2_train)
    
    X1_train=X1_train.reshape(X1_train.shape[0]*X1_train.shape[1],X1_train.shape[2],X1_train.shape[3])
    X2_train=X2_train.reshape(X2_train.shape[0]*X2_train.shape[1],X2_train.shape[2],X2_train.shape[3])
    idx = np.arange(X1_train.shape[0])
    np.random.shuffle(idx)
    if (numModalities == 2):
        X_train=np.stack((X1_train[idx],X2_train[idx]))
    if (numModalities == 3):
        X3_train=np.array(X3_train)
        X3_train=X3_train.reshape(X3_train.shape[0]*X3_train.shape[1],X3_train.shape[2],X3_train.shape[3])
        X_train=np.stack((X1_train[idx],X2_train[idx],X3_train[idx]))
    Y_train=np.array(Y_train)
    Y_train=Y_train.reshape(Y_train.shape[0]*Y_train.shape[1],Y_train.shape[2],Y_train.shape[3])
    Y_train=Y_train[idx]
    return X_train.transpose(1,0,2,3), Y_train, img_shape



#读取验证集数据
def load_data_test(paths, pathg, imageNames, numSamples, numModalities):
    samplesPerImage = int(numSamples / len(imageNames))
    X1_train = []
    X2_train = []
    X3_train = []
    Y_train = []
    for num in range(len(imageNames)):
        imageData_1 = nib.load(paths[0] + os.sep + imageNames[num]).get_data()
        imageData_2 = nib.load(paths[1] + os.sep  + imageNames[num]).get_data()
        if (numModalities==3):
            imageData_3 = nib.load(paths[2] +os.sep  + imageNames[num]).get_data()
        imageData_g = nib.load(pathg + os.sep  + imageNames[num]).get_data()

        # imageData_1 = nib.load(paths[0] + os.sep + imageNames[num]).get_data()[:,:,:,0]
        # imageData_2 = nib.load(paths[1] + os.sep  + imageNames[num]).get_data()[:,:,:,0]
        # if (numModalities==3):
        #     imageData_3 = nib.load(paths[2] +os.sep  + imageNames[num]).get_data()[:,:,:,0]
        # imageData_g = nib.load(pathg + os.sep  + imageNames[num]).get_data()

        num_classes = len(np.unique(imageData_g))

        if (numModalities == 2):
            imageData = np.stack((imageData_1, imageData_2, imageData_g))
            patchesList_1,patchesList_2,patchesList_g = build_set(imageData, numModalities)
        if (numModalities == 3):
            imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
            patchesList_1,patchesList_2,patchesList_3,patchesList_g = build_set(imageData, numModalities)

        img_shape = imageData.shape

        patchesList_1=patchesList_1
        patchesList_2=patchesList_2           
        X1_train.append(patchesList_1)
        X2_train.append(patchesList_2)

        if (numModalities == 3):
            patchesList_3=patchesList_3
            X3_train.append(patchesList_3)
        patchesList_g=patchesList_g
        Y_train.append(patchesList_g)
        
        del patchesList_1
        del patchesList_2
        del patchesList_3
        del patchesList_g
    X1_train=np.array(X1_train)
    X2_train=np.array(X2_train)
    
    X1_train=X1_train.reshape(X1_train.shape[0]*X1_train.shape[1],X1_train.shape[2],X1_train.shape[3])
    X2_train=X2_train.reshape(X2_train.shape[0]*X2_train.shape[1],X2_train.shape[2],X2_train.shape[3])

    if (numModalities == 2):
        X_train=np.stack((X1_train,X2_train))
    if (numModalities == 3):
        X3_train=np.array(X3_train)
        X3_train=X3_train.reshape(X3_train.shape[0]*X3_train.shape[1],X3_train.shape[2],X3_train.shape[3])
        X_train=np.stack((X1_train,X2_train,X3_train))
    Y_train=np.array(Y_train)
    Y_train=Y_train.reshape(Y_train.shape[0]*Y_train.shape[1],Y_train.shape[2],Y_train.shape[3])
    Y_train=Y_train
    return X_train.transpose(1,0,2,3), Y_train, img_shape
