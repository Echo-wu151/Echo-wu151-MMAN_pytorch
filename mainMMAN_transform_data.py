# "python.linting.enabled": false
from os.path import isfile, join
import os
import numpy as np
# from sampling import reconstruct_volume
# from sampling import my_reconstruct_volume
from sampling import load_data_trainG
from sampling import load_data_test

import torch
import torch.nn as nn
# from HyperDenseNet import *
from model import *
from medpy.metric.binary import dc,hd
import argparse

import pdb
from torch.autograd import Variable
from progressBar import printProgressBar
import nibabel as nib
from metric import SegmentationMetric
from dataset import CustomNpzFolderLoader,CustomNpzFolderLoader_test
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from plotResults import save_result
metric=SegmentationMetric(4)


def evaluateSegmentation(gt,pred):
    pred = pred.astype(dtype='int')
    gt=gt.astype(dtype='int')
    # print(np.unique(pred))
    # print('pred',pred.shape)
    numClasses = np.unique(gt)
    # print('gt',numClasses)

    dsc = np.zeros((1, len(numClasses) - 1))

    for i_n in range(1,len(numClasses)):
        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==i_n)]=1
        # print(gt_c[:,0:18,0:18])
        y_c[np.where(pred==i_n)]=1
        # print(y_c[:,0:18,0:18])
        dsc[0, i_n - 1] = dc(gt_c, y_c)
    return dsc
    
def numpy_to_var(x):
    torch_tensor = torch.from_numpy(x).type(torch.FloatTensor)#变成张量，与原矩阵共享内存，因此改变张量也会改变原矩阵
    
    if torch.cuda.is_available():
        torch_tensor = torch_tensor.cuda()
    return Variable(torch_tensor)
    


def inference(network, moda_n, moda_g, imageNames, epoch, folder_save, number_modalities,batch_size,CE_loss):
    '''root_dir = './Data/MRBrainS/DataNii/'
    model_dir = 'model'

    moda_1 = root_dir + 'Training/T1'
    moda_2 = root_dir + 'Training/T1_IR'
    moda_3 = root_dir + 'Training/T2_FLAIR'
    moda_g = root_dir + 'Training/GT'''
    network.eval()
    softMax = nn.Softmax()
    numClasses = 4
    if torch.cuda.is_available():
        softMax.cuda()
        network.cuda()
    accAll = np.zeros((len(imageNames), numClasses ))
    dscAll = np.zeros((len(imageNames), numClasses - 1))  # 1 class is the background!!
    val_loss=0
    for i_s in range(len(imageNames)):

        x_train, y_train, img_shape = load_data_test(moda_n, moda_g, imageNames, 100, number_modalities)
        data_transforms= transforms.Compose(
                [transforms.ToTensor(),
                 ])
        image_datasets =CustomNpzFolderLoader(x_train, y_train,data_transforms=data_transforms)
        dataloaders =DataLoader(image_datasets,
                        batch_size=batch_size,
                        shuffle=False)
        dataset_sizes = len(image_datasets)
        
        pred_numpy = np.zeros((0,numClasses,x_train.shape[2],x_train.shape[3]))
        pred_numpy = np.vstack((pred_numpy, np.zeros((x_train.shape[0], numClasses, x_train.shape[2],x_train.shape[3]))))
        totalOp = len(imageNames)*x_train.shape[0]
        running_loss=0.0
        for i_p,(train_data,label) in enumerate(dataloaders):

        # for i_p in range(x_train.shape[0]):
            # pred = network(numpy_to_var(x_train[i_p,:,:,:].reshape(1,number_modalities,x_train.shape[2],x_train.shape[3])))
            if torch.cuda.is_available():
                train_data = train_data.cuda().type(torch.cuda.FloatTensor)
            pred = network(train_data)
            

            # To adapt CE to 3D
            # LOGITS:
            #这里没明白
            
            pred_y = softMax(pred)
            pred_numpy[i_p*batch_size:(i_p+1)*batch_size,:,:,:] = pred_y.cpu().data.numpy()

            printProgressBar(i_s * ((totalOp + 0.0) / len(imageNames)) + i_p + 1, totalOp,
                             prefix="[Validation] ",
                             length=15)
            pred = pred.permute(0,2,3,1).contiguous()#permute是pytorch中的高维转置函数，返回的是张量
            pred = pred.view(pred.numel() // numClasses, numClasses)#numel返回数量
            CE_loss_batch = CE_loss(pred, label.view(-1).type(torch.cuda.LongTensor))
            running_loss+=CE_loss_batch.cpu().data.numpy()

        val_loss+=running_loss/dataset_sizes

        pred_classes = np.argmax(pred_numpy, axis=1)

        pred_classes = pred_classes.reshape((len(pred_classes), x_train.shape[2], x_train.shape[3]))
        dsc = evaluateSegmentation(y_train,pred_classes)
        dscAll[i_s, :] = dsc
        metric.addBatch(pred_classes.astype(dtype='int'), y_train.astype(dtype='int'))
        acc = metric.classPixelAccuracy()
        accAll[i_s, :] = acc

    return np.mean(np.array(dscAll),axis=0),np.mean(np.array(accAll),axis=0),val_loss/len(imageNames)

def runTraining(opts):
    print('' * 41)
    print('~' * 50)
    print('~~~~~~~~~~~~~~~~~  PARAMETERS ~~~~~~~~~~~~~~~~')
    print('~' * 50)
    print('  - Number of image modalities: {}'.format(opts.numModal))
    print('  - Number of classes: {}'.format(opts.numClasses))
    print('  - Directory to load images: {}'.format(opts.root_dir))
    for i in range(len(opts.modality_dirs)):
        print('  - Modality {}: {}'.format(i+1,opts.modality_dirs[i]))
    print('  - Directory to save results: {}'.format(opts.save_dir))
    print('  - To model will be saved as : {}'.format(opts.modelName))
    print('-' * 41)
    print('  - Number of epochs: {}'.format(opts.numClasses))
    print('  - Batch size: {}'.format(opts.batchSize))
    print('  - Number of samples per epoch: {}'.format(opts.numSamplesEpoch))
    print('  - Learning rate: {}'.format(opts.l_rate))
    print('' * 41)

    print('-' * 41)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 41)
    print('' * 40)

    samplesPerEpoch = opts.numSamplesEpoch
    batch_size = opts.batchSize

    lr = opts.l_rate
    epoch = opts.numEpochs
    
    root_dir = opts.root_dir
    model_name = opts.modelName


    if not (len(opts.modality_dirs)== opts.numModal): raise AssertionError

    moda_1 = root_dir + 'Training/' + opts.modality_dirs[0]
    moda_2 = root_dir + 'Training/' + opts.modality_dirs[1]

    if (opts.numModal == 3):
        moda_3 = root_dir + 'Training/' + opts.modality_dirs[2]

    moda_g = root_dir + 'Training/GT'

    print(' --- Getting image names.....')
    print(' - Training Set: -')
    if os.path.exists(moda_1):
        imageNames_train = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f))]
        imageNames_train.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_train)):
            print(' - {}'.format(imageNames_train[i])) 
    else:
        raise Exception(' - {} does not exist'.format(moda_1))

    moda_1_val = root_dir + 'Validation/' + opts.modality_dirs[0]
    moda_2_val = root_dir + 'Validation/' + opts.modality_dirs[1]

    if (opts.numModal == 3):
        moda_3_val = root_dir + 'Validation/' + opts.modality_dirs[2]
    moda_g_val = root_dir + 'Validation/GT'

    print(' --------------------')
    print(' - Validation Set: -')
    if os.path.exists(moda_1):
        imageNames_val = [f for f in os.listdir(moda_1_val) if isfile(join(moda_1_val, f))]
        imageNames_val.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_val)):
            print(' - {}'.format(imageNames_val[i])) 
    else:
        raise Exception(' - {} does not exist'.format(moda_1_val))
          
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = opts.numClasses
    
    # Define HyperDenseNet
    # To-Do. Get as input the config settings to create different networks
    if (opts.numModal == 2):
        hdNet = MMAN()
    if (opts.numModal == 3):
        # hdNet = HyperDenseNet(num_classes)
        hdNet = MMAN()
    #

    '''try:
        hdNet = torch.load(os.path.join(model_name, "Best_" + model_name + ".pkl"))
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        hdNet.cuda()
        softMax.cuda()
        CE_loss.cuda()

    # To-DO: Check that optimizer is the same (and same values) as the Theano implementation
    optimizer = torch.optim.Adam(hdNet.parameters(), lr=lr, betas=(0.9, 0.999))
    
    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    numBatches = int(samplesPerEpoch/batch_size)
    dscAll = []
    if (opts.numModal == 2):
        imgPaths = [moda_1, moda_2]

    if (opts.numModal == 3):
        imgPaths = [moda_1, moda_2, moda_3]
    x_train, y_train, img_shape = load_data_trainG(imgPaths, moda_g, imageNames_train, samplesPerEpoch, opts.numModal) # hardcoded to read the first file. Loop this to get all files. Karthik
    data_transforms= transforms.Compose(
                [transforms.ToTensor(),
                # transforms.RandomRotation(15),
                # transforms.Resize(),
                 transforms.RandomHorizontalFlip()
                 ])
    image_datasets =CustomNpzFolderLoader(x_train, y_train,data_transforms=data_transforms)
    dataloaders =DataLoader(image_datasets,
                       batch_size=batch_size,
                       shuffle=False)
    dataset_sizes = len(image_datasets) 

    begin_time = time.time()
    best_acc = 0.0
    arr_dc = []
    arr_loss = []
    val_loss=[]

    for e_i in range(epoch):
        epoch_begin_time = time.time()
        print("epoch({}) begins:".format(e_i))
        hdNet.train()
        epoch_dc=[]
        lossEpoch = []
        running_loss=0.0
        running_dc=[]
        for train_data,label in dataloaders:
            optimizer.zero_grad()#这两句代码的作用是等同的
            hdNet.zero_grad()
            # MRIs         = numpy_to_var(train_data)

            # Segmentation = numpy_to_var(label)
            Segmentation=label
            if torch.cuda.is_available():
                MRIs = train_data.cuda().type(torch.cuda.FloatTensor)
            segmentation_prediction = hdNet(MRIs)
            
            predClass_y = softMax(segmentation_prediction)
            pred_classes = np.argmax(predClass_y.cpu().data.numpy(), axis=1)

            pred_classes = pred_classes.reshape((len(pred_classes), x_train.shape[2], x_train.shape[3]))
            dsc_train = evaluateSegmentation(Segmentation.cpu().data.numpy(),pred_classes)
            metric.addBatch(pred_classes.astype(dtype='int'), Segmentation.cpu().data.numpy().astype(dtype='int'))
            acc = metric.pixelAccuracy()
            # To adapt CE to 3D
            # LOGITS:
            #这里没明白
            segmentation_prediction = segmentation_prediction.permute(0,2,3,1).contiguous()#permute是pytorch中的高维转置函数，返回的是张量
            segmentation_prediction = segmentation_prediction.view(segmentation_prediction.numel() // num_classes, num_classes)#numel返回数量
            CE_loss_batch = CE_loss(segmentation_prediction, Segmentation.view(-1).type(torch.cuda.LongTensor))
            loss = CE_loss_batch
            loss.backward()
            optimizer.step()
            running_loss+=CE_loss_batch.cpu().data.numpy()
            if dsc_train.shape[1]==3:
                running_dc.append(dsc_train)
            
            # lossEpoch.append(CE_loss_batch.cpu().data.numpy())
            # print('dsc_train',dsc_train)
            # print('acc',acc)

            # printProgressBar(b_i + 1, numBatches,
            #                  prefix="[Training] Epoch: {} ".format(e_i),
            #                  length=15)

            # del MRIs
            # del Segmentation
            del segmentation_prediction
            del predClass_y
        epoch_loss=running_loss/dataset_sizes
        # print(np.mean(np.array(running_dc),axis=0))
        epoch_dc=np.mean(np.array(running_dc),axis=0)

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        with open('{}/{}_train_loss.txt'.format(model_name,model_name), 'a') as f:
            message = (
                        "-*-" * 20 + '\n' +
                        'epoch={},  Train_Loss={:.4f}, DSC:1:{},2:{},3:{}'.format(
                            e_i,  epoch_loss, epoch_dc[0][0],epoch_dc[0][1],epoch_dc[0][2]))
            print(message)
            f.write(message + '\n')
        arr_loss.append(epoch_loss)
        arr_dc.append(epoch_dc)
        # np.save(os.path.join(model_name, model_name + '_loss.npy'), lossEpoch)
        epoch_cost_time = time.time() - epoch_begin_time
        print("epoch{} complete in {:.0f}m {:0f}s".format(
            e_i, epoch_cost_time // 60, epoch_cost_time % 60))
        # print(' Epoch: {}, loss: {}, dsc_train: {} per class: 1({}) 2({}) 3({})'.format(e_i,np.mean(epoch_loss),np.mean(epoch_dc),epoch_dc[0][0],epoch_dc[0][1],epoch_dc[0][2]))

        if (e_i%1)==0:

            if (opts.numModal == 2):
                moda_n = [moda_1_val, moda_2_val]
            if (opts.numModal == 3):
                moda_n = [moda_1_val, moda_2_val, moda_3_val]
            # dsc = inference(hdNet,moda_n, moda_g_val, imageNames_val,e_i, opts.save_dir,opts.numModal)
            dsc,acc,epoch_val_loss = inference(hdNet,moda_n, moda_g_val, imageNames_val,e_i, opts.save_dir,opts.numModal,batch_size,CE_loss)
            dscAll.append(dsc)
            val_loss.append(epoch_val_loss)
            # print('val_loss:{}, Metrics: DSC(mean): {} per class: 1({}) 2({}) 3({}), acc(mean): {} per class: 1({}) 2({}) 3({})'.format(epoch_val_loss,np.mean(dsc),dsc[0],dsc[1],dsc[2],np.mean(acc),acc[1],acc[2],acc[3]))
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            with open('{}/{}_val_loss.txt'.format(model_name,model_name), 'a') as f:
                message_val = ("-*-" * 20 + '\n' +
                        'epoch={},  Val_Loss={:.4f}, DSC:1:{},2:{},3:{}'.format(
                            e_i,  epoch_val_loss, dsc[0],dsc[1],dsc[2]))
                print(message_val)
                f.write(message_val + '\n')

            np.save(os.path.join(model_name, model_name + '_DSCs.npy'), dscAll)

        d1 = np.mean(dsc)
        if (d1>0.60):
            if not os.path.exists(model_name):
                os.makedirs(model_name)

            torch.save(hdNet, os.path.join(model_name, "Best2_" + model_name + ".pkl"))
            flag=0
            if len(val_loss)>2:
                if abs(val_loss[-1]-val_loss[-2])<1e-5:
                    flag+=1
            if flag>=10:
                torch.save(hdNet, os.path.join(model_name, "EarlyStop_" + model_name + ".pkl"))


        if ((100+e_i)%20)==0:
             lr = lr/2
             print(' Learning rate decreased to : {}'.format(lr))
             for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                        
    save_result(arr_loss,val_loss,'loss',model_name,'raw_10')
    save_result(np.mean(np.array(arr_dc),axis=1),np.mean(np.array(dscAll),axis=1),'acc',model_name,'raw_10')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./Data/MRBrainS/DataNii/', help='directory containing the train and val folders')
    parser.add_argument('--modality_dirs', nargs='+', default=['T1','T2_FLAIR','T1_IR'], help='subdirectories containing the multiple modalities')
    parser.add_argument('--save_dir', type=str, default='./Results_10000_100/', help='directory ot save results')
    parser.add_argument('--modelName', type=str, default='HyperDenseNet_3Mod', help='name of the model')
    parser.add_argument('--numModal', type=int, default=3, help='Number of image modalities')
    parser.add_argument('--numClasses', type=int, default=4, help='Number of classes (Including background)')
    parser.add_argument('--numSamplesEpoch', type=int, default=192, help='Number of samples per epoch')
    parser.add_argument('--numEpochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=2, help='Batch size')
    parser.add_argument('--l_rate', type=float, default=0.02, help='Learning rate')

    opts = parser.parse_args()
    print(opts)
    
    runTraining(opts)
