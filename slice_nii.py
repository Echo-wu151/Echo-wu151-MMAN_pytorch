#-*- coding : utf-8 -*-

from PIL import Image, ImageOps
import numpy as np

from contrast import ImageContraster
import imageio 
from pathlib import Path
import os
from glob import glob
import SimpleITK as sitk
import nibabel as nib
import shutil

def read_niifile(niifile):           #读取niifile文件
    img = nib.load(niifile)          #下载niifile文件（其实是提取文件）
    img_fdata = img.get_fdata()      #获取niifile数据
    return img_fdata



# CLAHE
nii_path_tmp = './Data/MRBrainS/DataNii_CLAHE'
if not os.path.exists(nii_path_tmp):
    os.mkdir(nii_path_tmp)

path_1=glob('./Data/MRBrainS/DataNii/*')

# CLAHE
nii_path_tmp = './Data/MRBrainS/DataNii_png'
if not os.path.exists(nii_path_tmp):
    os.mkdir(nii_path_tmp)


# contraster
icter = ImageContraster()
empt_mat=[]
for path in path_1:
    path_2=os.listdir(path)
    print(path_2)
    for files in path_2:
        path_3=os.listdir(os.path.join(path,files))
        if files !='GT':
            
            for file_nii in path_3:
                print(1)
                img = nib.load(os.path.join(path,files,file_nii))          #下载niifile文件（其实是提取文件）
                img_fdata = img.get_fdata()

                (x,y,z) = img_fdata.shape 
                if not os.path.exists(nii_path_tmp+os.sep+path.split(os.sep)[-1]):
                    os.mkdir(nii_path_tmp+os.sep+path.split(os.sep)[-1])  
                if not os.path.exists(nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files):
                    os.mkdir(nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files)
                nii_path=nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files
                for i in range(z):
                    imageio.imwrite(os.path.join(nii_path,'{}_{}.png'.format(file_nii.split('.')[0],i)),img_fdata[:,:,i])                               
        else:
            for file_nii  in path_3:
                from_path=os.path.join(path,files,file_nii)
                if not os.path.exists(nii_path_tmp+os.sep+path.split(os.sep)[-1]):
                    os.mkdir(nii_path_tmp+os.sep+path.split(os.sep)[-1])  
                if not os.path.exists(nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files):
                    os.mkdir(nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files)
                to_path=nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files
                # print(to_path)
                # if not os.path.exists(to_path):
                #     os.mkdir(to_path)
                shutil.copy(from_path,os.path.join(to_path,file_nii))





    
