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
import logging
from matplotlib import pyplot as plt

path_1=glob('./Data/MRBrainS/DataNii_png/*')

# CLAHE
nii_path_tmp = './Data/MRBrainS/DataNii_CLAHE_test'
if not os.path.exists(nii_path_tmp):
    os.mkdir(nii_path_tmp)


# contraster
icter = ImageContraster()

path =path_1[1]
# for path in path_1[1]:
path_2=os.listdir(path)
print(path_2)
for files in path_2:
    path_3=os.listdir(os.path.join(path,files))
    file_len=len(os.listdir(os.path.join(path,'GT')))
    if files !='GT':           
        # for file_nii in path_3:
        files_png = sorted(os.listdir(os.path.join(path,files)))         #下载niifile文件（其实是提取文件）
        file_png_len=int(len(files_png)/file_len)
        for j in range(file_len):
            empt_mat=[]
            for file_png in files_png[j*file_png_len:(j+1)*file_png_len]:
                if int(file_png.split('_')[0])!=j+1:
                    information_wromg_png = ("%s didn't merge the same images\n" %(files))
                    logging.warning(information_wromg_png)
                img1=Image.open(os.path.join(path,files,file_png))
                clahe_eq_img = icter.enhance_contrast(img1, method = "CLAHE", blocks = 8, threshold = 1.0)
                empt_mat.append((np.array(clahe_eq_img)[:,:]).T)
            emp=np.array(empt_mat)
            nii_file = sitk.GetImageFromArray(emp)
            # # 此处的emp的格式为样本数*高度*宽度*通道数不要颠倒这些维度的顺序，否则文件保存错误
            if not os.path.exists(nii_path_tmp+os.sep+path.split(os.sep)[-1]):
                    os.mkdir(nii_path_tmp+os.sep+path.split(os.sep)[-1])  
            if not os.path.exists(nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files):
                os.mkdir(nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files)
            nii_path=nii_path_tmp+os.sep+path.split(os.sep)[-1]+os.sep+files
            sitk.WriteImage(nii_file,os.path.join(nii_path,'{}.nii'.format(file_png.split('_')[0]))) # nii_path 为保存路径                    
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



# img = nib.load('Data/MRBrainS/DataNii_CLAHE/Training/T1/1.nii')          #下载niifile文件（其实是提取文件）
# img_fdata = img.get_fdata()
# plt.imshow(img_fdata[:,:,0])
# plt.show()
# imageio.imwrite('Data/MRBrainS/DataNii_CLAHE/Training/T1/1.png',img_fdata[:,:,0])


    
