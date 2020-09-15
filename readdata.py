# encoding=utf8
'''
查看和显示nii文件
'''

import matplotlib

matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D

example_filenameG = 'images\LabelsForTesting.nii'
example_filenameT1 = 'images\T1.nii'
example_filenameT2 = 'images\T1_IR.nii'
example_filenameFl = 'images\T2_FLAIR.nii'
img1 = nib.load(example_filenameG)
img2 = nib.load(example_filenameT1)
img3 = nib.load(example_filenameT2)
img4 = nib.load(example_filenameFl)
# volume=img.get_data()
# example_filename1 = 'GT_5_Epoch_10.nii.gz'
# example_filename2 = 'Pred_5_Epoch_10.nii.gz'
# img1 = nib.load(example_filename1)
# img2 = nib.load(example_filename2)
# volume1=img1.get_data()
# volume2=img2.get_data()
# print(img)
# print(img.header['db_name'])  # 输出头信息
# print(img.dataobj.shape)
# width, height, queue = img.dataobj.shape
# #
# OrthoSlicer3D(img.dataobj).show()
# plt.figure(1)
# width1, height1, queue1 = img1.dataobj.shape
# #
# OrthoSlicer3D(img1.dataobj).show()
# # # 计算看需要多少个位置来放切片图
# x1 = int((queue1/1) ** 0.5) + 1
# num1 = 1
# for i in range(0, queue1, 1):
#     img_arr = img1.dataobj[:, :, i]
#     plt.subplot(x1, x1, num1)
#     plt.imshow(img_arr, cmap='gray')
#     num1 += 1
#
# plt.show()
# plt.figure(2)
width2, height2, queue2 = img2.dataobj.shape
#
OrthoSlicer3D(img2.dataobj).show()
# # 计算看需要多少个位置来放切片图
x2 = int((queue2/1) ** 0.5) + 1
num2 = 1
for i in range(0, queue2, 1):
    img_arr = img2.dataobj[:, :, i]
    plt.subplot(x2, x2, num2)
    plt.imshow(img_arr, cmap='gray')
    num2 += 1

plt.show()
# from os.path import isfile, join
# import os
# moda_1=r'data\training\T1'
# if os.path.exists(moda_1):
#     imageNames_train = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f))]
#     imageNames_train.sort()
#     print(' ------- Images found ------')
#     for i in range(len(imageNames_train)):
#         print(' - {}'.format(imageNames_train[i]))
# else:
#     raise Exception(' - {} does not exist'.format(moda_1))

# for i in range(l):
#             for j in range(n):
#                 for k in range(m):
#                     if a[k,j,i]==2:
#                         a[k, j, i] =1
#                     elif a[k,j,i]==4:
#                         a[k, j, i] = 3
#                     elif a[k, j, i] ==6:
#                         a[k, j, i] = 5


# patch_shape=(27,27,27)
# extraction_step=(5,5,5)
# patchesList=[]
# for x_i in range(0, volume.shape[0] - patch_shape[0], extraction_step[0]):
#     for y_i in range(0, volume.shape[1] - patch_shape[1], extraction_step[1]):
#         for z_i in range(0, volume.shape[2] - patch_shape[2], extraction_step[2]):
#             # print('{}:{} to {}:{} to {}:{}'.format(x_i,x_i+patch_shape[0],y_i,y_i+patch_shape[1],z_i,z_i+patch_shape[2]))
#
#             patchesList.append(volume[x_i:x_i + patch_shape[0],
#                                y_i:y_i + patch_shape[1],
#                                z_i:z_i + patch_shape[2]])