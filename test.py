# import numpy as np
# from typing import List
# class Solution:
#     def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
#         idx=np.argsort(np.array(nums))
#         # nums_sort=list(np.zeros(len(nums)).astype('int'))
#         nums_sort=nums.copy()
#         nums_sort[idx[0]]=0
#         for i in range(1,len(nums)):
#             if nums[idx[i-1]]==nums[idx[i]]:
#                 nums_sort[idx[i]]=nums_sort[idx[i-1]]
#             else:
#                 nums_sort[idx[i]]=i
#         return nums_sort

# nums=[8,1,2,2,3]
# a= Solution()
# nums_sort=a.smallerNumbersThanCurrent(nums)
from PIL import Image
import matplotlib

matplotlib.use('TkAgg')

from matplotlib import pylab as plt
import nibabel as nib
from nibabel import nifti1
from nibabel.viewers import OrthoSlicer3D
import imageio 
import matplotlib as mpl
# from palettable.cartocolors.sequential import DarkMint_4
# 自定义colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#000000', '#4169E1', '#98FB98', '#FFFF00'], 256)

path_GT='HyperDenseNet_pytorch\\Data\MRBrainS\\DataNii\\Training\\GT\\1.nii'
path_train='HyperDenseNet_pytorch\\Data\\MRBrainS\\DataNii\\Training\\T1\\1.nii'
img_gt = nib.load(path_GT)
img_train = nib.load(path_train)
width, height, queue = img_gt.dataobj.shape

img1 = img_gt.dataobj[:, :, 24]
img2 = img_train.dataobj[:, :, 24]
plt.imsave('test.jpg',img1,cmap=colormap())
plt.imshow(img1)
plt.axis('off')
plt.show()

