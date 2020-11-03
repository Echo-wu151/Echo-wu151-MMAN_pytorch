#from Blocks import *
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
from matplotlib import pylab as plt
#from layers import *

def croppCenter(tensorToCrop,finalShape):

    org_shape = tensorToCrop.shape
    diff = org_shape[2] - finalShape[2]
    # print('diff',diff)
    croppBorders = int(diff/2)
    return tensorToCrop[:,
                        :,
                        croppBorders:org_shape[2]-croppBorders,
                        croppBorders:org_shape[3]-croppBorders,
                        croppBorders:org_shape[4]-croppBorders]

def convBlock(nin, nout, kernel_size=3, batchNorm = False, layer=nn.Conv2d, bias=True, dropout_rate = 0.0, padding=0,dilation = 1):
    
    if batchNorm == False:
        return nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, padding=padding,dilation=dilation)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(nin),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, padding=padding,dilation=dilation)
        )
def convBlock_3D(nin, nout, kernel_size=3, batchNorm = False, layer=nn.Conv3d, bias=True, dropout_rate = 0.0, padding=0,dilation = 1):
    
    if batchNorm == False:
        return nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias,padding=padding, dilation=dilation)
        )
    else:
        return nn.Sequential(
            nn.BatchNorm3d(nin),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            layer(nin, nout, kernel_size=kernel_size, bias=bias, padding=padding,dilation=dilation)
        )       


def convSoftmax_3D(nin, nout, kernel_size=1, stride=1, padding=0, bias=False, layer=nn.Conv3d, dilation = 1):
    return nn.Sequential(
        
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation),
        nn.Softmax()
    )

def convSoftmax(nin, nout, kernel_size=1, stride=1, padding=0, bias=False, layer=nn.Conv2d, dilation = 1):
    return nn.Sequential(
        nn.Softmax(),
        layer(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
    )

class DiB_Block_3D(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, batchNorm = False, dropout_rate = 0.0,):
        super(DiB_Block, self).__init__()
        rates=[1,2,4]
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                convBlock_3D(nin, nout, kernel_size=3, padding=rate, dilation=rate )
            )
    def forward(self, x):
        return torch.cat([stage(x) for stage in self.children()],dim=1)

class DiB_Block(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, batchNorm = False, dropout_rate = 0.0,):
        super(DiB_Block, self).__init__()
        rates=[1,2,4]
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                convBlock(nin, nout, kernel_size=3, padding=rate, dilation=rate )
            )
    def forward(self, x):
        return torch.cat([stage(x) for stage in self.children()],dim=1)


class DIB_3D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, dilation=1):
    #padding = (kernel_size - 1) // 2
        super(DIB, self).__init__()
        layer=[]
        layer.append(DiB_Block_3D(in_planes,out_planes,kernel_size=3))
        self.layer = nn.Sequential(*layer)
        self.conv=convBlock_3D(out_planes*3,out_planes,kernel_size=1)#*3
    def forward(self, x):
        x = self.layer(x)
        x = self.conv(x)
        return x


class DIB(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, dilation=1):
    #padding = (kernel_size - 1) // 2
        super(DIB, self).__init__()
        layer=[]
        layer.append(DiB_Block(in_planes,out_planes,kernel_size=3))
        self.layer = nn.Sequential(*layer)
        self.conv=convBlock(out_planes*3,out_planes,kernel_size=1)#*3
    def forward(self, x):
        x = self.layer(x)
        x = self.conv(x)
        return x


class MMAN(nn.Module):
    def __init__(self, nClasses):
        super(MMAN, self).__init__()
        
        # Path-Top
        self.DIB1_Top = DIB(1, 32)
        self.MaxPooling2_Top=nn.MaxPool2d(2)
        self.DIB3_Top = DIB(32, 32)#maxpooling是卷积核个数变少还是图片小大变小了呢
        self.MaxPooling4_Top=nn.MaxPool2d(2)
        self.DIB5_Top = DIB(32, 64)
        self.MaxPooling6_Top=nn.MaxPool2d(2)
        self.DIB7_Top = DIB(64, 64)

        
        # Path-Middle
        self.DIB1_Middle = DIB(1, 32)
        self.MaxPooling2_Middle=nn.MaxPool2d(2)
        self.DIB3_Middle = DIB(32, 32)
        self.MaxPooling4_Middle=nn.MaxPool2d(2)
        self.DIB5_Middle = DIB(32, 64)
        self.MaxPooling6_Middle=nn.MaxPool2d(2)
        self.DIB7_Middle = DIB(64, 64)

        
        # Path-Bottom
        self.DIB1_Bottom = DIB(1, 32)
        self.MaxPooling2_Bottom=nn.MaxPool2d(2)
        self.DIB3_Bottom = DIB(32, 32)
        self.MaxPooling4_Bottom=nn.MaxPool2d(2)
        self.DIB5_Bottom = DIB(32, 64)
        self.MaxPooling6_Bottom=nn.MaxPool2d(2)
        self.DIB7_Bottom = DIB(64, 64)

        #concatenate
        self.DIB1_cat = DIB(96, 16)
        self.DIB2_cat = DIB(96, 16)
        self.UpSampling3_cat=nn.Upsample(scale_factor=2)
        self.DIB4_cat = DIB(192, 16)
        self.UpSampling5_cat=nn.Upsample(scale_factor=4)
        self.DIB6_cat = DIB(192, 16)
        self.UpSampling7_cat=nn.Upsample(scale_factor=8)


        self.final = convSoftmax(64, nClasses, kernel_size=1)
        
    def forward(self, input):

        # ----- First layer ------ #
        # get the 3 channels as 5D tensors
        y1t = self.DIB1_Top(input[:,0:1,:,:])
        y1m = self.DIB1_Middle(input[:,1:2,:,:])
        y1b = self.DIB1_Bottom(input[:,2:3,:,:])
        
        # concatenate
        y1_i = torch.cat((y1t,y1m,y1b),dim=1)
        y1_o = self.DIB1_cat(y1_i)

        # ----- Second layer ------ #
        y2t_i = self.MaxPooling2_Top(y1t)
        y2m_i = self.MaxPooling2_Middle(y1m)
        y2b_i = self.MaxPooling2_Bottom(y1b)

        y2t = self.DIB3_Top(y2t_i)
        y2m = self.DIB3_Middle(y2m_i)
        y2b = self.DIB3_Bottom(y2b_i)

        # concatenate
        y2_i = torch.cat((y2t,y2m,y2b),dim=1)
        y2_i = self.DIB2_cat(y2_i)
        y2_o = self.UpSampling3_cat(y2_i)

         # ----- Third layer ------ #
        y3t_i = self.MaxPooling4_Top(y2t)
        y3m_i = self.MaxPooling4_Middle(y2m)
        y3b_i = self.MaxPooling4_Bottom(y2b)

        y3t = self.DIB5_Top(y3t_i)
        y3m = self.DIB5_Middle(y3m_i)
        y3b = self.DIB5_Bottom(y3b_i)

        # concatenate
        y3_i = torch.cat((y3t,y3m,y3b),dim=1)
        y3_i = self.DIB4_cat(y3_i)
        y3_o = self.UpSampling5_cat(y3_i)

        # ----- Fourth layer ------ #
        y4t_i = self.MaxPooling6_Top(y3t)
        y4m_i = self.MaxPooling6_Middle(y3m)
        y4b_i = self.MaxPooling6_Bottom(y3b)

        y4t = self.DIB7_Top(y4t_i)
        y4m = self.DIB7_Middle(y4m_i)
        y4b = self.DIB7_Bottom(y4b_i)

        # concatenate
        y4_i = torch.cat((y4t,y4m,y4b),dim=1)
        y4_i= self.DIB6_cat(y4_i)
        y4_o = self.UpSampling7_cat(y4_i)

        
        # concatenate
        y5_i=torch.cat((y1_o,y2_o,y3_o,y4_o),dim=1)
        y5_o=self.final(y5_i)
        return y5_o

class MMAN_3D(nn.Module):
    def __init__(self, nClasses):
        super(MMAN, self).__init__()
        
        # Path-Top
        self.DIB1_Top = DIB_3D(1, 32)
        self.MaxPooling2_Top=nn.MaxPool3d(2)
        self.DIB3_Top = DIB_3D(32, 32)#maxpooling是卷积核个数变少还是图片小大变小了呢
        self.MaxPooling4_Top=nn.MaxPool3d(2)
        self.DIB5_Top = DIB_3D(32, 64)
        self.MaxPooling6_Top=nn.MaxPool3d(2)
        self.DIB7_Top = DIB_3D(64, 64)

        
        # Path-Middle
        self.DIB1_Middle = DIB_3D(1, 32)
        self.MaxPooling2_Middle=nn.MaxPool3d(2)
        self.DIB3_Middle = DIB_3D(32, 32)
        self.MaxPooling4_Middle=nn.MaxPool3d(2)
        self.DIB5_Middle = DIB_3D(32, 64)
        self.MaxPooling6_Middle=nn.MaxPool3d(2)
        self.DIB7_Middle = DIB_3D(64, 64)

        
        # Path-Bottom
        self.DIB1_Bottom = DIB_3D(1, 32)
        self.MaxPooling2_Bottom=nn.MaxPool3d(2)
        self.DIB3_Bottom = DIB_3D(32, 32)
        self.MaxPooling4_Bottom=nn.MaxPool3d(2)
        self.DIB5_Bottom = DIB_3D(32, 64)
        self.MaxPooling6_Bottom=nn.MaxPool3d(2)
        self.DIB7_Bottom = DIB_3D(64, 64)

        #concatenate
        self.DIB1_cat = DIB_3D(96, 16)
        self.DIB2_cat = DIB_3D(96, 16)
        self.UpSampling3_cat=nn.Upsample(2)
        self.DIB4_cat = DIB_3D(192, 16)
        self.UpSampling5_cat=nn.Upsample(4)
        self.DIB6_cat = DIB_3D(192, 16)
        self.UpSampling7_cat=nn.Upsample(8)


        self.final = convSoftmax(64, 4, kernel_size=1)
        
    def forward(self, input):

        # ----- First layer ------ #
        # get the 3 channels as 5D tensors
        y1t = self.DIB1_Top(input[:,0:1,:,:,:])
        y1m = self.DIB1_Middle(input[:,1:2,:,:,:])
        y1b = self.DIB1_Bottom(input[:,2:3,:,:,:])
        
        # concatenate
        y1_i = torch.cat((y1t,y1m,y1b),dim=1)
        y1_o = self.DIB1_cat(y1_i)

        # ----- Second layer ------ #
        y2t_i = self.MaxPooling2_Top(y1t)
        y2m_i = self.MaxPooling2_Middle(y1m)
        y2b_i = self.MaxPooling2_Bottom(y1b)

        y2t = self.DIB3_Top(y2t_i)
        y2m = self.DIB3_Middle(y2m_i)
        y2b = self.DIB3_Bottom(y2b_i)

        # concatenate
        y2_i = torch.cat((y2t,y2m,y2b),dim=1)
        y2_i = self.DIB2_cat(y2_i)
        y2_o = self.UpSampling3_cat(y2_i)

         # ----- Third layer ------ #
        y3t_i = self.MaxPooling4_Top(y2t)
        y3m_i = self.MaxPooling4_Middle(y2m)
        y3b_i = self.MaxPooling4_Bottom(y2b)

        y3t = self.DIB5_Top(y3t_i)
        y3m = self.DIB5_Middle(y3m_i)
        y3b = self.DIB5_Bottom(y3b_i)

        # concatenate
        y3_i = torch.cat((y3t,y3m,y3b),dim=1)
        y3_i = self.DIB4_cat(y3_i)
        y3_o = self.UpSampling5_cat(y3_i)

        # ----- Fourth layer ------ #
        y4t_i = self.MaxPooling6_Top(y3t)
        y4m_i = self.MaxPooling6_Middle(y3m)
        y4b_i = self.MaxPooling6_Bottom(y3b)

        y4t = self.DIB7_Top(y4t_i)
        y4m = self.DIB7_Middle(y4m_i)
        y4b = self.DIB7_Bottom(y4b_i)

        # concatenate
        y4_i = torch.cat((y4t,y4m,y4b),dim=1)
        y4_i= self.DIB6_cat(y4_i)
        y4_o = self.UpSampling7_cat(y4_i)

        
        # concatenate
        y5_i=torch.cat((y1_o,y2_o,y3_o,y4_o),dim=1)
        y5_o=self.final(y5_i)
        return y5_o


if __name__ == "__main__":
    model = MMAN(4)
    model.eval()
    image = torch.randn(1, 3, 24, 24)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
    print(model(image))
    print(image)