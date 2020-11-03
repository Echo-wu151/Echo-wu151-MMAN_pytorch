import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
from matplotlib import pylab as plt
from torch import Tensor
from typing import Callable, Any, Optional, Tuple, List
#DIB结构
class DIB_Block(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,conv_block:Optional[Callable[...,nn.Module]]= None) -> None:
        super(DIB_Block,self).__init__()
        if conv_block is None:
            conv_block=BasicConv2d
        self.branch3x3dr1x1=conv_block(in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1)
        self.branch3x3dr2x2=conv_block(in_channels,out_channels,kernel_size=3,stride=1,padding=2,dilation=2)
        self.branch3x3dr4x4=conv_block(in_channels,out_channels,kernel_size=3,stride=1,padding=4,dilation=4)
        self.conv=conv_block(out_channels*3,out_channels,kernel_size=1,stride=1,padding=0,dilation=1)
    def _forward(self,x:Tensor) -> Tensor:
        branch3x3dr1x1=self.branch3x3dr1x1(x)
        branch3x3dr2x2=self.branch3x3dr2x2(x)
        branch3x3dr4x4=self.branch3x3dr4x4(x)
        outputs=[branch3x3dr1x1,branch3x3dr2x2,branch3x3dr4x4]
        return torch.cat(outputs, 1)

    def forward(self,x:Tensor) -> Tensor:
        outputs=self._forward(x)
        outputs=self.conv(outputs)
        return outputs

##基本卷积
class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        # if BatchNorm2d==True:
        x = self.bn(x)
        return F.relu(x, inplace=True)

#带softmax的卷积
class SoftmaxConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs: Any,
    ) -> None:
        super(SoftmaxConv2d, self).__init__()
        self.softmax=nn.Softmax()
        self.conv = nn.Conv2d(in_channels, 4, bias=False, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x=self.softmax(x)
        outputs = self.conv(x)
        # if BatchNorm2d==True:
        # x = self.bn(x)
        return outputs

#模型
class MMAN(nn.Module):
    def __init__(self,num_classes:int=4,inception_blocks:Optional[List[Callable[...,nn.Module]]]=None)->None:
        super(MMAN,self).__init__()
        if inception_blocks is None:
            inception_blocks=[BasicConv2d,DIB_Block,SoftmaxConv2d]
        assert len(inception_blocks)== 3
        conv=inception_blocks[0]#此处的名称不能跟上面列表同名，否则就会报变量未引用就先
        DIB=inception_blocks[1]
        Conv2d_output=inception_blocks[2]

        #(N,1,240,240)->(N,32,240,240)
        self.DIB_Top1=DIB(1,32)
        self.DIB_Middle1=DIB(1,32)
        self.DIB_Bottom1=DIB(1,32)

        #(N,32,240,240)->(N,32,120,120)
        self.maxpool_Top1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool_Middle1=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool_Bottom1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        #(N,32,120,120)->(N,32,120,120)
        self.DIB_Top2=DIB(32,32)
        self.DIB_Middle2=DIB(32,32)
        self.DIB_Bottom2=DIB(32,32)

        #(N,32,120,120)->(N,32,60,60)
        self.maxpool_Top2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool_Middle2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool_Bottom2=nn.MaxPool2d(kernel_size=2,stride=2)

        #(N,32,60,60)->(N,64,60,60)
        self.DIB_Top3=DIB(32,64)
        self.DIB_Middle3=DIB(32,64)
        self.DIB_Bottom3=DIB(32,64)

        #(N,64,60,60)->(N,64,30,30)
        self.maxpool_Top3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool_Middle3=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool_Bottom3=nn.MaxPool2d(kernel_size=2,stride=2)

        #(N,64,30,30)->(N,64,30,30)
        self.DIB_Top4=DIB(64,64)
        self.DIB_Middle4=DIB(64,64)
        self.DIB_Bottom4=DIB(64,64)

        
        self.DIB_cat1=DIB(96,16)
        self.DIB_cat2=DIB(96,16)
        self.DIB_cat3=DIB(192,16)
        self.DIB_cat4=DIB(192,16)

        self.upsampling_cat1=nn.Upsample(scale_factor=2)
        self.upsampling_cat2=nn.Upsample(scale_factor=4)
        self.upsampling_cat3=nn.Upsample(scale_factor=8)

        self.final=Conv2d_output(64,num_classes,kernel_size=1,stride=1,padding=0)
    
    def _forward(self,input) -> Tensor:
        y1t = self.DIB_Top1(input[:,0:1,:,:])
        y1m = self.DIB_Middle1(input[:,1:2,:,:])
        y1b = self.DIB_Bottom1(input[:,2:3,:,:])
        
        # concatenate
        y1_i = torch.cat((y1t,y1m,y1b),dim=1)
        y1_o = self.DIB_cat1(y1_i)

        # ----- Second layer ------ #
        y2t_i = self.maxpool_Top1(y1t)
        y2m_i = self.maxpool_Middle1(y1m)
        y2b_i = self.maxpool_Bottom1(y1b)

        y2t = self.DIB_Top2(y2t_i)
        y2m = self.DIB_Middle2(y2m_i)
        y2b = self.DIB_Bottom2(y2b_i)

        # concatenate
        y2_i = torch.cat((y2t,y2m,y2b),dim=1)
        y2_i = self.DIB_cat2(y2_i)
        y2_o = self.upsampling_cat1(y2_i)

         # ----- Third layer ------ #
        y3t_i = self.maxpool_Top2(y2t)
        y3m_i = self.maxpool_Middle2(y2m)
        y3b_i = self.maxpool_Bottom2(y2b)

        y3t = self.DIB_Top3(y3t_i)
        y3m = self.DIB_Middle3(y3m_i)
        y3b = self.DIB_Bottom3(y3b_i)

        # concatenate
        y3_i = torch.cat((y3t,y3m,y3b),dim=1)
        y3_i = self.DIB_cat3(y3_i)
        y3_o = self.upsampling_cat2(y3_i)

        # ----- Fourth layer ------ #
        y4t_i = self.maxpool_Top3(y3t)
        y4m_i = self.maxpool_Middle3(y3m)
        y4b_i = self.maxpool_Bottom3(y3b)

        y4t = self.DIB_Top4(y4t_i)
        y4m = self.DIB_Middle4(y4m_i)
        y4b = self.DIB_Bottom4(y4b_i)

        # concatenate
        y4_i = torch.cat((y4t,y4m,y4b),dim=1)
        y4_i= self.DIB_cat4(y4_i)
        y4_o = self.upsampling_cat3(y4_i)

        
        # concatenate
        y5_i=torch.cat((y1_o,y2_o,y3_o,y4_o),dim=1)
        y5_o=self.final(y5_i)
        return y5_o

    def forward(self,x:Tensor) -> Tensor:
        x=self._forward(x)
        return x

if __name__ == "__main__":
    model = MMAN(4)
    model.eval()
    image = torch.randn(1, 3, 24, 24)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
    print(model(image))
    print(image)