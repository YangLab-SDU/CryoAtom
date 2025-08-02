import torch
import einops
from torch import nn
import random
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ReLU,
        conv_class=nn.Conv3d,
        affine=False,
        checkpoint=True,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.bn1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.bn3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut_conv = nn.Conv3d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
                groups=groups,
            )

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x):
        out = self.activation_fn(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

    def forward_checkpoint(self, x):
        return torch_checkpoint(self.forward_normal, x, preserve_rng_state=False)
class ConvBuildingBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,activate_class:nn.Module=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm3d(out_channels,affine=True),
            self.activate_function,
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True
            )
    def forward(self,x:torch.Tensor):
        return self.activate_function(self.conv1(x)+self.shortcut_conv(x))

class ShortConv(nn.Module):
    def __init__(self,in_channels:int,out_channels:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
        self.relu1 = nn.ReLU()
    def forward(self,x:torch.Tensor):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y
class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=4,activate_class:nn.Module=nn.ReLU):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels,out_channels*self.scale,1,1,0,bias=False),nn.InstanceNorm3d(out_channels*self.scale,affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels*self.scale,affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1) for _ in range(self.scale - 1)])
        self.activate_class = activate_class()
        self.conv2 = nn.Sequential(nn.Conv3d(out_channels*self.scale,out_channels,1,1,0,bias=False),nn.InstanceNorm3d(out_channels,affine=True))
        self.shortcut_conv = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut_conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=True
            )
    def forward(self, x):
        x_list = self.activate_class(self.conv1(x)).chunk(self.scale,dim=1)

        y_list = []
        for ii,xi in enumerate(x_list):
            if ii == 0:
                y_list.append(xi)
            elif ii == 1:
                y_list.append(self.conv_list[ii-1](xi))
            else:
                y_list.append(self.conv_list[ii-1](xi+y_list[-1]))
        y = self.conv2(self.activate_class(self.norm1(torch.cat(y_list,dim=1))))
        y = self.activate_class(y+self.shortcut_conv(x))
        return y
class AttentionGate(nn.Module):
    def __init__(self,down_features:int,up_features:int,out_features:int,attention_features:int=64,attention_heads:int=8):
        super(AttentionGate, self).__init__()
        self.dfz = down_features
        self.ufz = up_features
        self.ofz = out_features
        self.afz = attention_features
        self.ahz = attention_heads
        self.conv_q = nn.Sequential(nn.Conv3d(
            in_channels=self.ufz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        self.conv_k = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.afz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.afz,affine=True))
        self.conv_v = nn.Sequential(nn.Conv3d(
            in_channels=self.dfz,
            out_channels=self.ufz,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        ),nn.InstanceNorm3d(self.ufz,affine=True))
        self.gate = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(
                in_channels=self.afz,
                out_channels=self.ahz,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.conv_back = ConvBuildingBlock(self.ufz,self.ofz)
    def forward(self,us,ds):
        ds_shape = ds.shape
        D, H, W = ds_shape[2:]
        upsampled = nn.functional.interpolate(input=us, size=(D, H, W), mode='trilinear', align_corners=True)
        query = self.conv_q(upsampled)
        key = self.conv_k(ds)
        value = self.conv_v(ds)
        value = einops.rearrange(value,"N (afz ahz) d h w -> N afz ahz d h w", ahz=self.ahz)
        gate = self.gate(query+key) #N ahz d h w
        out = value*gate[:,None]
        out = einops.rearrange(out,"N afz ahz d h w -> N (afz ahz) d h w", ahz=self.ahz)
        return self.conv_back(self.relu(out+upsampled))

class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet,self).__init__()
        self.shortconv0 = ShortConv(1,256)
        self.downsample1 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample2 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample3 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample4 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.main0 = self.main_layer(256, 2, 4)
        self.attn1 = AttentionGate(256,256,256)
        self.main1 = self.main_layer(256, 3, 4)
        self.attn2 = AttentionGate(256,256,128)
        self.main2 = self.main_layer(128, 4, 4)
        self.attn3 = AttentionGate(256,128,64)
        self.main3 = self.main_layer(64, 4, 4)
        self.attn4 = AttentionGate(256,64,64)
        self.main4 = self.main_layer(64, 4, 4)
        self.conv_addition = nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv3d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv3d(64, 32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv13 = nn.Conv3d(in_channels=32 * 3, out_channels=1, padding=1, kernel_size=3)
    def main_layer(self,input_channels,expansion,num_layers):
        layer=[]
        for i in range(num_layers):
            layer.append(Res2NetBlock(input_channels,input_channels,scale=expansion))
        return nn.Sequential(*layer)
    #multi_scale_conv
    def forward(self,V):
        ds_0 = self.shortconv0(V)
        ds_1 = self.downsample1(ds_0)
        ds_2 = self.downsample2(ds_1)
        ds_3 = self.downsample3(ds_2)
        ds_4 = self.downsample4(ds_3)
        c4 = self.main0(ds_4)
        c3 = self.main1(self.attn1(c4, ds_3))
        c2 = self.main2(self.attn2(c3,ds_2))
        c1 = self.main3(self.attn3(c2,ds_1))
        c0 = self.main4(self.attn4(c1,ds_0))
        f3 = self.conv14(c0)
        f5 = self.conv11(c0)
        f7 = self.conv12(c0)
        f=torch.cat((f3,f5,f7),dim=1)
        f=self.relu1(f)
        f=self.conv13(f)
        return f
