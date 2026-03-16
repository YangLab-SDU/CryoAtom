import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=1000):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (freq_inv ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq.to(tensor.device))[...,None]
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).flatten(-2)
        return emb_x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        groups=1,
        activation_class=nn.ELU,
        conv_class=nn.Conv3d,
        affine=False,
        **kwargs,
    ):
        super().__init__()
        self.activation_fn = activation_class()
        self.conv1 = conv_class(
            in_planes, planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm1 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv2 = conv_class(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.norm2 = nn.InstanceNorm3d(planes, affine=affine)
        self.conv3 = conv_class(
            planes, self.expansion * planes, kernel_size=1, bias=False, groups=groups
        )
        self.norm3 = nn.InstanceNorm3d(self.expansion * planes, affine=affine)

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


    def forward(self, x):
        out = self.activation_fn(self.norm1(self.conv1(x)))
        out = self.activation_fn(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut_conv(x)
        out = self.activation_fn(out)
        return out

class SpatialAvg(nn.Module):
    def forward(self, x):
        return x.mean(dim=[-3, -2, -1])
class FcResBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        activation_class=nn.ReLU,
        normalization_class=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
        )
        self.forward = (
            self.residual_forward
            if in_features == out_features
            else self.non_residual_forward
        )
        self.activation = nn.Sequential(
            activation_class(),
            normalization_class(out_features),
        )

    def residual_forward(self, x):
        y = self.net(x)
        return self.activation(x + y / np.sqrt(2))

    def non_residual_forward(self, x):
        return self.activation(self.net(x))

def ThreeD_Rope(q, k, pos_emb,edge_index=None):
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # * afz
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # * afz
    if edge_index is None:
        # q (* ahz afz)  k (* kz ahz afz)
        q_new = q * cos_pos[...,None,:] + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[...,None,:]
        k_new = k * cos_pos[...,None,:] + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[...,None,:]
    else:
        # q (* ahz afz)  k (* kz ahz afz)
        q_new = q * cos_pos[...,None,:] + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[...,None,:]
        k_new = k * cos_pos[edge_index][...,None,:] + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[edge_index][...,None,:]
    return q_new,k_new

