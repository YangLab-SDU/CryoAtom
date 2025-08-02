import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, channels, freq_inv=100):
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
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq.to(tensor.device))
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x

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
class SpatialAvg(nn.Module):
    def forward(self, x):
        return x.mean(dim=[-3, -2, -1])


def Rope(q, k, pos_emb,edge_index):
    # q (N ahz afz)  k (N kz ahz afz)
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1) # N afz
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1) # N afz
    q_new = q * cos_pos[:,None] + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape) * sin_pos[:,None]
    k_new = k * cos_pos[edge_index][:,:,None] + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape) * sin_pos[edge_index][:,:,None]
    return q_new,k_new

