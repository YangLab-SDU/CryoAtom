import torch
import einops
from torch import nn
import numpy as np
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops.layers.torch import Rearrange
import contextlib

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
        sin_inp_x = torch.einsum("...i,j->...ij", tensor, self.inv_freq.to(tensor.device))[...,None]
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).flatten(-2)
        return emb_x
def get_lattice_meshgrid_np(shape_size, no_shift=False):
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh
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
        checkpoint=False,
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

        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(self, x):
        out = self.activation_fn(self.norm1(self.conv1(x)))
        out = self.activation_fn(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
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
class ShortConvAdd(nn.Module):
    def __init__(self,in_channels:int):
        super().__init__()
        self.conv1 = nn.Conv3d(1,in_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.InstanceNorm3d(in_channels,affine=False)
        self.norm2 = nn.InstanceNorm3d(in_channels,affine=True)
        self.relu1 = nn.ELU()
    def forward(self,x0,x1):
        y = self.norm1(self.conv1(x0))
        y = self.relu1(y+self.norm2(x1))
        return y
class Res2NetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, scale=4,activate_class:nn.Module=nn.ReLU):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels,out_channels*self.scale,1,1,0,bias=False),nn.InstanceNorm3d(out_channels*self.scale,affine=True))
        self.norm1 = nn.InstanceNorm3d(out_channels*self.scale,affine=True)
        self.conv_list = nn.ModuleList([nn.Conv3d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False) for _ in range(self.scale - 1)])
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
class Transition(nn.Module):
    def __init__(self,in_features:int,norm:nn.Module,n:int=3):
        super().__init__()
        self.norm = norm(in_features)
        self.w1 = nn.Linear(in_features,in_features*n,bias=False)
        self.w2 = nn.Linear(in_features,in_features*n,bias=False)
        self.w3 = nn.Linear(in_features*n,in_features,bias=False)
        self.short = nn.Identity()
    def forward(self,x):
        y = self.w3(nn.functional.silu(self.w1(x))*self.w2(x))
        y = self.norm(y + self.short(x))
        return y
class AttentionWith3DRoPE(nn.Module):
    def __init__(
            self,
            in_features:int,
            attention_heads:int,
            attention_features:int
    ):
        super(AttentionWith3DRoPE, self).__init__()
        self.ifz = in_features
        self.ahz = attention_heads
        self.afz = attention_features
        self.attention_scale = np.sqrt(self.afz)
        self.q = nn.Sequential(
            nn.Linear(self.ifz,self.ahz * self.afz),
            Rearrange("B L (ahz afz) -> B L ahz afz",ahz=self.ahz,afz=self.afz)
        )
        self.k = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz,bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.v = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz,bias=False),
            Rearrange("B L (ahz afz) -> B L ahz afz", ahz=self.ahz, afz=self.afz)
        )
        self.back = nn.Sequential(
            Rearrange("B L ahz afz -> B L (ahz afz)",ahz=self.ahz,afz=self.afz),
            nn.Linear(self.ahz*self.afz,self.ifz,bias=False)
        )
        self.pos_encoding = SinusoidalPositionalEncoding(channels=attention_features//3)
        self.norm1 = nn.LayerNorm(in_features)
        self.transition1 = Transition(in_features,nn.LayerNorm)
    def forward(self, x):
        B,C,H,D,W = x.shape
        pos_emb = torch.from_numpy(get_lattice_meshgrid_np((H,D,W),no_shift=True)).float().to(x.device)[None].repeat(B,1,1,1,1)
        pos_emb = einops.rearrange(pos_emb,"B H D W C -> B (H D W) C")*1.5
        pos_emb = self.pos_encoding(pos_emb).flatten(-2)
        x_vec = einops.rearrange(x,"B C H D W -> B (H D W) C")
        query = self.q(x_vec)
        key = self.k(x_vec)
        value = self.v(x_vec)
        query,key = ThreeD_Rope(query,key,pos_emb)
        attention_weights = (torch.einsum('blai,bkai->blka', query, key) / self.attention_scale).softmax(dim=-2)
        out = torch.einsum('blka,bkai->blai', attention_weights, value)
        out = self.norm1(x_vec + self.back(out))
        out = self.transition1(out)
        return einops.rearrange(out,"B (H D W) C -> B C H D W",B=B,C=C,H=H,D=D,W=W)


class RUNet(nn.Module):
    def __init__(self):
        super(RUNet,self).__init__()
        self.shortconvadd = ShortConvAdd(64)
        self.shortconv1 = ShortConv(64,256)
        self.downsample1 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample2 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample3 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.downsample4 = Bottleneck(256, 256 // 4, stride=2, affine=True)
        self.main0 = nn.Sequential(*[AttentionWith3DRoPE(256,8,48) for _ in range(4)])
        # self.attn1 = AttentionGate(256,256,256)
        self.main1 = self.main_layer(256, 3, 4)
        self.attn2 = AttentionGate(256,256,128)
        self.main2 = self.main_layer(128, 4, 4)
        self.attn3 = AttentionGate(256,128,64)
        self.main3 = self.main_layer(64, 4, 4)
        self.attn4 = AttentionGate(256,64,64)
        self.main4 = self.main_layer(64, 4, 4)
        self.conv14 = nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv3d(64, 32, kernel_size=5, stride=1, padding=2)
        self.conv12 = nn.Conv3d(64, 32, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv13 = nn.Conv3d(in_channels=32 * 3, out_channels=4, padding=1, kernel_size=3)
    def main_layer(self,input_channels,expansion,num_layers):
        layer=[]
        for i in range(num_layers):
            layer.append(Res2NetBlock(input_channels,input_channels,scale=expansion))
        return nn.Sequential(*layer)
    def upsample_add(self,f, g):
        g_shape = g.shape
        D, H, W = g_shape[2:]
        upsampled = nn.functional.interpolate(input=f, size=(D, H, W), mode='trilinear', align_corners=True)
        return (g + upsampled)
    #multi_scale_conv
    def forward(self,V0,run_iters:int=1):
        V_recycle = torch.zeros_like(V0)
        V_recycle = V_recycle.repeat(1,64,1,1,1)
        for run_iter in range(run_iters):
            notlast_flag = (run_iter != (run_iters - 1))
            with torch.no_grad() if notlast_flag else contextlib.nullcontext():
                V = self.shortconvadd(V0,V_recycle)
                ds_0 = self.shortconv1(V)
                ds_1 = self.downsample1(ds_0)
                ds_2 = self.downsample2(ds_1)
                ds_3 = self.downsample3(ds_2)
                ds_4 = self.downsample4(ds_3)
                c4 = self.main0(ds_4)
                c3 = self.main1(self.upsample_add(c4, ds_3))
                c2 = self.main2(self.attn2(c3,ds_2))
                c1 = self.main3(self.attn3(c2,ds_1))
                c0 = self.main4(self.attn4(c1,ds_0))
                f3 = self.conv14(c0)
                f5 = self.conv11(c0)
                f7 = self.conv12(c0)
                f=torch.cat((f3,f5,f7),dim=1)
                f=self.relu1(f)
                f=self.conv13(f)
                V_recycle = c0.detach()
        return f

