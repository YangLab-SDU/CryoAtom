from torch import nn
import torch
import einops
from einops.layers.torch import Rearrange
from CryoAtom2.CryoNet.backbone_distance_embedding import BackBoneDistanceEmbedding
from CryoAtom2.CryoNet.common_modules import Bottleneck, SpatialAvg, FcResBlock
from CryoAtom2.utils.affine_utils import get_affine_translation, get_affine_rot, sample_centered_cube_rot_matrix, \
    sample_centered_rectangle_along_vector
from CryoAtom2.utils.torch_utlis import get_batches_to_idx
class ShortConv(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,activation_class=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
        self.relu1 = activation_class()
    def forward(self,x:torch.Tensor):
        y = self.norm1(self.conv1(x))
        y = self.relu1(y)
        return y
class FastDownSample(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,activation_class=nn.ReLU):
        super().__init__()
        self.activate_fn = activation_class()
        self.conv0 = nn.Conv3d(in_channels,out_channels*3,kernel_size=1,bias=False)
        self.norm0 = nn.InstanceNorm3d(out_channels*3,affine=True)
        self.conv1 = nn.Conv3d(out_channels*3,out_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels,affine=True)
    def forward(self,x:torch.Tensor):
        y = self.activate_fn(self.norm0(self.conv0(x)))
        y = self.activate_fn(self.norm1(self.conv1(y)))
        return y
class MultiCryoVision(nn.Module):
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,activation_class=nn.ReLU):
        super().__init__()
        self.avg_pool = SpatialAvg()
        self.conv0 = ShortConv(in_channels,hidden_channels*4,activation_class)
        self.conv1 = FastDownSample(hidden_channels*4,hidden_channels,activation_class)
        self.vision1 = nn.Conv3d(hidden_channels,out_channels,kernel_size=9,bias=False)
        self.conv2 = FastDownSample(hidden_channels,hidden_channels*4,activation_class)
        self.vision2 = nn.Conv3d(hidden_channels*4,out_channels,kernel_size=5,bias=False)
        self.conv3 = FastDownSample(hidden_channels*4,hidden_channels*16,activation_class)
        self.vision3 = nn.Conv3d(hidden_channels*16,out_channels,kernel_size=3,bias=False)
        self.norm = nn.LayerNorm(out_channels)
    def forward(self,x:torch.Tensor):
        y = self.conv0(x)
        y = self.conv1(y)
        y1 = self.avg_pool(self.vision1(y))
        y = self.conv2(y)
        y2 = self.avg_pool(self.vision2(y))
        y = self.conv3(y)
        y3 = self.avg_pool(self.vision3(y))
        return self.norm(y1+y2+y3)
class CryoFeatures(nn.Module):
    def __init__(
            self,
            in_features:int,
            attention_features:int = 48,
            number_neighbour:int = 30,
            cube_size:int = 17,
            rectangle_length:int = 12,
            cryo_emb_dim:int = 256,
            activation_class:nn.Module = nn.ReLU,
            **kwargs,
    ):
        super().__init__()
        assert cryo_emb_dim % 4 == 0
        self.norm_residual = nn.LayerNorm(in_features)
        self.ifz = in_features
        self.kz = number_neighbour
        self.c_length = cube_size
        self.r_length = rectangle_length
        self.cryo_emb_dim = cryo_emb_dim
        self.activation_class = activation_class
        self.backbone_distance_emb = BackBoneDistanceEmbedding(
            num_neighbours=number_neighbour,
            position_encoding_dim=attention_features//3,
            distance_encoding_dim = self.ifz//2,
        )
        self.prot_emb_bias = nn.Embedding(2,self.ifz)
        self.conv_cube = MultiCryoVision(in_channels=1,hidden_channels=cryo_emb_dim//4,out_channels=self.ifz)
        self.conv_rectangle = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.cryo_emb_dim//4,
                kernel_size=3,
                bias=False,
            ),
            Rearrange(
                "(b kz) c z y x -> b kz (c z y x)",
                kz=self.kz,
                c=self.cryo_emb_dim//4,
                z=(self.r_length-2),
                x=1,
                y=1,
            ),
            nn.LayerNorm(self.cryo_emb_dim//4 * (self.r_length-2)),
            activation_class(),
            nn.Linear(
                self.cryo_emb_dim//4 * (self.r_length-2), self.ifz//2, bias=False
            )
        )
    def forward(
            self,
            affines,
            prot_mask,
            residual_repr:torch.Tensor,
            torsion_angles = None,
            cryo_grids=None,
            cryo_global_origins=None,
            cryo_voxel_sizes=None,
            edge_index=None,
            batch=None,
            **kwargs,
    ):
        assert cryo_grids is not None
        batch_to_idx = (
            get_batches_to_idx(batch)
            if batch is not None
            else [torch.arange(0, len(affines), dtype=int, device=affines.device)]
        )
        with torch.no_grad():
            positions = get_affine_translation(affines)
            batch_cryo_grids = [
                cg.expand(len(b), -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_points = [
                (positions[b].reshape(-1, 3) - go) / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]

            cryo_points_rot_matrices = [
                get_affine_rot(affines[b]).reshape(-1, 3, 3) for b in batch_to_idx
            ]

            cryo_points_cube = sample_centered_cube_rot_matrix(
                batch_cryo_grids,
                cryo_points_rot_matrices,
                cryo_points,
                cube_side=self.c_length,
            ) #N 1 c_len c_len c_len
        cryo_aa_logits = self.conv_cube(cryo_points_cube.requires_grad_()) # N ifz
        x_1 = cryo_aa_logits + self.prot_emb_bias(prot_mask.long()) + self.norm_residual(residual_repr)
        bde_out = self.backbone_distance_emb(affines,prot_mask,torsion_angles, edge_index, batch)
        with torch.no_grad():
            batch_cryo_grids = [
                cg.expand(len(b) * self.kz, -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_vectors = bde_out.neighbour_positions.detach()
            cryo_vectors = [einops.rearrange(cryo_vectors[b],"b kz c -> (b kz) c", c=3) for b in batch_to_idx]
            cryo_vectors_center_positions = [
                (
                        einops.rearrange(bde_out.positions[b]
                        .unsqueeze(1)
                        .expand(len(b), self.kz, 3)
                        ,"b kz c -> (b kz) c", c=3)
                        - go
                )
                / vz
                for (b, go, vz) in zip(
                    batch_to_idx, cryo_global_origins, cryo_voxel_sizes
                )
            ]
            cryo_vectors_rec = sample_centered_rectangle_along_vector(
                batch_cryo_grids,
                cryo_vectors,
                cryo_vectors_center_positions,
                rectangle_length=self.r_length,
            )  # (N kz) self.r_length 3 3
        x_2 = self.conv_rectangle(cryo_vectors_rec.requires_grad_()) + bde_out.neighbour_distances # N kz ifz
        return (x_1,x_2,bde_out.edge_index,bde_out.full_edge_index,cryo_aa_logits,bde_out.pos3d_emb)
