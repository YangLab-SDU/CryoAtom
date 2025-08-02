from torch import nn
import torch
from einops.layers.torch import Rearrange
from CryoAtom.CryNet.backbone_distance_embedding import BackBoneDistanceEmbedding
from CryoAtom.CryNet.common_modules import Bottleneck, SpatialAvg
from CryoAtom.utils.affine_utils import get_affine_translation, get_affine_rot, sample_centered_cube_rot_matrix, \
    sample_centered_rectangle_along_vector
from CryoAtom.utils.torch_utlis import get_batches_to_idx
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
class CryoFeatures(nn.Module):
    def __init__(
            self,
            in_features:int,
            attention_features:int = 48,
            number_neighbour:int = 20,
            cube_size:int = 17,
            rectangle_length:int = 12,
            cryo_emb_dim:int = 256,
            activation_class:nn.Module = nn.ReLU,
            **kwargs,
    ):
        super().__init__()
        assert cryo_emb_dim % 4 == 0
        self.ifz = in_features
        self.kz = number_neighbour
        self.c_length = cube_size
        self.r_length = rectangle_length
        self.cryo_emb_dim = cryo_emb_dim
        self.activation_class = activation_class
        self.backbone_distance_emb = BackBoneDistanceEmbedding(
            num_neighbours=number_neighbour,
            position_encoding_dim=attention_features//3,
        )
        self.conv_cube = nn.Sequential(ShortConv(1,self.cryo_emb_dim),
            Bottleneck(self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True),
            Bottleneck(
                self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True
            ),
            Bottleneck(
                self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True
            ),
            Bottleneck(self.cryo_emb_dim, self.cryo_emb_dim // 4, stride=2, affine=True),
            nn.Conv3d(self.cryo_emb_dim,self.cryo_emb_dim,kernel_size=2),
            SpatialAvg(),
            nn.Linear(self.cryo_emb_dim, self.ifz, bias=False),
        )
        self.conv_rectangle = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=self.cryo_emb_dim//2,
                kernel_size=(1, 3, 3),
                bias=False,
            ),
            Rearrange(
                "(b kz) c z y x -> b kz (c z y x)",
                kz=self.kz,
                c=self.cryo_emb_dim//2,
                z=self.r_length,
                x=1,
                y=1,
            ),
            nn.LayerNorm(self.cryo_emb_dim//2 * self.r_length),
            activation_class(),
            nn.Linear(
                self.cryo_emb_dim//2 * self.r_length, self.ifz, bias=False
            )
        )
    def forward(
            self,
            affines,
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
        x_1 = cryo_aa_logits
        bde_out = self.backbone_distance_emb(affines, edge_index, batch)
        with torch.no_grad():
            batch_cryo_grids = [
                cg.expand(len(b) * self.kz, -1, -1, -1, -1)
                for (cg, b) in zip(cryo_grids, batch_to_idx)
            ]
            cryo_vectors = bde_out.neighbour_positions.detach()
            cryo_vectors = [cryo_vectors[b].reshape(-1, 3) for b in batch_to_idx]
            cryo_vectors_center_positions = [
                (
                        bde_out.positions[b]
                        .unsqueeze(1)
                        .expand(len(b), self.kz, 3)
                        .reshape(-1, 3)
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
        x_2 = self.conv_rectangle(cryo_vectors_rec.requires_grad_()) # N kz ifz
        neighbour_distances_emb = torch.cat([x_2,bde_out.neighbour_positions,bde_out.neighbour_distances],dim=-1)
        return (x_1,x_2,bde_out.edge_index,bde_out.full_edge_index,cryo_aa_logits,neighbour_distances_emb,bde_out.pos3d_emb)
