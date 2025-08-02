import math

import torch
import torch.nn as nn

from CryoAtom.utils.affine_utils import affine_from_3_points, affine_mul_vecs,affine_composition,quaternion_to_matrix,get_affine


class BackboneFrameNet(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.hfz = in_features
        self.backbone_fc = nn.Linear(self.hfz, 6)
        torch.nn.init.normal_(self.backbone_fc.weight, std=0.02)
        self.backbone_fc.bias.data = torch.Tensor([0, 0, 0, 0, 0, 0])
        self.f1 = nn.Parameter(torch.tensor(1.5,dtype=torch.float))
        self.eps = 1e-6
    def forward(self, x, affine):
        y = self.backbone_fc(x)
        y = torch.cat([torch.sqrt(torch.square(self.f1)+self.eps)*torch.ones(size=y.shape[:-1]+(1,),dtype=torch.float,device=y.device),y],dim=-1)
        quaternions = y[...,:4]
        translations = y[...,4:]
        rotations = quaternion_to_matrix(quaternions)
        new_affine = affine_composition(affine,get_affine(rotations,translations))

        return new_affine
