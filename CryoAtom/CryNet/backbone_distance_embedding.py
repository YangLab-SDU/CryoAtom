from collections import namedtuple

import numpy as np
import torch

from CryoAtom.CryNet.common_modules import SinusoidalPositionalEncoding
from torch import nn

from CryoAtom.utils.torch_utlis import get_batches_to_idx
from CryoAtom.utils.affine_utils import get_affine_translation,vecs_to_local_affine
from CryoAtom.utils.knn_graph import knn_graph
from CryoAtom.utils.protein import frames_and_literature_positions_to_atom3_pos
BackboneDistanceEmbeddingOutput = namedtuple(
    "BackboneDistanceEmbeddingOutput",
    [
        "pos3d_emb",
        "positions",
        "neighbour_positions",
        "neighbour_distances",
        "edge_index",
        "full_edge_index",
    ],
)
class BackBoneDistanceEmbedding(nn.Module):
    def __init__(self,
                 num_neighbours: int =20,
                 position_encoding_dim: int =16,
                 ) -> None:
        super().__init__()
        self.num_n = num_neighbours
        self.ped = position_encoding_dim
        self.distance_encoding = SinusoidalPositionalEncoding(self.ped)
    def forward(self,affines,edge_index = None,batch = None)->BackboneDistanceEmbeddingOutput:
        positions = get_affine_translation(affines)
        if edge_index is None:
            edge_index = knn_graph(positions,self.num_n,batch=batch,loop=False,flow="source_to_target")
            full_edge_index = edge_index
            edge_index = edge_index[0].reshape(len(positions),self.num_n) #N num_n
        neighbour_positions = vecs_to_local_affine(affines,positions[edge_index])# N num_n 3
        neighbour_distances = self.distance_encoding(neighbour_positions.norm(dim=-1))# N num_n ped
        position3d_embeddings = self.distance_encoding(positions).flatten(1) # N 3*ped
        return BackboneDistanceEmbeddingOutput(
            pos3d_emb=position3d_embeddings,
            positions=positions,
            neighbour_positions=neighbour_positions,
            neighbour_distances=neighbour_distances,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
        )
