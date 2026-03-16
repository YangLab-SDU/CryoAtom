from collections import namedtuple

import torch

from CryoAtom2.CryoNet.common_modules import SinusoidalPositionalEncoding
from torch import nn
from CryoAtom2.utils.residue_constants import restype3_to_atoms, restype_3_to_index, select_torsion_angles
from CryoAtom2.utils.affine_utils import get_affine_translation
from CryoAtom2.utils.knn_graph import knn_graph
from CryoAtom2.utils.PNAComplex import frames_and_literature_positions_to_atomc_pos,torsion_angles_to_frames
def AtomPredictor(backbone_affines,torsion_angles,aatypes):
    all_frames = torsion_angles_to_frames(aatypes,backbone_affines,torsion_angles)
    all_atoms = frames_and_literature_positions_to_atomc_pos(aatypes, all_frames)
    return all_atoms
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
                 num_neighbours: int =30,
                 position_encoding_dim: int =16,
                 distance_encoding_dim: int = 128,
                 ) -> None:
        super().__init__()
        self.num_n = num_neighbours
        self.num_rbf = 20
        self.ped = position_encoding_dim
        self.distance_encoding = SinusoidalPositionalEncoding(self.ped)
        self.distance_encoding2 = nn.Linear(15*15*self.num_rbf,distance_encoding_dim,bias=False)
        self.distance_encoding_dim = distance_encoding_dim
    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 32., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def forward(self,affines,prot_mask,torsion_angles=None,edge_index = None,batch = None)->BackboneDistanceEmbeddingOutput:
        positions = get_affine_translation(affines)
        if edge_index is None:
            edge_index = knn_graph(positions,self.num_n,batch=batch,loop=False,flow="source_to_target")
            full_edge_index = edge_index
            edge_index = edge_index[0].reshape(len(positions),self.num_n) #N num_n
        # neighbour_positions = vecs_to_local_affine(affines,positions[edge_index])# N num_n 3
        neighbour_positions = positions[edge_index] - positions[:,None]# N num_n 3
        # neighbour_distances = self.distance_encoding(neighbour_positions.norm(dim=-1))# N num_n ped
        position3d_embeddings = self.distance_encoding(positions).flatten(1) # N 3*ped
        if torsion_angles is None:
            neighbour_distances = torch.zeros(
                positions.shape[0],self.num_n,15*15*self.num_rbf,device=positions.device,dtype=positions.dtype
            ).requires_grad_()
            neighbour_distances = self.distance_encoding2(neighbour_distances)

        else:
            with torch.no_grad():
                neighbour_atoms = {"ALA":["N", "CA", "C", "O", "CB"],"C":["P","O3'","O2","N3","N4"],"G":["P","O3'","N3","N2","N1","N7","O6"]}
                neighbour_atom_coords = {}
                for aa_name, aa_atoms in neighbour_atoms.items():
                    aa_atom_indexs = torch.tensor([restype3_to_atoms[aa_name].index(aa_atom) for aa_atom in aa_atoms],dtype=torch.long,device=affines.device)
                    aa_atypes = torch.tensor([restype_3_to_index[aa_name]]*len(affines),dtype=torch.long,device=affines.device)
                    torsion_angles_sin_cos = select_torsion_angles(torsion_angles, aa_atypes)
                    all_atoms = AtomPredictor(affines,torsion_angles_sin_cos,aa_atypes.detach().cpu().numpy())
                    neighbour_atom_coords[aa_name] = all_atoms[:,aa_atom_indexs]
                node_all_atoms = (neighbour_atom_coords["C"][...,:2,:] + neighbour_atom_coords["G"][...,:2,:])/2
                node_all_atoms = torch.cat((neighbour_atom_coords["ALA"],node_all_atoms,neighbour_atom_coords["C"][...,2:,:],neighbour_atom_coords["G"][...,2:,:]),dim=-2) #N 15 3
                node_all_atom_masks = torch.zeros(node_all_atoms.shape[:-1],device=affines.device,dtype=affines.dtype)
                node_all_atom_masks[prot_mask][...,5:] = 1
                node_all_atom_masks[~prot_mask][..., :5] = 1
                neighbour_distance_masks = node_all_atom_masks.unsqueeze(1).unsqueeze(3) + node_all_atom_masks[edge_index].unsqueeze(2) # N num_n 15 15
                neighbour_distances = (node_all_atoms.unsqueeze(1).unsqueeze(3) - node_all_atoms[edge_index].unsqueeze(2)).norm(dim=-1) # N num_n 15 15
                neighbour_distances = neighbour_distances + 1e3*neighbour_distance_masks
                neighbour_distances = self._rbf(neighbour_distances).flatten(2)
            neighbour_distances = self.distance_encoding2(neighbour_distances.requires_grad_())


        return BackboneDistanceEmbeddingOutput(
            pos3d_emb=position3d_embeddings,
            positions=positions,
            neighbour_positions=neighbour_positions,
            neighbour_distances=neighbour_distances,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
        )
