import contextlib
import einops
import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange
from CryoAtom2.CryoNet.features_init import CryoFeatures
from CryoAtom2.CryoNet.CryoFormer import CryoFormer
from CryoAtom2.CryoNet.structure_module import InvariantPointAttention,Transition,LinearWithShortcut,LinearWithSeq,LinearWithEdge
from CryoAtom2.CryoNet.backbone_frame import BackboneFrameNet
from CryoAtom2.CryoNet.CryoOut import CryoOutput
from CryoAtom2.utils.affine_utils import get_affine_translation,affine_from_3_points,get_affine,get_affine_rot
from CryoAtom2.utils.residue_constants import canonical_num_residues
class CryoFolder(nn.Module):
    def __init__(
            self,
            hidden_features:int,
            attention_features: int = 48,
            attention_heads:int = 8,
            query_points:int = 4,
            num_neighbours:int = 30,
            num_layers_former:int = 16,
            num_layers_ipa: int = 4,
            activation_function:nn.Module = nn.ReLU,

    ):
        super().__init__()
        self.hfz = hidden_features
        self.afz = attention_features
        self.ahz = attention_heads
        self.qpz = query_points
        self.kz = num_neighbours
        self.num_layers_former = num_layers_former
        self.num_layers_ipa = num_layers_ipa
        self.cryofeature = CryoFeatures(in_features=hidden_features,attention_features=self.afz)
        self.formers = nn.ModuleList(
            [
                CryoFormer(
                    in_features=hidden_features,
                    attention_features=self.afz,
                    attention_heads=attention_heads,
                    num_neighbours=num_neighbours,
                    do_seq_attn=False
                ) for ii in range(num_layers_former)
            ]
        )
        self.ipa = InvariantPointAttention(in_features=hidden_features,c = self.afz)
        self.ipa_transition = Transition(in_features=hidden_features)
        self.cryo_aa = LinearWithSeq(
            in_features = self.hfz,
            hidden_features = self.hfz//2,
            out_features = canonical_num_residues
        )
        self.cryo_edge = LinearWithSeq(
            in_features = self.hfz//2,
            hidden_features = self.hfz//4,
            out_features = 3
        )
        self.cryo_edge_MRF = LinearWithEdge(in_features=self.hfz//2,out_features=canonical_num_residues*canonical_num_residues)
        self.backbome_frame = BackboneFrameNet(self.hfz)
        self.local_confidence_predictor = LinearWithShortcut(
            in_features = self.hfz,
            hidden_features = self.hfz//2,
            out_features = 1
        )
        self.existence_mask_predictor = LinearWithShortcut(
            in_features = self.hfz,
            hidden_features = self.hfz//2,
            out_features = 1
        )
        self.torsion_angle_fc = LinearWithShortcut(in_features = self.hfz,hidden_features = self.hfz//2,out_features = (canonical_num_residues * 5 + 3) * 2)

    def forward(
            self,
            sequence=None,
            sequence_mask=None,
            positions = None,
            init_affine = None,
            prot_mask = None,
            run_iters:int=1,
            seq_attention_batch_size: int = 200,
            **kwargs,
    ) -> CryoOutput:
        assert positions is not None
        result = CryoOutput(positions=positions,init_affine=init_affine,hidden_features=self.hfz)

        for run_iter in range(run_iters):
            notlast_flag = (run_iter!=(run_iters-1))
            with torch.no_grad() if notlast_flag else contextlib.nullcontext():
                result['x'],x2,edge_index,cryo_edges,cryo_aa_logits,pos3d_emb = self.cryofeature(
                    affines = result["pred_affines"][-1],
                    prot_mask=prot_mask,
                    residual_repr=result['x'],
                    torsion_angles=result["pred_torsions"],
                    **kwargs)
                neighbour_emb = x2
                for idx in range(self.num_layers_former):
                    result['x'],x2,attention_scores = self.formers[idx](x_1=result['x'],x_2=x2,pos_emb=pos3d_emb,edge_index=edge_index,
                                                packed_sequence_emb=sequence,
                                                packed_sequence_mask = sequence_mask,
                                                prot_mask = prot_mask,
                                                attention_batch_size=seq_attention_batch_size,
                                                **kwargs
                                                )
                cryo_edge_logits = self.cryo_edge(neighbour_emb,x2)
                cryo_aa_logits = self.cryo_aa(cryo_aa_logits,result['x'])
                result["edge_aa_logits"] = self.cryo_edge_MRF(neighbour_emb,x2)
                result["edge_aa_logits"] = einops.rearrange(result["edge_aa_logits"], "n k (f d) -> n k f d",
                                                           f=canonical_num_residues, d=canonical_num_residues)
                node_residual = result['x']
                for idx in range(self.num_layers_ipa):
                    result['x'] = self.ipa(x1=result['x'],x2=x2,affines = result["pred_affines"][-1],pos_emb=pos3d_emb,edge_index=edge_index)
                    result['x'] = self.ipa_transition(result['x'])
                    new_affine = self.backbome_frame(
                        x=result['x'],
                        affine=result["pred_affines"][-1]
                    )
                    local_confidence_score = self.local_confidence_predictor(node_residual,result["x"])
                    local_confidence_score = local_confidence_score.flatten()
                    pred_existence_mask = self.existence_mask_predictor(node_residual,result["x"])
                    pred_existence_mask = pred_existence_mask.flatten()
                    result.update(
                        pred_affines=new_affine,
                        pred_positions=get_affine_translation(new_affine),
                        cryo_edges=cryo_edges,
                        cryo_edge_logits=cryo_edge_logits,
                        cryo_aa_logits=cryo_aa_logits,
                        local_confidence_score=local_confidence_score,
                        pred_existence_mask=pred_existence_mask,
                    )
                result["pred_torsions"] = self.torsion_angle_fc(node_residual,result["x"])
                result["pred_torsions"] = einops.rearrange(result["pred_torsions"], "n (f d) -> n f d",f=canonical_num_residues * 5 + 3,d=2)

            if notlast_flag :
                result = CryoOutput(
                    positions=result["pred_positions"][-1].detach(),
                    init_affine=result["pred_affines"][-1].detach(),
                    hidden_features=self.hfz,
                    residual_repr = result['x'].detach(),
                    torsion_angles = result["pred_torsions"].detach()
                )
        return result
