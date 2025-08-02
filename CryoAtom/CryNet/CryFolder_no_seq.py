import contextlib
import einops
import numpy as np
import torch
from torch import nn
from einops.layers.torch import Rearrange
from CryoAtom.CryNet.features_init import CryoFeatures
from CryoAtom.CryNet.Cryformer import Cryformer_no_seq
from CryoAtom.CryNet.structure_module import InvariantPointAttention,Transition,LinearWithShortcut,LinearWithSeq,LinearWithEdge
from CryoAtom.CryNet.backbone_frame import BackboneFrameNet
from CryoAtom.CryNet.Cryout import GNNOutput
from CryoAtom.utils.affine_utils import get_affine_translation,affine_from_3_points,get_affine,get_affine_rot

class CryFolder(nn.Module):
    def __init__(
            self,
            hidden_features:int,
            attention_features: int = 48,
            attention_heads:int = 8,
            query_points:int = 4,
            num_neighbours:int = 20,
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
                Cryformer_no_seq(
                    in_features=hidden_features,
                    attention_features=self.afz,
                    attention_heads=attention_heads,
                    num_neighbours=num_neighbours,
                    activation_class=activation_function
                ) for _ in range(num_layers_former)
            ]
        )
        self.lin_edge = torch.nn.Linear(hidden_features,hidden_features//2)
        self.ipa = InvariantPointAttention(in_features=hidden_features,c = self.afz)
        self.ipa_transition = Transition(in_features=hidden_features)
        self.cryo_aa = LinearWithSeq(
            in_features = self.hfz,
            hidden_features = self.hfz//2,
            out_features = 20
        )
        self.cryo_edge = LinearWithEdge(
            in_features = self.hfz,
            hidden_features = self.hfz//2,
            neibour_dim = 3+self.afz//3,
            out_features = 3
        )
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
        self.torsion_angle_fc = LinearWithShortcut(in_features = self.hfz,hidden_features = self.hfz//2,out_features = 83*2)
    def forward(
            self,
            positions = None,
            init_affine = None,
            run_iters:int=1,
            **kwargs,
    ) -> GNNOutput:
        assert positions is not None
        result = GNNOutput(positions=positions,init_affine=init_affine,hidden_features=self.hfz)
        for run_iter in range(run_iters):
            notlast_flag = (run_iter!=(run_iters-1))
            with torch.no_grad() if notlast_flag else contextlib.nullcontext():
                result['x'],x2,edge_index,cryo_edges,cryo_aa_logits,neighbour_emb,pos3d_emb = self.cryofeature(affines = result["pred_affines"][-1],**kwargs)
                for idx in range(self.num_layers_former):
                    result['x'], x2 = self.formers[idx](x_1=result['x'], x_2=x2, pos_emb=pos3d_emb,
                                                        edge_index=edge_index)
                cryo_edge_logits = self.cryo_edge(neighbour_emb,x2)
                cryo_aa_logits = self.cryo_aa(cryo_aa_logits,result['x'])
                node_residual = result['x']
                x2 = self.lin_edge(x2)
                for idx in range(self.num_layers_ipa):
                    result.update(pred_affines=get_affine(get_affine_rot(result["pred_affines"][-1]).detach(),get_affine_translation(result["pred_affines"][-1])))
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
                result["pred_torsions"] = einops.rearrange(result["pred_torsions"], "n (f d) -> n f d",f=83,d=2)
            if notlast_flag :
                result = GNNOutput(
                    positions=result["pred_positions"][-1],
                    init_affine=result["pred_affines"][-1],
                    hidden_features=self.hfz,
                )
        return result