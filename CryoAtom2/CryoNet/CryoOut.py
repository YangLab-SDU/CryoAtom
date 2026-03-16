import torch

from CryoAtom2.utils.affine_utils import init_random_affine_from_translation


class CryoOutput:
    def __init__(
        self,
        positions: torch.Tensor = None,
        hidden_features: int = 256,
        init_affine: torch.Tensor = None,
        residual_repr: torch.Tensor = None,
        torsion_angles: torch.Tensor = None
    ):
        self.result_dict = {}
        self.keys = [
            "pred_positions",
            "cryo_edges",
            "cryo_edge_logits",
            "cryo_aa_logits",
            "edge_aa_logits",
            "local_confidence_score",
            "pred_existence_mask",
            "pred_affines",
            "pred_torsions",
            "seq_attention_scores",
            "x",
        ]

        self.refresh(
            positions=positions,
            hidden_features=hidden_features,
            init_affine=init_affine,
            residual_repr=residual_repr,
            torsion_angles = torsion_angles
        )

    def update(
        self,
        **kwargs,
    ):
        for key, value in kwargs.items():
            if key in self.result_dict:
                self.result_dict[key] = [value]

    def __getitem__(self, item):
        return self.result_dict[item]

    def __setitem__(self, key, value):
        self.result_dict[key] = value

    def refresh(
        self,
        positions: torch.Tensor = None,
        hidden_features: int = 256,
        init_affine: torch.Tensor = None,
        residual_repr: torch.Tensor = None,
        torsion_angles = None
    ):
        self.result_dict = {}
        for key in self.keys:
            self.result_dict[key] = []

        if positions is not None:
            self.result_dict["x"] = torch.zeros(
                positions.shape[0], hidden_features, device=positions.device,dtype=positions.dtype
            ) if residual_repr is None else residual_repr
            self.result_dict["x"].requires_grad_()

            self.result_dict["pred_affines"] = [
                (
                    init_random_affine_from_translation(positions).to(positions.device)
                    if init_affine is None
                    else init_affine
                ).requires_grad_()
            ]
            self.result_dict["pred_torsions"] = torsion_angles
    def to(self, device: str):
        for key in self.keys:
            if torch.is_tensor(self.result_dict[key]):
                self.result_dict[key] = self.result_dict[key].to(device)
            elif self.result_dict[key] is None:
                pass
            else:
                self.result_dict[key] = [x.to(device) for x in self.result_dict[key]]
        return self
