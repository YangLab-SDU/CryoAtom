"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/gnn_output.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import torch

from CryoAtom.utils.affine_utils import init_random_affine_from_translation


class GNNOutput:
    def __init__(
        self,
        positions: torch.Tensor = None,
        hidden_features: int = 256,
        init_affine: torch.Tensor = None,
    ):
        self.result_dict = {}
        self.keys = [
            "pred_positions",
            "cryo_edges",
            "cryo_edge_logits",
            "cryo_aa_logits",
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
    ):
        self.result_dict = {}
        for key in self.keys:
            self.result_dict[key] = []

        if positions is not None:
            self.result_dict["x"] = torch.zeros(
                positions.shape[0], hidden_features, device=positions.device
            )
            self.result_dict["x"].requires_grad_()

            self.result_dict["pred_affines"] = [
                (
                    init_random_affine_from_translation(positions).to(positions.device)
                    if init_affine is None
                    else init_affine
                ).requires_grad_()
            ]

    def to(self, device: str):
        for key in self.keys:
            if torch.is_tensor(self.result_dict[key]):
                self.result_dict[key] = self.result_dict[key].to(device)
            else:
                self.result_dict[key] = [x.to(device) for x in self.result_dict[key]]
