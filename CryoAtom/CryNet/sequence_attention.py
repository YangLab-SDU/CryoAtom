"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/sequence_attention.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
from collections import namedtuple, OrderedDict
import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from functools import partial
from CryoAtom.utils.torch_utlis import get_batch_slices, padded_sequence_softmax

SequenceAttentionOutput = namedtuple(
    "SequenceAttentionOutput", ["x", "seq_aa_logits", "seq_attention_scores"]
)
def get_batched_sequence_attention_scores(
    sequence_query,  # naf
    sequence_key,  # 1saf
    batch,  # n
    attention_scale,
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_query.shape[0], *sequence_key.shape[1:3], device=device
    )  # nsa
    n_len, s_len = output.shape[:2]
    seq_batches = get_batch_slices(s_len, batch_size)
    sequence_query = sequence_query[:, None]
    for seq_batch in seq_batches:
        output[:, seq_batch] = (sequence_query * sequence_key[:, seq_batch][batch]).sum(
            dim=-1
        ) / attention_scale
    return output
def get_batched_sequence_attention_features(
    sequence_attention_weights,  # nsa
    sequence_value,  # 1saf
    batch,  # n
    batch_size=200,
    device="cpu",
):
    output = torch.zeros(
        sequence_attention_weights.shape[0], *sequence_value.shape[2:], device=device
    )  # naf
    n_len, s_len = sequence_attention_weights.shape[:2]
    seq_batches = get_batch_slices(s_len, batch_size)

    for seq_batch in seq_batches:
        output += (
            sequence_attention_weights[:, seq_batch][..., None]
            * sequence_value[:, seq_batch][batch]
        ).sum(dim=1)
    return output  # naf

class SequenceAttention(nn.Module):
    def __init__(
        self,
        sequence_features: int,
        in_features: int,
        attention_features: int = 64,
        attention_heads: int = 8,
        activation_class: nn.Module = nn.ReLU,
        checkpoint: bool = True,
    ):
        super().__init__()
        self.sfz = sequence_features
        self.ifz = in_features
        self.afz = in_features if attention_features is None else attention_features
        self.ahz = attention_heads
        self.attention_scale = math.sqrt(self.afz)
        self.sqrt2 = math.sqrt(2)
        self.norm = nn.LayerNorm(in_features)
        self.q = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=False),
            Rearrange(
                "n (ahz afz) -> n ahz afz",
                ahz=self.ahz,
                afz=self.afz,
            ),
        )
        self.k = nn.Sequential(
            nn.Linear(self.sfz, self.ahz * self.afz, bias=False),
            Rearrange(
                "b s (ahz afz) -> b s ahz afz",
                ahz=self.ahz,
                afz=self.afz,
            ),
        )
        self.v = nn.Sequential(
            nn.Linear(self.sfz, self.ahz * self.afz, bias=False),
            Rearrange(
                "b s (ahz afz) -> b s ahz afz",
                ahz=self.ahz,
                afz=self.afz,
            ),
        )
        self.gated = nn.Sequential(
            nn.Linear(self.ifz, self.ahz * self.afz, bias=True),
            Rearrange("N (ahz afz) -> N ahz afz", ahz=self.ahz, afz=self.afz),
            nn.Sigmoid()
        )
        # self.ag = nn.Sequential(
        #     OrderedDict(
        #         [
        #             (
        #                 "rearrange",
        #                 Rearrange(
        #                     "n ahz afz -> n (ahz afz)",
        #                     ahz=self.ahz,
        #                     afz=self.afz,
        #                 ),
        #             ),
        #             ("ln", nn.LayerNorm(self.ahz * self.afz)),
        #             ("linear", nn.Linear(self.ahz * self.afz, self.ifz, bias=False)),
        #             (
        #                 "dropout",
        #                 nn.Dropout(p=0.3),
        #             ),
        #         ]
        #     )
        # )
        self.back = nn.Sequential(
            Rearrange("N ahz afz -> N (ahz afz)", ahz=self.ahz, afz=self.afz),
            nn.Linear(self.ahz * self.afz, self.ifz)
        )
        self.forward = self.forward_checkpoint if checkpoint else self.forward_normal

    def forward_normal(
        self,
        x,
        packed_sequence_emb,
        packed_sequence_mask,
        batch=None,
        attention_batch_size=200,
        **kwargs,
    ):
        return self._intern_forward(
            x, packed_sequence_emb, packed_sequence_mask, batch, attention_batch_size
        )

    def forward_checkpoint(
        self,
        x: torch.Tensor,
        packed_sequence_emb: torch.Tensor,
        packed_sequence_mask: torch.Tensor,
        batch=None,
        attention_batch_size: int = 200,
        **kwargs,
    ) -> SequenceAttentionOutput:
        new_forward = partial(
            self._intern_forward,
            packed_sequence_emb=packed_sequence_emb,
            packed_sequence_mask=packed_sequence_mask,
            batch=batch,
            attention_batch_size=attention_batch_size
        )
        return torch.utils.checkpoint.checkpoint(
            new_forward,
            x,
            preserve_rng_state=False,
        )

    def _intern_forward(
        self,
        x: torch.Tensor,
        packed_sequence_emb: torch.Tensor,
        packed_sequence_mask: torch.Tensor,
        batch,
        attention_batch_size: int,
    ) -> SequenceAttentionOutput:
        device = x.device
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        y = x
        sequence_query = self.q(y)  # (n, ahz, afz)
        sequence_key = self.k(packed_sequence_emb)  # (1, seq_len, ahz, afz)
        sequence_value = self.v(packed_sequence_emb)  # (1, seq_len, ahz, afz)

        sequence_attention_scores = get_batched_sequence_attention_scores(
            sequence_query,
            sequence_key,
            batch,
            self.attention_scale,
            batch_size=attention_batch_size,
            device=device,
        )

        batched_mask = packed_sequence_mask[batch].unsqueeze(-1)  # (n, seq_len, 1)
        # Since sequence emb was padded, do not consider the padded parts for attention
        sequence_attention_weights = padded_sequence_softmax(
            sequence_attention_scores, batched_mask, dim=1
        )

        new_features_attention = get_batched_sequence_attention_features(
            sequence_attention_weights,
            sequence_value,
            batch,
            batch_size=attention_batch_size,
            device=device,
        )
        gate = self.gated(y)
        new_features = self.back(gate*new_features_attention)
        new_features = self.norm(math.sqrt(2)*x + new_features)
        return new_features,torch.sum(sequence_attention_scores[...,:2],dim=-1)
