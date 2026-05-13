"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/gnn_inference_utils.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import torch
import torch.nn.functional as F
import argparse

import numpy as np
from contextlib import nullcontext
from scipy.spatial import cKDTree

from CryoAtom2.utils.PNAComplex import PNAComplex, get_PNAComplex_empty_except, get_lm_emb_to_PNAcomplex
from CryoAtom2.utils.affine_utils import init_random_affine_from_translation, get_affine_translation, get_affine, get_affine_rot
from CryoAtom2.utils.residue_constants import canonical_num_residues
from CryoAtom2.utils.pdb_utils import load_cas_c1s_from_structure,load_affines_cas_from_structure
from CryoAtom2.utils.fasta_utils import is_valid_fasta_ending, load_sequence_from_fasta_dict


def argmin_random(
    count_tensor: torch.Tensor,
    neighbours: torch.LongTensor,
    batch_size: int = 1,
    repeat_per_residue: int = 3,
):
    # We first look at the individual counts for each residue
    counts = count_tensor.clamp(max=repeat_per_residue)
    # If the proportion of clamped counts is too high, we use the full count tensor
    if torch.sum(counts == repeat_per_residue).item() / len(counts) > 0.7:
        neighbour_counts = count_tensor
    else:
        neighbour_counts = counts[neighbours].sum(dim=-1)
    rand_idxs = torch.randperm(len(neighbour_counts))
    corr_idxs = torch.arange(len(neighbour_counts))[rand_idxs]
    random_argmin = neighbour_counts[rand_idxs].argsort()[:batch_size]
    original_argmin = corr_idxs[random_argmin]
    return original_argmin


def get_neighbour_idxs(protein, k: int, idxs=None):
    # Get an initial set of pointers to neighbours for more efficient inference
    backbone_frames = protein.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    translation = get_affine_translation(backbone_frames)
    kd = cKDTree(translation)
    if idxs is None:
        _, init_neighbours = kd.query(translation, k=k)
    else:
        _, init_neighbours = kd.query(translation[idxs], k=k)
    return torch.from_numpy(init_neighbours)


def init_empty_collate_results(num_residues, device="cpu"):
    result = {}
    result["counts"] = torch.zeros(num_residues, device=device)
    result["pred_positions"] = torch.zeros(num_residues, 3, device=device)
    result["pred_affines"] = torch.zeros(num_residues, 3, 4, device=device)
    result["pred_torsions"] = torch.zeros(num_residues, canonical_num_residues * 5 + 3, 2, device=device)
    result["aa_logits"] = torch.zeros(num_residues, canonical_num_residues, device=device)
    result["local_confidence"] = torch.zeros(num_residues, device=device)
    result["existence_mask"] = torch.zeros(num_residues, device=device)
    result["edge_logits"] = torch.zeros(
        num_residues, 30, 2,device=device
    )


    result["edge_index"] = torch.zeros(
        num_residues, 30,dtype=torch.long, device=device
    )
    return result


def get_inference_data(complex, grid_data, idxs, seq_emb_masks=None,crop_length=300,num_devices: int = 1):
    grid = ((grid_data.grid - np.mean(grid_data.grid)) / np.std(grid_data.grid)).astype(
        np.float32
    )
    backbone_frames = complex.rigidgroups_gt_frames[:, 0]  # (num_res, 3, 4)
    prot_mask = complex.prot_mask
    res_positions = get_affine_translation(backbone_frames)
    picked_indices = np.arange(len(res_positions), dtype=int)

    batch = None
    batch_num = 1
    output_list = []
    batch_num_per_device = len(idxs) // num_devices
    for j in range(num_devices):
        if len(res_positions) >= crop_length:
            kd = cKDTree(res_positions)
            _, picked_indices = kd.query(
                res_positions[idxs[j * batch_num_per_device: (j + 1) * batch_num_per_device]], k=crop_length
            )
            batch_num = batch_num_per_device
            batch = torch.cat(
                [torch.ones(crop_length, dtype=torch.long) * i for i in range(batch_num)],
                dim=0,
            )

        output_dict = {
            "affines": torch.from_numpy(backbone_frames[picked_indices]),
            "cryo_grids": torch.from_numpy(grid[None]),  # Add channel dim
            "prot_mask": torch.from_numpy(prot_mask[picked_indices]),
            "cryo_global_origins": torch.from_numpy(
                grid_data.global_origin.astype(np.float32)
            ),
            "cryo_voxel_sizes": torch.from_numpy(grid_data.voxel_size.astype(np.float32)),
            "indices": torch.from_numpy(picked_indices),
            "num_nodes": len(picked_indices),
            "batch_num": batch_num,
            "batch": batch,
        }

        # add language embeddings
        if seq_emb_masks is not None:
            output_dict["sequence"] = torch.from_numpy(np.copy(complex.residue_to_lm_embedding).astype(np.float32))
            output_dict["sequence_mask"] = torch.from_numpy(np.copy(seq_emb_masks).astype(np.float32))
        output_list.append(output_dict)
    return output_list


def update_complex_gt_frames(
    complex: PNAComplex, update_indices: np.ndarray, update_affines: np.ndarray
) -> PNAComplex:
    complex.rigidgroups_gt_frames[update_indices, 0] = update_affines
    return complex


def collate_nn_results(
    collated_results, results, indices, complex, end_flag=False, crop_length=300, repeat_num:int=3, offset=0,
):
    num_pred_residues = crop_length//3
    update_slice = np.s_[offset : num_pred_residues + offset]
    if end_flag:
        repeat_logits = (collated_results["counts"][indices[update_slice]] > -1)
    else:
        repeat_logits = (collated_results["counts"][indices[update_slice]]<repeat_num)
    collated_results["counts"][indices[update_slice][repeat_logits]] += 1
    collated_results["pred_positions"][indices[update_slice][repeat_logits]] += results[
        "pred_positions"
    ][-1][update_slice][repeat_logits]
    collated_results["pred_torsions"][indices[update_slice][repeat_logits]] += torch.nn.functional.normalize(
        results["pred_torsions"][update_slice][repeat_logits], p=2, dim=-1
    )

    curr_pos_avg = (
        collated_results["pred_positions"][indices[update_slice][repeat_logits]]
        / collated_results["counts"][indices[update_slice][repeat_logits]][..., None]
    )
    collated_results["pred_affines"][indices[update_slice][repeat_logits]] = get_affine(
        get_affine_rot(results["pred_affines"][-1][update_slice][repeat_logits]).cpu(),
        curr_pos_avg
    )
    collated_results["aa_logits"][indices[update_slice][repeat_logits]] += results[
        "cryo_aa_logits"
    ][-1][update_slice][repeat_logits]
    collated_results["local_confidence"][indices[update_slice][repeat_logits]] = results[
        "local_confidence_score"
    ][-1][update_slice][repeat_logits]
    collated_results["existence_mask"][indices[update_slice][repeat_logits]] = results[
        "pred_existence_mask"
    ][-1][update_slice][repeat_logits]
    edge_index = results["cryo_edges"][-1][0].reshape(crop_length,30).long()
    collated_results["edge_index"][indices[update_slice][repeat_logits]] = indices[edge_index][update_slice][repeat_logits]
    collated_results["edge_logits"][indices[update_slice][repeat_logits]] = results[
        "cryo_edge_logits"
    ][-1][update_slice][repeat_logits].softmax(dim=2)[...,1:]

    protein = update_complex_gt_frames(
        complex,
        indices[update_slice].numpy(),
        collated_results["pred_affines"][indices[update_slice]].numpy(),
    )
    return collated_results, protein


@torch.no_grad()
def run_inference_on_data(
    module,
    meta_batch_list,
    run_iters: int = 3,
    seq_attention_batch_size: int = 200,
):
    with_seq = "sequence" in meta_batch_list[0]
    meta_input_list = []
    for data in meta_batch_list:
        affines = data["affines"]
        kwargs = {
            "positions": get_affine_translation(affines),
            "prot_mask": data['prot_mask'],
            "init_affine": affines,
            "run_iters": run_iters,
        }

        if with_seq:
            kwargs["seq_attention_batch_size"] = seq_attention_batch_size

        # others
        if data["batch_num"] == 1:
            if with_seq:
                kwargs["sequence"] = data["sequence"][None]
                kwargs["sequence_mask"] = data["sequence_mask"][None]
            kwargs["batch"] = None
            kwargs["cryo_grids"] = [data["cryo_grids"]]
            kwargs["cryo_global_origins"] = [data["cryo_global_origins"]]
            kwargs["cryo_voxel_sizes"] = [data["cryo_voxel_sizes"]]

        else:
            if with_seq:
                kwargs["sequence"] = (
                    data["sequence"][None].expand(data["batch_num"], -1, -1,)
                )
                kwargs["sequence_mask"] = (
                    data["sequence_mask"][None].expand(data["batch_num"], -1, -1,)
                )

            kwargs["batch"] = data["batch"]
            kwargs["cryo_grids"] = [
                data["cryo_grids"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_global_origins"] = [
                data["cryo_global_origins"] for _ in range(data["batch_num"])
            ]
            kwargs["cryo_voxel_sizes"] = [
                data["cryo_voxel_sizes"] for _ in range(data["batch_num"])
            ]

        meta_input_list.append(kwargs)
    result = module(meta_input_list)
    return result


def init_protein_from_see_alpha(see_alpha_file: str, fasta_dict: dict = None) -> PNAComplex:

    atom_locations = load_cas_c1s_from_structure(see_alpha_file)
    res_locations = np.concatenate([atom_locations['CA'],atom_locations['P']],axis=0)
    prot_mask = np.array(
        [True]*len(atom_locations['CA']) + [False]*len(atom_locations['P'])
    )
    rigidgroups_gt_frames = np.zeros((len(res_locations), 1, 3, 4), dtype=np.float32)
    rigidgroups_gt_frames[:, 0] = init_random_affine_from_translation(
        torch.from_numpy(res_locations)
    ).numpy()
    rigidgroups_gt_exists = np.ones((len(rigidgroups_gt_frames), 1), dtype=np.float32)
    if fasta_dict is None:
        return get_PNAComplex_empty_except(
            rigidgroups_gt_frames=rigidgroups_gt_frames,
            rigidgroups_gt_exists=rigidgroups_gt_exists,
            prot_mask=prot_mask,
        )
    else:
        unified_seq, unified_seq_len = load_sequence_from_fasta_dict(fasta_dict)

        return get_PNAComplex_empty_except(
            rigidgroups_gt_frames=rigidgroups_gt_frames,
            rigidgroups_gt_exists=rigidgroups_gt_exists,
            unified_seq=unified_seq,
            unified_seq_len=unified_seq_len,
            prot_mask = prot_mask,
        )


def get_final_nn_results(collated_results):
    final_results = {}

    final_results["pred_positions"] = (
        collated_results["pred_positions"] / collated_results["counts"][..., None]
    )
    final_results["pred_torsions"] = (
        collated_results["pred_torsions"] / collated_results["counts"][..., None, None]
    )
    final_results["pred_affines"] = get_affine(
        get_affine_rot(collated_results["pred_affines"]),
        final_results["pred_positions"],
    )
    final_results["aa_logits"] = (
        collated_results["aa_logits"] / collated_results["counts"][..., None]
    )

    final_results["local_confidence"] = collated_results["local_confidence"]
    final_results["existence_mask"] = collated_results["existence_mask"]

    final_results["raw_aa_entropy"] = (
        final_results["aa_logits"].softmax(dim=-1).log().sum(dim=-1)
    )
    final_results["normalized_aa_entropy"] = final_results["raw_aa_entropy"].add(
        -final_results["raw_aa_entropy"].min()
    )
    final_results["normalized_aa_entropy"] = final_results["normalized_aa_entropy"].div(
        final_results["normalized_aa_entropy"].max()
    )

    final_results["edge_logits"] = collated_results["edge_logits"]
    final_results["edge_index"] = collated_results["edge_index"]

    return dict([(k, v.numpy()) for (k, v) in final_results.items()])

