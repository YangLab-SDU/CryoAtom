"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/flood_fill.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import math
import os

import numpy as np
from collections import namedtuple
from typing import Dict, List
import torch
from scipy.spatial import cKDTree

from CryoAtom2.utils.aa_probs_to_hmm import dump_aa_logits_to_hmmsearch_file
from CryoAtom2.utils.hmm_sequence_align import (
    FixChainsOutput,
    fix_chains_pipeline,
    prune_and_connect_chains,
    get_aa_from_aalogits
)
from CryoAtom2.utils.save_pdb_utils import chain_atomc_to_cif, number_to_chain_str, write_chain_report, \
    write_chain_probabilities
from CryoAtom2.utils.PNAComplex import (
    frames_and_literature_positions_to_atomc_pos,
    torsion_angles_to_frames,
)
from CryoAtom2.utils.residue_constants import restype_atomc_mask, select_torsion_angles, restype3_to_atoms, num_prot, \
    canonical_num_residues
from CryoAtom2.utils.affine_utils import get_affine_translation
FloodFillChain = namedtuple("FloodFillChain", ["start_N", "end_C", "residues"])

def numpy_softmax(x,axis):
    exp_logits = np.exp(x)
    return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)

def shift_score(
    local_confidence_score: np.ndarray,
    best_value= (1,0.4),
    worst_value= (4,1.2),
) -> np.ndarray:
    x1,y1 = best_value
    x2,y2 = worst_value
    return (local_confidence_score-x1)*(y2-y1)/(x2-x1) + y1
def normalize_local_confidence_score(
    local_confidence_score: np.ndarray,
    best_value= 0.4,
    worst_value= 1.2,
) -> np.ndarray:
    normalized_score = (worst_value - local_confidence_score) / (
        worst_value - best_value
    )
    normalized_score = np.clip(normalized_score, 0, 1)
    return normalized_score

def remove_overlapping_ca(
    ca_positions: np.ndarray,bfactors,existence_mask=None,radius_threshold: float = 0.3,
) -> np.ndarray:
    kdtree = cKDTree(ca_positions)
    bfactors_copy = np.copy(bfactors)
    sorted_indices = np.argsort(bfactors_copy)[::-1]
    if existence_mask is None:
        existence_mask = np.ones(len(ca_positions), dtype=bool)

    for i in sorted_indices:
        if existence_mask[i]:
            too_close = np.array(
                kdtree.query_ball_point(ca_positions[i], r=radius_threshold,)
            )
            too_close = too_close[too_close != i]
            existence_mask[too_close] = False
    return existence_mask

def chains_to_atoms(
    final_results: Dict,
    fix_chains_output: FixChainsOutput,
    backbone_affine,
    existence_mask,
):
    fixed_aatype_from_sequence = fix_chains_output.best_match_output.new_sequences
    chains = fix_chains_output.chains
    aa_probs = torch.from_numpy(final_results["aa_logits"][existence_mask]).softmax(dim=-1).numpy()

    (
        chain_all_atoms, chain_atom_mask, chain_bfactors, chain_aa_probs,
    ) = (
        [],
        [],
        [],
        [],
    )
    # Everything below is in the order of chains
    for chain_id in range(len(chains)):
        chain_id_backbone_affine = backbone_affine[chains[chain_id]]
        torsion_angles = select_torsion_angles(
            torch.from_numpy(final_results["pred_torsions"][existence_mask])[
                chains[chain_id]
            ],
            aatype=fixed_aatype_from_sequence[chain_id],
        )

        all_frames = torsion_angles_to_frames(
            fixed_aatype_from_sequence[chain_id],
            chain_id_backbone_affine,
            torsion_angles,
        )
        chain_all_atoms.append(
            frames_and_literature_positions_to_atomc_pos(
                fixed_aatype_from_sequence[chain_id], all_frames
            )
        )
        chain_atom_mask.append(
            restype_atomc_mask[fixed_aatype_from_sequence[chain_id]]
        )
        chain_bfactors.append(
            normalize_local_confidence_score(
                final_results["local_confidence"][existence_mask][chains[chain_id]]
            )
            * 100
        )
        chain_aa_probs.append(
            aa_probs[chains[chain_id]]
        )
    return (
        chain_all_atoms,
        chain_atom_mask,
        chain_bfactors,
        chain_aa_probs,
    )
def final_results_to_cif(
    final_results,
    cif_path,
    protein_sequences: List = None,
    rna_sequences: List = None,
    dna_sequences: List = None,
    prot_mask=None,
    aatype=None,
    verbose=False,
    print_fn=print,
    aggressive_pruning=False,
    mask_threshold = 0.3,
    end_flag = False
):
    """
    Currently assumes the ordering it comes with, I will change this later
    """
    # prot_mask = np.copy(prot_mask)
    # prot_mask_change = torch.from_numpy(final_results["aa_logits"]).softmax(dim=-1).numpy()
    # prot_mask_change = prot_mask_change[...,:num_prot].sum(axis=-1)
    # prot_mask[prot_mask_change<0.1] = False
    # prot_mask[prot_mask_change>0.9] = True
    final_results["local_confidence"][~prot_mask] = shift_score(final_results["local_confidence"][~prot_mask])
    has_sequences = protein_sequences is not None or rna_sequences is not None or dna_sequences is not None
    final_results["aa_logits"][prot_mask] *= math.log(canonical_num_residues,num_prot)
    final_results["aa_logits"][~prot_mask] *= math.log(canonical_num_residues,canonical_num_residues-num_prot)
    final_results["aa_logits"][prot_mask][...,num_prot:] = -100
    final_results["aa_logits"][~prot_mask][...,:num_prot] = -100
    bfactors = normalize_local_confidence_score(final_results["local_confidence"]) * 100
    backbone_affine = torch.from_numpy(final_results["pred_affines"])
    existence_mask = (
        torch.from_numpy(final_results["existence_mask"]).sigmoid() > mask_threshold
    ).numpy()
    existence_mask = remove_overlapping_ca(ca_positions=get_affine_translation(backbone_affine),bfactors=bfactors,existence_mask=existence_mask,radius_threshold=1.5 if end_flag else 0.5)
    existence_mask[np.logical_and(existence_mask,~prot_mask)] *= remove_overlapping_ca(ca_positions=get_affine_translation(backbone_affine[np.logical_and(existence_mask,~prot_mask)]),bfactors=bfactors[np.logical_and(existence_mask,~prot_mask)],radius_threshold=3 if end_flag else 1)
    if has_sequences:
        if protein_sequences is None:
            existence_mask *= ~prot_mask
        if rna_sequences is None and dna_sequences is None:
            existence_mask *= prot_mask
    if aatype is None:
        aatype = np.zeros((len(final_results["aa_logits"]),), dtype=np.int64)
        aatype[prot_mask] = np.argmax(
            final_results["aa_logits"][prot_mask][..., :num_prot], axis=-1
        )
        aatype[~prot_mask] = (
                np.argmax(final_results["aa_logits"][~prot_mask][..., num_prot:], axis=-1)
                + num_prot
        )
        aatype = aatype[existence_mask]
    backbone_affine = backbone_affine[existence_mask]
    mask2unmask = np.arange(len(existence_mask))[existence_mask]

    torsion_angles = select_torsion_angles(
        torch.from_numpy(final_results["pred_torsions"][existence_mask]), aatype=aatype
    )
    edge_logits = final_results["edge_logits"][existence_mask]
    edge_index = final_results['edge_index'][existence_mask]
    all_frames = torsion_angles_to_frames(aatype, backbone_affine, torsion_angles)
    all_atoms = frames_and_literature_positions_to_atomc_pos(aatype, all_frames)
    atom_mask = restype_atomc_mask[aatype]
    bfactors = (
        normalize_local_confidence_score(
            final_results["local_confidence"][existence_mask]
        )
        * 100
    )

    all_atoms_np = all_atoms.numpy()
    all_atom_idxs = np.arange(len(all_atoms_np))
    prot_mask = prot_mask[existence_mask]
    chains = []
    if np.any(prot_mask):
        idxs = all_atom_idxs[prot_mask]
        prot_chains = flood_fill(
            all_atoms_np[prot_mask], bfactors[prot_mask],edge_logits[prot_mask],edge_index[prot_mask],mask2unmask[prot_mask], is_nucleotide=False
        )
        chains += [idxs[c] for c in prot_chains]
    if np.any(~prot_mask):
        idxs = all_atom_idxs[~prot_mask]
        nuc_chains = flood_fill(
            all_atoms_np[~prot_mask],
            bfactors[~prot_mask],
            edge_logits[~prot_mask],
            edge_index[~prot_mask],
            mask2unmask[~prot_mask],
            is_nucleotide=True,
            n_c_distance_threshold=4,
        )
        chains += [idxs[c] for c in nuc_chains]


    # Prune chains based on length
    pruned_chains = chains
    for cc in pruned_chains:
        if np.all(~prot_mask[cc]):
            aatype[cc] = get_aa_from_aalogits(final_results["aa_logits"][existence_mask][cc])
    torsion_angles = select_torsion_angles(
        torch.from_numpy(final_results["pred_torsions"][existence_mask]), aatype=aatype
    )
    all_frames = torsion_angles_to_frames(aatype, backbone_affine, torsion_angles)
    all_atoms = frames_and_literature_positions_to_atomc_pos(aatype, all_frames)
    atom_mask = restype_atomc_mask[aatype]
    all_atoms_np = all_atoms.numpy()
    chain_atomc_to_cif(
        [aatype[c] for c in pruned_chains],
        [all_atoms[c] for c in pruned_chains],
        [atom_mask[c] for c in pruned_chains],
        cif_path,
        bfactors=[bfactors[c] for c in pruned_chains],
    )

    new_final_results = {}
    new_final_results["chain_aa_logits"] = [
        final_results["aa_logits"][existence_mask][c] for c in chains
    ]
    new_final_results["pruned_chain_aa_logits"] = [
        final_results["aa_logits"][existence_mask][c] for c in pruned_chains
    ]
    new_final_results["chain_prot_mask"] = [prot_mask[c] for c in chains]
    new_final_results["pruned_chain_prot_mask"] = [prot_mask[c] for c in pruned_chains]
    if end_flag:
        # Can make HMM profiles with the aa_probs
        hmm_dir_path = os.path.join(os.path.dirname(cif_path), "net_hmm_profiles")
        os.makedirs(hmm_dir_path, exist_ok=True)
        for i, chain_aa_logits in enumerate(new_final_results["pruned_chain_aa_logits"]):
            chain_name = number_to_chain_str(i)
            if np.any(new_final_results["pruned_chain_prot_mask"][i]):
                dump_aa_logits_to_hmmsearch_file(
                    chain_aa_logits,
                    os.path.join(hmm_dir_path, f"{chain_name}_prot.hmm"),
                    name=f"{chain_name}",
                    alphabet_type="amino",
                )
            else:
                dump_aa_logits_to_hmmsearch_file(
                    chain_aa_logits,
                    os.path.join(hmm_dir_path, f"{chain_name}_na.hmm"),
                    name=f"{chain_name}",
                    alphabet_type="PP",
                )
    if has_sequences:
        ca_pos = all_atoms_np[:, 1]

        fix_chains_output = fix_chains_pipeline(
            prot_sequences=protein_sequences,
            rna_sequences=rna_sequences,
            dna_sequences=dna_sequences,
            chains=chains,
            chain_aa_logits=new_final_results["chain_aa_logits"],
            ca_pos=ca_pos,
            chain_prot_mask=new_final_results["chain_prot_mask"],
            base_dir=os.path.dirname(cif_path),
        )

        chain_all_atoms, chain_atom_mask, chain_bfactors, chain_aa_probs = chains_to_atoms(
            final_results, fix_chains_output, backbone_affine, existence_mask
        )

        for chain_id, chain in enumerate(fix_chains_output.chains):
            ca_pos[chain] = chain_all_atoms[chain_id][:, 1]

        chain_atomc_to_cif(
            fix_chains_output.best_match_output.new_sequences,
            chain_all_atoms,
            chain_atom_mask,
            cif_path.replace("net.cif", "fix.cif"),
            bfactors=chain_bfactors,
        )

        write_chain_report(
            cif_path.replace("net.cif", "_chain_report.csv"),
            sequence_idxs=fix_chains_output.best_match_output.sequence_idxs,
            bfactors=chain_bfactors,
            match_scores=fix_chains_output.best_match_output.match_scores,
            chain_prune_length=4,
            hmm_output_match_sequences=fix_chains_output.best_match_output.hmm_output_match_sequences,
        )
        merged_sequences = []
        for merged_sequence in (protein_sequences,rna_sequences,dna_sequences):
            if merged_sequence is not None:
                merged_sequences = merged_sequences + merged_sequence
        match_original_seq_len = np.array([len(seq) for seq in merged_sequences])

        fix_chains_output = prune_and_connect_chains(
            fix_chains_output.chains,
            fix_chains_output.best_match_output,
            ca_pos,
            aggressive_pruning=aggressive_pruning,
            chain_prune_length=4,
            match_original_seq_len=match_original_seq_len
        )

        chain_all_atoms, chain_atom_mask, chain_bfactors, chain_aa_probs = chains_to_atoms(
            final_results, fix_chains_output, backbone_affine, existence_mask
        )

        chain_atomc_to_cif(
            fix_chains_output.best_match_output.new_sequences,
            chain_all_atoms,
            chain_atom_mask,
            cif_path.replace("net.cif", "prune.cif"),
            bfactors=chain_bfactors,
            sequence_idxs=fix_chains_output.best_match_output.sequence_idxs,
            res_idxs=fix_chains_output.best_match_output.residue_idxs
            if aggressive_pruning
            else None,
        )

        write_chain_probabilities(
            cif_path.replace("net.cif", "_aa_probabilities.aap"),
            bfactors=chain_bfactors,
            aa_probs=chain_aa_probs,
            chain_prune_length=4,
        )

        if (
            verbose
            and fix_chains_output.unmodelled_sequences is not None
            and len(fix_chains_output.unmodelled_sequences) > 0
        ):
            print_fn(
                f"These sequence ids have been left unmodelled: {fix_chains_output.unmodelled_sequences}"
            )

    return new_final_results
def BayesCoreect(possible_indices,idx,dists,edge_logits,edge_index,mask2unmask,eps=1e-3):
    possible_logits = np.array([
        edge_logits[idx][edge_index[idx] == mask2unmask[poi]][0]
        if np.any(edge_index[idx] == mask2unmask[poi])
        else -1.0
        for poi in possible_indices
    ], dtype=np.float64)
    possible_logits[possible_logits == -1] = max(np.max(possible_logits),0.01)
    dist_logits = 1-np.exp(-(1.5/(dists+eps))**6)
    return np.argsort(1-dist_logits*possible_logits)
def flood_fill(atomc_positions, b_factors,edge_logits,edge_index,mask2unmask,n_c_distance_threshold=2.1, is_nucleotide=False,):
    if is_nucleotide:
        n_idx, c_idx = (
            restype3_to_atoms["A"].index("P"),
            restype3_to_atoms["A"].index("O3'"),
        )
        n_c_distance_threshold = n_c_distance_threshold + 0.5
    else:
        n_idx, c_idx = (
            restype3_to_atoms["ALA"].index("N"),
            restype3_to_atoms["ALA"].index("C"),
        )

    n_positions = atomc_positions[:, n_idx]
    c_positions = atomc_positions[:, c_idx]
    kdtree = cKDTree(c_positions)
    b_factors_copy = np.copy(b_factors)

    chains = []
    chain_ends = {}
    while np.any(b_factors_copy != -1):
        idx = np.argmax(b_factors_copy)
        possible_indices = np.array(
            kdtree.query_ball_point(
                n_positions[idx],
                r=n_c_distance_threshold,
                return_sorted=True
            )
        )
        possible_indices = possible_indices[possible_indices != idx]
        got_chain = False
        if len(possible_indices) > 0:
            pos_dits = np.sqrt(np.sum(np.square(n_positions[idx][None] - c_positions[possible_indices]), axis=-1))
            possible_indices = possible_indices[BayesCoreect(possible_indices, idx, pos_dits, edge_logits[...,1], edge_index, mask2unmask)]
            for possible_prev_residue in possible_indices:
                if possible_prev_residue == idx:
                    continue
                if possible_prev_residue in chain_ends:
                    chains[chain_ends[possible_prev_residue]].append(idx)
                    chain_ends[idx] = chain_ends[possible_prev_residue]
                    del chain_ends[possible_prev_residue]
                    got_chain = True
                    break
                elif b_factors_copy[possible_prev_residue] >= 0.0:
                    chains.append([possible_prev_residue, idx])
                    chain_ends[idx] = len(chains) - 1
                    b_factors_copy[possible_prev_residue] = -1
                    got_chain = True
                    break

        if not got_chain:
            chains.append([idx])
            chain_ends[idx] = len(chains) - 1

        b_factors_copy[idx] = -1

    og_chain_starts = np.array([c[0] for c in chains])
    og_chain_ends = np.array([c[-1] for c in chains])

    chain_starts = og_chain_starts.copy()
    chain_ends = og_chain_ends.copy()

    n_chain_starts = n_positions[chain_starts]
    c_chain_ends = c_positions[chain_ends]
    N = len(chain_starts)
    spent_starts, spent_ends = set(), set()

    kdtree = cKDTree(n_chain_starts)

    no_improvement = 0
    chain_end_match = 0

    while no_improvement < 2 * N:
        found_match = False
        if chain_end_match in spent_ends:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N
            continue

        start_matches = kdtree.query_ball_point(
            c_chain_ends[chain_end_match], r=n_c_distance_threshold, return_sorted=True
        )
        if len(start_matches)>0:
            start_matches = np.array(start_matches)
            pos_dits = np.sqrt(np.sum(np.square(c_chain_ends[chain_end_match][None] - n_chain_starts[start_matches]), axis=-1))
            start_matches = start_matches[BayesCoreect(og_chain_starts[start_matches], og_chain_ends[chain_end_match], pos_dits, edge_logits[...,0], edge_index, mask2unmask)]
        for chain_start_match in start_matches:
            if (
                chain_start_match not in spent_starts
                and chain_end_match != chain_start_match
            ):
                chain_start_match_reidx = np.nonzero(
                    chain_starts == og_chain_starts[chain_start_match]
                )[0][0]
                chain_end_match_reidx = np.nonzero(
                    chain_ends == og_chain_ends[chain_end_match]
                )[0][0]
                if chain_start_match_reidx == chain_end_match_reidx:
                    continue

                new_chain = (
                    chains[chain_end_match_reidx] + chains[chain_start_match_reidx]
                )

                chain_arange = np.arange(len(chains))
                tmp_chains = np.array(chains, dtype=object)[
                    chain_arange[
                        (chain_arange != chain_start_match_reidx)
                        & (chain_arange != chain_end_match_reidx)
                    ]
                ].tolist()
                tmp_chains.append(new_chain)
                chains = tmp_chains

                chain_starts = np.array([c[0] for c in chains])
                chain_ends = np.array([c[-1] for c in chains])

                spent_starts.add(chain_start_match)
                spent_ends.add(chain_end_match)
                no_improvement = 0
                found_match = True
                chain_end_match = (chain_end_match + 1) % N
                break

        if not found_match:
            no_improvement += 1
            chain_end_match = (chain_end_match + 1) % N

    return chains