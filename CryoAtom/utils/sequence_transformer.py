"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/apply_sequence_transformer.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
from collections import namedtuple
from typing import List, Dict

from Bio import SeqIO

from CryoAtom.utils.torch_utlis import get_module_device
from CryoAtom.utils.residue_constants import index_to_restype_1
FASTASequence = namedtuple("FASTASequence", field_names=["seq", "rep", "chains"])
import torch
import numpy as np


def remove_non_aa(sequence: str) -> str:
    return "".join([s for s in sequence if s in index_to_restype_1])
def filter_small_sequences(
    fasta_sequences: List[FASTASequence], sequence_names: List[str]
):

    filtered_sequences, filtered_sequence_names = [], []
    for sequence, sequence_name in zip(fasta_sequences, sequence_names):
        filtered_seq = FASTASequence(
            seq=remove_non_aa(sequence.seq), rep=sequence.seq, chains=sequence.chains
        )
        if len(filtered_seq.seq) > 2:
            filtered_sequences.append(filtered_seq)
            filtered_sequence_names.append(sequence_name)
    return filtered_sequences, sequence_names
def crop_long_chain(chain: FASTASequence, max_chain_length=1024):
    length = len(chain.seq)
    i = 0
    chain_crops = []
    chain_starts = []
    while True:
        if i < length - max_chain_length:
            chain_crops.append(chain.seq[i : i + max_chain_length])
            chain_starts.append(i)
            i += max_chain_length // 2
        else:
            chain_crops.append(chain.seq[i:])
            chain_starts.append(i)
            break
    return chain_crops, chain_starts


def crop_long_chains(
    sequences: List[FASTASequence], seq_names: List[str], max_chain_length=1024
):
    new_sequences, chain_ids = [], []
    old_to_new_sequence = {}
    j = 0
    for sequence, seq_name in zip(sequences, seq_names):
        old_to_new_sequence[seq_name] = {}
        old_to_new_sequence[seq_name]["full_seq_len"] = len(sequence.seq)
        if len(sequence.seq) > max_chain_length:
            cropped_chain_list, chain_starts = crop_long_chain(
                sequence, max_chain_length=max_chain_length
            )
            chain_ids += [seq_name + f"_{i}" for i in range(len(cropped_chain_list))]
            new_sequences += cropped_chain_list
            old_to_new_sequence[seq_name]["mapping"] = [
                k + j for k in range(len(cropped_chain_list))
            ]
            old_to_new_sequence[seq_name]["multi_part"] = True
            old_to_new_sequence[seq_name]["chain_starts"] = chain_starts
            j += len(cropped_chain_list)
        else:
            chain_ids += [seq_name + "_0"]
            new_sequences += [sequence.seq]
            old_to_new_sequence[seq_name]["mapping"] = [j]
            old_to_new_sequence[seq_name]["multi_part"] = False
            j += 1
    return new_sequences, chain_ids, old_to_new_sequence
def empty_transformer_results(
    seq_name: str,
    seq_len: int,
    emb_dim: int = 1280,
    device: str = "cpu",
    repr_layers: List = [32, 33],
):
    results = {}
    results["label"] = seq_name
    results["representations"], results["mean_representations"] = {}, {}
    for layer in repr_layers:
        results["representations"][layer] = torch.zeros(seq_len, emb_dim).to(device)
        results["mean_representations"][layer] = torch.zeros(emb_dim).to(device)
    results["contacts"] = torch.zeros(seq_len, seq_len).to(device)
    return results
def process_transformer_result(
    result: Dict,
    str_length: int,
    batch_idx: int = 0,
    seq_name: str = None,
):
    processed_result = {}
    if seq_name is not None:
        processed_result["label"] = seq_name

    processed_result["representations"] = {
        layer: t[batch_idx, 1 : str_length + 1].cpu().clone()
        for layer, t in result["representations"].items()
    }
    processed_result["mean_representations"] = {
        layer: t[batch_idx, 1 : str_length + 1].cpu().mean(0).clone()
        for layer, t in result["representations"].items()
    }
    processed_result["contacts"] = (
        result["contacts"][batch_idx, :str_length].cpu().clone()
    )
    processed_result["str_length"] = str_length
    return processed_result
def collate_sequence_results(
    seq_name: str,
    batch_results: List,
    sequence_mapping: Dict,
    repr_layers: List = [32, 33],
):
    full_result = empty_transformer_results(
        seq_name, sequence_mapping["full_seq_len"], repr_layers=repr_layers
    )
    sequence_results = [batch_results[i] for i in sequence_mapping["mapping"]]
    representation_counts = torch.zeros_like(
        full_result["representations"][repr_layers[0]]
    )
    contacts_counts = torch.zeros_like(full_result["contacts"])
    for result, chain_start in zip(sequence_results, sequence_mapping["chain_starts"]):
        str_length = result["str_length"]
        for layer in repr_layers:
            full_result["representations"][layer][
                chain_start : chain_start + str_length
            ] += result["representations"][layer]
            representation_counts[chain_start : chain_start + str_length] += 1
        full_result["contacts"][
            chain_start : chain_start + str_length,
            chain_start : chain_start + str_length,
        ] += result["contacts"]
        contacts_counts[
            chain_start : chain_start + str_length,
            chain_start : chain_start + str_length,
        ] += 1
    for layer in repr_layers:
        full_result["representations"][layer] /= representation_counts + 1e-6
        full_result["mean_representations"][layer] = (
            full_result["representations"][layer].mean(0).clone()
        )
    full_result["contacts"] /= contacts_counts + 1e-6
    return full_result
@torch.no_grad()
def run_transformer_on_fasta(
    model,
    batch_converter,
    raw_chains,
    sequence_names,
    repr_layers=[32, 33],
    device=None,
    max_chain_length=1000,
):
    if device is None:
        device = get_module_device(model)
    # TODO make this more efficient with proper batching
    chains, sequence_names = filter_small_sequences(raw_chains, sequence_names)
    chains, updated_sequence_names, old_to_new_sequence = crop_long_chains(
        chains,
        sequence_names,
        max_chain_length=max_chain_length,
    )

    batch_results = []

    for seq_name, seq in zip(updated_sequence_names, chains):
        _, batch_strs, batch_tokens = batch_converter([(seq_name, seq)])
        batch_tokens = batch_tokens.to(device)
        batch_results.append(
            process_transformer_result(
                model(batch_tokens, repr_layers=repr_layers, return_contacts=True),
                len(batch_strs[0]),
            ),
        )

    full_result = {}
    for old_seq_name in old_to_new_sequence:
        if not old_to_new_sequence[old_seq_name]["multi_part"]:
            result = batch_results[old_to_new_sequence[old_seq_name]["mapping"][0]]
            result["label"] = old_seq_name
            full_result[old_seq_name] = result
        else:
            full_result[old_seq_name] = collate_sequence_results(
                old_seq_name,
                batch_results,
                old_to_new_sequence[old_seq_name],
                repr_layers=repr_layers,
            )
    return full_result

def get_lm_embeddings_from_fasta(
    lang_model, batch_converter, fasta, max_chain_length=1000
):
    sequences = load_sequence_from_fasta(fasta)
    sequences = [FASTASequence(seq, "", "A") for seq in sequences]
    seq_names = [str(x) for x in range(len(sequences))]
    result = run_transformer_on_fasta(
        lang_model,
        batch_converter,
        sequences,
        seq_names,
        repr_layers=[33],
        max_chain_length=max_chain_length,
    )
    lm_embeddings = np.concatenate(
        [result[s]["representations"][33].cpu().numpy() for s in seq_names],
        axis=0,
    )
    return lm_embeddings
# lang_model,alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# batch_convert = alphabet.get_batch_converter()
# device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
# lang_model = lang_model.to(device)
# a=get_lm_embeddings_from_protein(lang_model,batch_convert,'7u67.cif')
# print(len(''.join(load_sequence('7u67.cif'))))
# print(len(load_sequence('7u67.cif')))
def load_sequence_from_fasta(fasta_file:str):
    fasta = SeqIO.parse(fasta_file,format='fasta')
    seq_list = []
    seq_len = 0
    for seq_record in fasta:
        temp = str(seq_record.seq)
        seq_list.append(temp)
        seq_len += len(temp)
    seq_list = '|||'.join(seq_list)
    return seq_list,seq_len