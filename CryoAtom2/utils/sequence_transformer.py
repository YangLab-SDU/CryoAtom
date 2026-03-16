"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/apply_sequence_transformer.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import random
import sys
if __name__ == "__main__":
    sys.path.insert(0, "/home/yjygroup/cryo-EM/subq/CryFold_train/")
from collections import namedtuple
from typing import List, Dict

from Bio import SeqIO

from CryoAtom2.utils.torch_utlis import get_module_device
from CryoAtom2.utils.residue_constants import index_to_restype_1
FASTASequence = namedtuple("FASTASequence", field_names=["seq", "rep", "chains"])
import torch
import numpy as np

restype_1_alpha = {
    "A":"A",
    "R":"R",
    "N":"N",
    "D":"D",
    "C":"C",
    "Q":"Q",
    "E":"E",
    "G":"G",
    "H":"H",
    "I":"I",
    "L":"L",
    "K":"K",
    "M":"M",
    "F":"F",
    "P":"P",
    "S":"S",
    "T":"T",
    "W":"W",
    "Y":"Y",
    "V":"V",
    "x":"A",
    "y":"C",
    "z":"G",
    "t":"T",
    "a":"A",
    "c":"C",
    "g":"G",
    "u":"U",
    "U":"U",

}
def remove_non_aa(sequence: str) -> str:
    return "".join([restype_1_alpha[s] for s in sequence if s in index_to_restype_1])
def filter_small_sequences(
    fasta_sequences: List[FASTASequence], sequence_names: List[str]
):

    filtered_sequences, filtered_sequence_names = [], []
    for sequence, sequence_name in zip(fasta_sequences, sequence_names):
        filtered_seq = FASTASequence(
            seq=remove_non_aa(sequence.seq), rep=sequence.rep, chains=sequence.chains
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
            chain_crops.append(FASTASequence(chain.seq[i : i + max_chain_length],chain.rep,chain.chains))
            chain_starts.append(i)
            i += max_chain_length // 2
        else:
            chain_crops.append(FASTASequence(chain.seq[i:],chain.rep,chain.chains))
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
            new_sequences += [sequence]
            old_to_new_sequence[seq_name]["mapping"] = [j]
            old_to_new_sequence[seq_name]["multi_part"] = False
            j += 1
    return new_sequences, chain_ids, old_to_new_sequence
def empty_transformer_results(
    seq_name: str,
    seq_len: int,
    emb_dim: int = 1280,
    device: str = "cpu",
):
    results = {}
    results["label"] = seq_name
    results["representations"], results["mean_representations"] = torch.zeros(seq_len, emb_dim).to(device), torch.zeros(emb_dim).to(device)
    return results
def process_transformer_result(
    result: Dict,
    str_length: int,
    batch_idx: int = 0,
    seq_name: str = None,
    emb_dim: int = 1280
):
    processed_result = {}
    if seq_name is not None:
        processed_result["label"] = seq_name
    processed_result["representations"] = {}
    processed_result["mean_representations"] = {}
    for layer, t in result["representations"].items():
        processed_result["representations"] = torch.zeros((t.shape[1]-2,emb_dim),dtype=t.dtype,device=torch.device('cpu'))
        processed_result["mean_representations"] = processed_result["representations"].mean(0).clone()
        processed_result["representations"][...,:t.shape[-1]] = t[batch_idx, 1 : str_length + 1].cpu().clone()
        processed_result["mean_representations"][...,:t.shape[-1]] = t[batch_idx, 1 : str_length + 1].cpu().mean(0).clone()
    processed_result["str_length"] = str_length
    return processed_result
def collate_sequence_results(
    seq_name: str,
    batch_results: List,
    sequence_mapping: Dict,
    emb_dim:int,
):
    full_result = empty_transformer_results(
        seq_name, sequence_mapping["full_seq_len"],emb_dim=emb_dim
    )
    sequence_results = [batch_results[i] for i in sequence_mapping["mapping"]]
    representation_counts = torch.zeros_like(
        full_result["representations"]
    )
    for result, chain_start in zip(sequence_results, sequence_mapping["chain_starts"]):
        str_length = result["str_length"]
        full_result["representations"][
            chain_start : chain_start + str_length
        ] += result["representations"]
        representation_counts[chain_start : chain_start + str_length] += 1
    full_result["representations"] /= representation_counts + 1e-6
    full_result["mean_representations"] = (
        full_result["representations"].mean(0).clone()
    )
    return full_result
@torch.no_grad()
def run_transformer_on_fasta(
    model,
    batch_converter,
    repr_layers,
    emb_dim,
    raw_chains,
    sequence_names,
    device=None,
    max_chain_length=1000,
):
    if device is None:
        device = get_module_device(model['protein'])
    # TODO make this more efficient with proper batching
    # chains, sequence_names = filter_small_sequences(raw_chains, sequence_names)
    chains, updated_sequence_names, old_to_new_sequence = crop_long_chains(
        raw_chains,
        sequence_names,
        max_chain_length=max_chain_length,
    )

    batch_results = []

    for seq_name, seq in zip(updated_sequence_names, chains):
        _, batch_strs, batch_tokens = batch_converter[seq.rep]([(seq_name, seq.seq)])
        batch_tokens = batch_tokens.to(device)
        batch_results.append(
            process_transformer_result(
                model[seq.rep](batch_tokens, repr_layers=repr_layers[seq.rep]),
                len(batch_strs[0]),emb_dim=emb_dim
            )
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
                emb_dim=emb_dim,
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
        max_chain_length=max_chain_length,
    )
    lm_embeddings = np.concatenate(
        [result[s]["representations"][33].cpu().numpy() for s in seq_names],
        axis=0,
    )
    return lm_embeddings

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