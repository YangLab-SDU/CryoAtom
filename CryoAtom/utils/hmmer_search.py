import pyhmmer
from scipy.spatial import cKDTree
import argparse
import pandas as pd
import os
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import DisorderedAtom
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
from CryoAtom.utils.save_pdb_utils import number_to_chain_str
import tqdm
def load_cas_from_structure(stu_fn, all_structs=False, quiet=True):

    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])

    structure = parser.get_structure("structure", stu_fn)
    if not quiet and len(structure) > 1:
        print(f"WARNING: {len(structure)} structures found in model file: {stu_fn}")

    if not all_structs:
        structure = [structure[0]]

    ca_coords = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")

        for i, a in enumerate(model.get_atoms()):
            if a.get_name() == "CA":
                if isinstance(a, DisorderedAtom):
                    ca_coords.append(
                        a.disordered_get_list()[0].get_vector().get_array()
                    )
                else:
                    ca_coords.append(a.get_vector().get_array())

    return np.array(ca_coords)
def load_ca_score_from_structure(stu_fn, all_structs=False, quiet=True):

    if stu_fn.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=quiet)
    elif stu_fn.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=quiet)
    else:
        raise RuntimeError("Unknown type for structure file:", stu_fn[-3:])

    structure = parser.get_structure("structure", stu_fn)
    if not quiet and len(structure) > 1:
        print(f"WARNING: {len(structure)} structures found in model file: {stu_fn}")

    if not all_structs:
        structure = [structure[0]]

    ca_coords = []
    conf_scores = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for chain in model:
            ca_coord = []
            conf_score = []
            for residue in chain:
                if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                    ca_coord.append(residue['CA'].get_coord())
                    conf_score.append(residue['CA'].get_bfactor())
            ca_coords.append(np.array(ca_coord))
            conf_scores.append(np.mean(np.array(conf_score)))

    return ca_coords,conf_scores

def hmmer_search(input_dir:str,fasta_database:str,raw_fasta=None,output_dir=None,threshold:int=50,cpus:int=4,Evalue=10,total_round:int=3):
    if output_dir is None:
        output_dir = input_dir
    hits_csv = {
        "target_name": [],
        "query_name": [],
        "query_len":[],
        "E-value": [],
        "score": [],
        "bias": [],
        "accession": [],
        "description": [],
    }
    with pyhmmer.easel.SequenceFile(fasta_database,alphabet=pyhmmer.easel.Alphabet.amino(),digital=True) as seq_file:
        sequences = seq_file.read_block()
    base_dir = input_dir[:-1] if input_dir.endswith('/') else input_dir
    model_prune_path = base_dir + f'/{os.path.basename(base_dir)}.cif'
    model_net_path = base_dir + f'/CryNet_round_{total_round}/model_net.cif'
    net_hmm_dir = base_dir + f'/CryNet_round_{total_round}/net_hmm_profiles/'
    if raw_fasta:
        prune_cas = load_cas_from_structure(model_prune_path)
        prune_cas_tree = cKDTree(prune_cas)
    net_cas,net_scores = load_ca_score_from_structure(model_net_path)
    hmms = []
    for ii in range(len(net_cas)):
        if net_scores[ii] < threshold:
            continue
        if raw_fasta:
            dist,_ = prune_cas_tree.query(net_cas[ii], k=1)
            if np.sum(dist>1)/len(dist) <0.5:
                continue
        chain_name = number_to_chain_str(ii)
        query_len = len(net_cas[ii])
        hmms.append(((chain_name,query_len),pyhmmer.plan7.HMMFile(net_hmm_dir+f'{chain_name}.hmm').read()))
    all_hits = pyhmmer.hmmer.hmmsearch(
        [hmm for name, hmm in hmms],
        sequences,
        cpus=cpus
    )
    for (hits,name) in tqdm.tqdm(zip(all_hits,[name for name,hmm in hmms])):
        for hit in hits:
            if hit.evalue < Evalue:
                try:
                    hits_csv["target_name"].append(hit.name.decode("utf-8"))
                    hits_csv["query_name"].append(name[0])
                    hits_csv["query_len"].append(name[1])
                    hits_csv["accession"].append(hit.accession.decode("utf-8") if hit.accession else "")
                    hits_csv["E-value"].append(hit.evalue)
                    hits_csv["score"].append(hit.score)
                    hits_csv["bias"].append(hit.bias)
                    hits_csv["description"].append(hit.description.decode("utf-8"))
                except:
                    pass
        try:
            msa = hits.to_msa(pyhmmer.easel.Alphabet.amino())
            with open(os.path.join(net_hmm_dir, f"{name[0]}.a2m"), "wb") as f:
                msa.write(f, "a2m")
        except:
            pass
    with pd.ExcelWriter(os.path.join(output_dir, "new_hits.xlsx"),engine='openpyxl',mode='w') as writer:
        hits_df = pd.DataFrame(hits_csv)
        hits_df.sort_values(by=["E-value"], inplace=True)
        try:
            hits_df.to_excel(writer,sheet_name='all_hits',index=False)
        except:
            hits_df.to_csv(os.path.join(output_dir, "all_hits.csv"), index=False)
        min_evalue_indices = hits_df.groupby('target_name')['E-value'].idxmin()
        filtered_df = hits_df.loc[min_evalue_indices]
        min_evalue_indices = filtered_df.groupby('query_name')['E-value'].idxmin()
        filtered_df = filtered_df.loc[min_evalue_indices]
        find_new_seq_name = list(filtered_df['target_name'])
        filtered_df.sort_values(by=["E-value"], inplace=True)
        try:
            filtered_df.to_excel(writer,sheet_name='best_hits',index=False)
        except:
            filtered_df.to_csv(os.path.join(output_dir, "best_hits.csv"), index=False)
    alphabet = pyhmmer.easel.Alphabet.amino()
    out_seq_list = []
    sequences = list(sequences)
    new_seq_list = {seq.name.decode('utf-8'): seq for seq in sequences}
    if raw_fasta:
        raw_seq_list = SeqIO.parse(raw_fasta, "fasta")
        for sss in raw_seq_list:
            out_seq_list.append(sss)
    new_seq_list = [new_seq_list[cn] for cn in find_new_seq_name]
    for sss2 in new_seq_list:
        out_seq_list.append(SeqRecord(Seq(alphabet.decode(sss2.sequence)),id=sss2.name.decode("utf-8"),description=sss2.description.decode("utf-8")))
    SeqIO.write(out_seq_list, os.path.join(output_dir,f'{os.path.basename(base_dir)}.fasta'), "fasta")
    return os.path.join(output_dir, "new_hits.xlsx"),os.path.join(output_dir,f'{os.path.basename(base_dir)}.fasta')
