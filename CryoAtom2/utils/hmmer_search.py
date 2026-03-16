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
from CryoAtom2.utils.save_pdb_utils import number_to_chain_str
from CryoAtom2.utils.PNAComplex import get_PNAComplex_from_file_path
from CryoAtom2.utils.residue_constants import atom_order
import tqdm

def _to_str(x):
    if x is None:
        return ""
    return x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
def hmmer_search(input_dir:str,fasta_database:str,raw_fasta=None,output_dir=None,threshold:int=50,cpus:int=4,Evalue=10,total_round:int=3,alphabet="prot"):
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
    alphabet_func = {"prot":pyhmmer.easel.Alphabet.amino(),"na":pyhmmer.easel.Alphabet.rna()}
    if alphabet == "prot":
        with pyhmmer.easel.SequenceFile(fasta_database,digital=False) as seq_file:
            sequences = seq_file.read_block()
        with pyhmmer.easel.SequenceFile(fasta_database,alphabet=pyhmmer.easel.Alphabet.amino(),digital=True) as seq_file:
            pp_sequences = seq_file.read_block()
    elif alphabet == "na":
        sequences = []
        pp_sequences = []
        with pyhmmer.easel.SequenceFile(
                fasta_database,
                digital=False
        ) as seq_file:
            for seq in seq_file:
                sequences.append(seq)
                pp_sequences.append(pyhmmer.easel.TextSequence(
                    name=seq.name,  # bytes
                    sequence=_to_str(seq.sequence).upper().replace("T", "U"),
                    description=seq.description  # bytes or None
                ).digitize(pyhmmer.easel.Alphabet.rna()))
    base_dir = input_dir[:-1] if input_dir.endswith('/') else input_dir
    model_prune_path = base_dir + f'/{os.path.basename(base_dir)}.cif'
    model_net_path = base_dir + f'/CryoNet_round_{total_round}/model_net.cif'
    net_hmm_dir = base_dir + f'/CryoNet_round_{total_round}/net_hmm_profiles/'
    if raw_fasta:
        prune_cas = get_PNAComplex_from_file_path(model_prune_path).atomc_positions[:,1]
        prune_cas_tree = cKDTree(prune_cas)
    net_cas = get_PNAComplex_from_file_path(model_net_path)
    prot_mask = net_cas.prot_mask
    net_scores = np.zeros(len(prot_mask))
    net_scores[prot_mask] = net_cas.b_factors[prot_mask,atom_order["CA"]]
    net_scores[~prot_mask] = net_cas.b_factors[~prot_mask,atom_order["P"]]
    chain_names = net_cas.chain_id
    chain_index = net_cas.chain_index
    net_cas = net_cas.atomc_positions[:,1]
    hmms = []
    for ii,chain_name in enumerate(chain_names):
        chain_idx = chain_index == ii
        query_len = np.sum(chain_idx)
        if (np.any(prot_mask[chain_idx]) and alphabet=="na") or (np.any(~prot_mask[chain_idx]) and alphabet=="prot"):
            continue
        if np.mean(net_scores[chain_idx]) < threshold or query_len < 5:
            continue
        if raw_fasta:
            dist,_ = prune_cas_tree.query(net_cas[chain_idx], k=1)
            if np.sum(dist>1)/len(dist) <0.5:
                continue
        hmms.append(((chain_name,query_len),pyhmmer.plan7.HMMFile(net_hmm_dir+f'{chain_name}_{alphabet}.hmm').read()))
    all_hits = pyhmmer.hmmer.hmmsearch(
        [hmm for name, hmm in hmms],
        pp_sequences,
        cpus=cpus
    )
    for (hits,name) in tqdm.tqdm(zip(all_hits,[name for name,hmm in hmms])):
        for hit in hits:
            if hit.evalue < Evalue:
                hits_csv["target_name"].append(hit.name.decode("utf-8"))
                hits_csv["query_name"].append(name[0])
                hits_csv["query_len"].append(name[1])
                hits_csv["accession"].append(hit.accession.decode("utf-8") if hit.accession else "")
                hits_csv["E-value"].append(hit.evalue)
                hits_csv["score"].append(hit.score)
                hits_csv["bias"].append(hit.bias)
                hits_csv["description"].append(_to_str(hit.description))
        try:
            msa = hits.to_msa(alphabet_func[alphabet])
            with open(os.path.join(net_hmm_dir, f"{name[0]}.a2m"), "wb") as f:
                msa.write(f, "a2m")
        except:
            pass
    with pd.ExcelWriter(os.path.join(output_dir, f"new_hits_{alphabet}.xlsx"),engine='openpyxl',mode='w') as writer:
        hits_df = pd.DataFrame(hits_csv)
        hits_df.sort_values(by=["E-value"], inplace=True)
        try:
            hits_df.to_excel(writer,sheet_name=f'all_hits_{alphabet}',index=False)
        except:
            hits_df.to_csv(os.path.join(output_dir, f"all_hits_{alphabet}.csv"), index=False)
        min_evalue_indices = hits_df.groupby('target_name')['E-value'].idxmin()
        filtered_df = hits_df.loc[min_evalue_indices]
        min_evalue_indices = filtered_df.groupby('query_name')['E-value'].idxmin()
        filtered_df = filtered_df.loc[min_evalue_indices]
        find_new_seq_name = list(filtered_df['target_name'])
        filtered_df.sort_values(by=["E-value"], inplace=True)
        try:
            filtered_df.to_excel(writer,sheet_name=f'best_hits_{alphabet}',index=False)
        except:
            filtered_df.to_csv(os.path.join(output_dir, f"best_hits_{alphabet}.csv"), index=False)
    out_seq_list = []
    sequences = list(sequences)
    new_seq_list = {seq.name.decode('utf-8'): seq for seq in sequences}
    if raw_fasta:
        raw_seq_list = SeqIO.parse(raw_fasta, "fasta")
        for sss in raw_seq_list:
            out_seq_list.append(sss)
    new_seq_list = [new_seq_list[cn] for cn in find_new_seq_name]
    for sss2 in new_seq_list:
        out_seq_list.append(SeqRecord(Seq(sss2.sequence),id=sss2.name.decode("utf-8"),description=_to_str(sss2.description)))
    SeqIO.write(out_seq_list, os.path.join(output_dir,f'{os.path.basename(base_dir)}_{alphabet}.fasta'), "fasta")
    return os.path.join(output_dir, f"new_hits_{alphabet}.xlsx"),os.path.join(output_dir,f'{os.path.basename(base_dir)}_{alphabet}.fasta')
