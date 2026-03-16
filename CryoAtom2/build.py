from pathlib import Path
import os
pkg_dir = os.path.abspath(str(Path(__file__).parent.parent))
import sys
sys.path.insert(0, pkg_dir)
import argparse
import json
import shutil
import torch
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser, is_nucleic
import numpy as np
from CryoAtom2.utils.fasta_utils import read_fasta
from CryoAtom2.RUNet.inference import infer as RUNet_infer
from CryoAtom2.CryoNet.inference import infer as CryoNet_infer
from CryoAtom2.CryoNet.inference_no_seq import infer as CryoNet_infer_no_seq
from CryoAtom2.utils.fasta_utils import is_valid_fasta_ending
from CryoAtom2.utils.misc_utils import filter_useless_warnings, Args
from CryoAtom2.utils.hmmer_search import hmmer_search
from CryoAtom2.utils.torch_utlis import get_device_names
import time

def filter_chains(input_pdb_file, output_pdb_file, bfactor_threshold=50):
    if input_pdb_file.split(".")[-1][:3] == "pdb":
        parser = PDBParser(QUIET=True)
        io = PDB.PDBIO()
    elif input_pdb_file.split(".")[-1][:3] == "cif":
        parser = MMCIFParser(QUIET=True)
        io = PDB.MMCIFIO()
    else:
        raise RuntimeError("Unknown type for structure file:", input_pdb_file[-3:])
    # read pdb file
    structure = parser.get_structure("protein", input_pdb_file)


    # set bfactor filter
    class BFactorFilter(PDB.Select):
        def accept_chain(self, chain):
            b_factors = [atom.get_bfactor() for atom in chain.get_atoms()]
            is_na = [is_nucleic(res) for res in chain.get_residues()]
            if any(is_na):
                return True
            if not b_factors:
                return False
            average_bfactor = np.mean(b_factors)
            return average_bfactor > bfactor_threshold

    io.set_structure(structure)
    io.save(output_pdb_file, select=BFactorFilter())
    return output_pdb_file
def add_args(parser):
    main_args = parser.add_argument_group(
        "Main arguments",
        description="If you are not very familiar with this software, you can just fill in this part of the parameters. These are also the main parameters of this software."
    )
    main_args.add_argument(
        '--map-path',
        "-v",
        "--v",
        help="input cryo-em density map",
        type=str,
        required=True
    )
    main_args.add_argument(
        '--protein-sequence-path',
        "-ps",
        "--ps",
        help="input protein sequence fasta file",
        type=str
    )
    main_args.add_argument(
        '--rna-sequence-path',
        "-rs",
        "--rs",
        help="input RNA sequence fasta file",
        type=str
    )
    main_args.add_argument(
        '--dna-sequence-path',
        "-ds",
        "--ds",
        help="input DNA sequence fasta file",
        type=str
    )
    main_args.add_argument(
        "--output-dir",
        "-o",
        "--o",
        help="output directory",
        type=str,
        default="output",
    )
    main_args.add_argument(
        "--device",
        "-d",
        "--d",
        help="compute device, pick one of {cpu, gpu_number}. "
        "Default set to find an available GPU.",
        type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    additional_args = parser.add_argument_group(
        "Additional arguments",
        description="Adjusting these additional parameters can help you build protein-NA models more efficiently."
    )
    additional_args.add_argument(
        "--mask-path",
        "-m",
        "--m",
        help="Providing the mask map corresponding to the original density map can mask out the redundant regions in the original density map.",
        type=str,
    )
    additional_args.add_argument(
        '--protein-fasta-path',
        "-pf",
        "--pf",
        help="Path to the protein FASTA database containing all sequences",
        type=str,
    )
    additional_args.add_argument(
        '--na-fasta-path',
        "-nf",
        "--nf",
        help="Path to the nucleic acid FASTA database containing all sequences",
        type=str,
    )
    additional_args.add_argument(
        '--refine-backbone-path',
        "-r",
        "--r",
        help="Protein-NA backbone atom (C-alpha and P) file for refinement and identification.",
        type=str,
    )
    additional_args.add_argument(
        "--keep-intermediate-results",
        '-k',
        '--k',
        action="store_true",
        help="Keep intermediate results, ie see_alpha_output and CryoNet_round_x"
    )

    additional_args.add_argument(
        "--config-path", "-c", "--c", help="Provide an additional parameter file path. It is recommended to have a detailed understanding of this software before using this parameter, otherwise use the default parameter values.", type=str,
    )

    return parser

@torch.no_grad()
def main(parsed_args):
    start_time = time.time()
    filter_useless_warnings()

    if parsed_args.config_path:
        with open(parsed_args.config_path, "r") as f:
            config = json.load(f)
    else:
        with open(pkg_dir + "/CryoAtom2/config.json", "r") as f:
            config = json.load(f)

    RUNet_model_logdir = pkg_dir + "/CryoAtom2/checkpoint/RUNet.pth"
    RUNet_getp_pth = pkg_dir+"/CryoAtom2/utils/getp"
    sequence_path_ls = [parsed_args.protein_sequence_path,parsed_args.rna_sequence_path,parsed_args.dna_sequence_path]
    has_sequences_flag = not all([ss is None for ss in sequence_path_ls])
    if has_sequences_flag:
        CryoNet_model_logdir = pkg_dir + "/CryoAtom2/checkpoint/CryoNet.pth"
    else:
        CryoNet_model_logdir = pkg_dir + "/CryoAtom2/checkpoint/CryoNet_no_seq.pth"

    parsed_args.output_dir = os.path.normpath(parsed_args.output_dir)
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    torch.no_grad()

    print("-------------------------------------------------------------------- CryoAtom2 --------------------------------------------------------------------")
    print("By Baoquan Su, Yang lab.")
    for ss in sequence_path_ls:
        if ss is not None:
            if not is_valid_fasta_ending(ss):
                raise RuntimeError(f"File {ss} is not a fasta file format.")

            _ = read_fasta(ss)


    if parsed_args.refine_backbone_path:
        ca_cif_path = parsed_args.refine_backbone_path
        config["CryoNet_args"]["num_rounds"] = 1
        config["CryoNet_args"]["mask_threshold"] = 0
        config["CryoNet_args"]["is_refine"] = False
    else:
        # Run C-alpha inference ----------------------------------------------------------------------------------------
        print("--------------------- CryoAtom2 Stage1 (Predict C-alpha or P atoms by RU-Net) ---------------------")

        RUNet_args = Args(config["RUNet_args"])
        RUNet_args.log_dir = RUNet_model_logdir
        RUNet_args.getp_path = RUNet_getp_pth
        RUNet_args.map_path = parsed_args.map_path
        RUNet_args.output_path = os.path.join(parsed_args.output_dir, "see_alpha_output")
        RUNet_args.device = parsed_args.device
        RUNet_args.mask_path = parsed_args.mask_path
        config["CryoNet_args"]["is_refine"] = False
        ca_cif_path = RUNet_infer(RUNet_args)


    current_ca_cif_path = ca_cif_path
    total_CryoNet_rounds = config["CryoNet_args"]["num_rounds"]
    for i in range(total_CryoNet_rounds):
        print(f"------------------ CryoAtom2 Stage2 (Build all-atom structure by Cryo-Net), round {i + 1} / {total_CryoNet_rounds} ------------------")

        current_output_dir = os.path.join(
            parsed_args.output_dir, f"CryoNet_round_{i + 1}"
        )
        os.makedirs(current_output_dir, exist_ok=True)

        CryoNet_args = Args(config["CryoNet_args"])
        if CryoNet_args.seq_attention_batch_size <= 0 :
            CryoNet_args.seq_attention_batch_size = CryoNet_args.crop_length
        CryoNet_args.map_path = parsed_args.map_path
        CryoNet_args.protein_fasta = parsed_args.protein_sequence_path
        CryoNet_args.rna_fasta = parsed_args.rna_sequence_path
        CryoNet_args.dna_fasta = parsed_args.dna_sequence_path
        CryoNet_args.struct = current_ca_cif_path
        CryoNet_args.output_dir = current_output_dir
        CryoNet_args.device = parsed_args.device
        CryoNet_args.aggressive_pruning = True
        CryoNet_args.model_dir = CryoNet_model_logdir
        if (i+1) >= total_CryoNet_rounds:
            CryoNet_args.end_flag = True
        else:
            CryoNet_args.end_flag = False
        if has_sequences_flag:
            CryoNet_output = CryoNet_infer(CryoNet_args)

        else:
            CryoNet_output = CryoNet_infer_no_seq(CryoNet_args)
        current_ca_cif_path = os.path.join(
            current_output_dir, "model_net.cif"
        )
    if has_sequences_flag:
        pruned_file_src = CryoNet_output.replace("model_net.cif", "model_prune.cif")
        raw_file_src = CryoNet_output.replace("model_net.cif", "model_fix.cif")

        name = os.path.basename(parsed_args.output_dir)
        pruned_file_dst = os.path.join(parsed_args.output_dir, f"{name}.cif")
        raw_file_dst = os.path.join(parsed_args.output_dir, f"{name}_raw.cif")

        os.replace(pruned_file_src, pruned_file_dst)
        os.replace(raw_file_src, raw_file_dst)
        filter_chains(pruned_file_dst,pruned_file_dst,bfactor_threshold=CryoNet_args.filter_threshold)
        if CryoNet_args.raw_filter:
            filter_chains(raw_file_dst, raw_file_dst, bfactor_threshold=CryoNet_args.filter_threshold)
    if parsed_args.protein_fasta_path is not None or parsed_args.na_fasta_path is not None:
        print(f"---------------------------- hmmer_search ----------------------------")
        new_hits_evalue = config["HMM_search"]["Evalue"]
        new_hits_excel_prot,new_hits_seq_prot,new_hits_excel_na,new_hits_seq_na="","","",""
        if parsed_args.protein_fasta_path is not None:
            new_hits_excel_prot,new_hits_seq_prot = hmmer_search(input_dir = parsed_args.output_dir,fasta_database=parsed_args.protein_fasta_path,raw_fasta=parsed_args.protein_sequence_path,threshold=config["HMM_search"]["confidence_threshold"],cpus=config["HMM_search"]["cpus"],Evalue=new_hits_evalue if has_sequences_flag else min(new_hits_evalue*100,10),total_round=total_CryoNet_rounds,alphabet="prot")
        if parsed_args.na_fasta_path is not None:
            new_hits_excel_na,new_hits_seq_na = hmmer_search(input_dir = parsed_args.output_dir,fasta_database=parsed_args.na_fasta_path,raw_fasta=parsed_args.rna_sequence_path,threshold=config["HMM_search"]["confidence_threshold"],cpus=config["HMM_search"]["cpus"],Evalue=new_hits_evalue if has_sequences_flag else min(new_hits_evalue*100,10),total_round=total_CryoNet_rounds,alphabet="na")
        if parsed_args.protein_fasta_path is not None and parsed_args.na_fasta_path is not None:
            print_hmmsearch_str = " and "
        else:
            print_hmmsearch_str = ""
        print(f'The new sequence results found cryo-EM density map, in addition to the input sequence, have been saved to {new_hits_excel_prot}{print_hmmsearch_str}{new_hits_excel_na}.')
        print(f"The additional sequences have been integrated into {new_hits_seq_prot}{print_hmmsearch_str}{new_hits_seq_na}, and it is recommended to rerun CryoAtom2 using these new sequences.")
    if not has_sequences_flag:
        raw_file_src = CryoNet_output
        hmm_profiles_src = os.path.join(os.path.dirname(CryoNet_output), "net_hmm_profiles")

        name = os.path.basename(parsed_args.output_dir)
        raw_file_dst = os.path.join(parsed_args.output_dir, f"{name}_raw.cif")
        pruned_file_dst = os.path.join(parsed_args.output_dir, f"{name}.cif")
        hmm_profiles_dst = os.path.join(parsed_args.output_dir, "hmm_profiles")

        os.replace(raw_file_src, raw_file_dst)
        filter_chains(raw_file_dst,pruned_file_dst,bfactor_threshold=config["HMM_search"]["confidence_threshold"])
        shutil.rmtree(hmm_profiles_dst, ignore_errors=True)
        os.replace(hmm_profiles_src, hmm_profiles_dst)

    if not parsed_args.keep_intermediate_results:
        if parsed_args.refine_backbone_path:
            pass
        else:
            shutil.rmtree(RUNet_args.output_path, ignore_errors=True)
        for i in range(total_CryoNet_rounds):
            shutil.rmtree(
                os.path.join(
                    parsed_args.output_dir, f"CryoNet_round_{i + 1}"
                )
            )
    start_time = time.time()-start_time
    # with open(os.path.join(parsed_args.output_dir,'running_time.log'),'w') as f11:
    #     f11.write(str(int(start_time))+'s')

    print("-" * 70)
    print("CryoAtom2 has completed the construction of the all-atom structure model.")
    print("-" * 70)
    if has_sequences_flag:
        print(f"You can find the final model filtered by the fasta sequence at: {pruned_file_dst}.\n")
    else:
        print(f'You can find the final model without sequence correction at: {pruned_file_dst}.\n')
        print(f'The model contains many uncertain regions. It is recommended to use hmm_search to search for sequences and use it as input to iteratively run CryoAtom2.\n')
    print("The confidence scores of the model predictions are saved in the bfactor field of the mmcif file.")
    print("-" * 70)
    print(f"If you want to see more of the built regions in the map, you can go to: {raw_file_dst}")
    print("However, it often has many untreated unknown regions\n(which may be caused by the noise in the density map itself).")
    print("-" * 70)
    print("done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)
