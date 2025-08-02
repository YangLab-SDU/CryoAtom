from pathlib import Path
import os
pkg_dir = os.path.abspath(str(Path(__file__).parent.parent))
import sys
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
import argparse
import json
import shutil
import torch
from Bio import PDB
from Bio.PDB import PDBParser, MMCIFParser
import numpy as np
from CryoAtom.utils.fasta_utils import read_fasta
from CryoAtom.Unet.inference import infer as UNet_infer
from CryoAtom.CryNet.inference import infer as CryNet_infer
from CryoAtom.CryNet.inference_no_seq import infer as CryNet_infer_no_seq
from CryoAtom.utils.fasta_utils import is_valid_fasta_ending
from CryoAtom.utils.misc_utils import filter_useless_warnings, Args
from CryoAtom.utils.hmmer_search import hmmer_search
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
        '--sequence-path',
        "-s",
        "--s",
        help="input sequence fasta file",
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
        help="compute device, pick one of {cpu, cuda:number}. "
             "Default set to use cuda.",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    additional_args = parser.add_argument_group(
        "Additional arguments",
        description="Adjusting these additional parameters can help you build protein models more efficiently."
    )
    additional_args.add_argument(
        "--mask-path",
        "-m",
        "--m",
        help="Providing the mask map corresponding to the original density map can mask out the redundant regions in the original density map.",
        type=str,
    )
    additional_args.add_argument(
        '--fasta-path',
        "-f",
        "--f",
        help="The FASTA database containing all sequences",
        type=str,
    )
    additional_args.add_argument(
        '--refine-backbone-path',
        "-r",
        "--r",
        help="Protein backbone atom file for refinement and identification.",
        type=str,
    )
    additional_args.add_argument(
        "--crop-length",
        "-n",
        "--n",
        help="The CryNet takes in 'crop_length' number of residues at a time. It can trade space for time.",
        type=int,
        default=300,
    )
    additional_args.add_argument(
        "--keep-intermediate-results",
        '-k',
        '--k',
        action="store_true",
        help="Keep intermediate results, ie see_alpha_output and CryNet_round_x"
    )

    additional_args.add_argument(
        "--config-path", "-c", "--c", help="Provide an additional parameter file path. It is recommended to have a detailed understanding of this software before using this parameter, otherwise use the default parameter values.", type=str,
    )

    return parser

@torch.no_grad()
def main(parsed_args):
    start_time = time.time()
    filter_useless_warnings()

    parsed_args.device = torch.device(parsed_args.device)
    if parsed_args.config_path:
        with open(parsed_args.config_path, "r") as f:
            config = json.load(f)
    else:
        with open(pkg_dir + "/CryoAtom/config.json", "r") as f:
            config = json.load(f)

    UNet_model_logdir = pkg_dir + "/CryoAtom/checkpoint/SimpleUnet.pth"
    if parsed_args.sequence_path:
        CryNet_model_logdir = pkg_dir + "/CryoAtom/checkpoint/CryNet.pth"
    else:
        CryNet_model_logdir = pkg_dir + "/CryoAtom/checkpoint/CryNet_no_seq.pth"

    parsed_args.output_dir = os.path.normpath(parsed_args.output_dir)
    os.makedirs(parsed_args.output_dir, exist_ok=True)

    torch.no_grad()

    print("---------------------------- CryoAtom -----------------------------")
    print("By Baoquan Su, Yang lab.")
    if parsed_args.sequence_path:
        if not is_valid_fasta_ending(parsed_args.sequence_path):
            raise RuntimeError(f"File {parsed_args.sequence_path} is not a fasta file format.")

        _ = read_fasta(parsed_args.sequence_path)


    if parsed_args.refine_backbone_path:
        ca_cif_path = parsed_args.refine_backbone_path
        config["CryNet_args"]["num_rounds"] = 1
        config["CryNet_args"]["mask_threshold"] = 0
        config["CryNet_args"]["is_refine"] = True
    else:
        # Run C-alpha inference ----------------------------------------------------------------------------------------
        print("--------------------- CryoAtom Stage1 (Predict C-alpha atoms by U-Net) ---------------------")

        UNet_args = Args(config["UNet_Args"])
        UNet_args.log_dir = UNet_model_logdir
        UNet_args.map_path = parsed_args.map_path
        UNet_args.output_path = os.path.join(parsed_args.output_dir, "see_alpha_output")
        UNet_args.device = parsed_args.device
        UNet_args.mask_path = parsed_args.mask_path
        config["CryNet_args"]["is_refine"] = False
        ca_cif_path = UNet_infer(UNet_args)


    current_ca_cif_path = ca_cif_path
    total_CryNet_rounds = config["CryNet_args"]["num_rounds"]
    for i in range(total_CryNet_rounds):
        print(f"------------------ CryoAtom Stage2 (Build Protein Model by Cry-Net), round {i + 1} / {total_CryNet_rounds} ------------------")

        current_output_dir = os.path.join(
            parsed_args.output_dir, f"CryNet_round_{i + 1}"
        )
        os.makedirs(current_output_dir, exist_ok=True)

        CryNet_args = Args(config["CryNet_args"])
        CryNet_args.crop_length = parsed_args.crop_length
        if CryNet_args.seq_attention_batch_size <= 0 :
            CryNet_args.seq_attention_batch_size = CryNet_args.crop_length*2//3
        CryNet_args.map_path = parsed_args.map_path
        CryNet_args.fasta = parsed_args.sequence_path
        CryNet_args.struct = current_ca_cif_path
        CryNet_args.output_dir = current_output_dir
        CryNet_args.device = parsed_args.device
        CryNet_args.aggressive_pruning = True
        CryNet_args.model_dir = CryNet_model_logdir
        if (i+1) >= total_CryNet_rounds:
            CryNet_args.end_flag = True
        else:
            CryNet_args.end_flag = False
        if parsed_args.sequence_path:
            CryNet_output = CryNet_infer(CryNet_args)
            current_ca_cif_path = os.path.join(
                current_output_dir, "model_fix.cif"
            )
        else:
            CryNet_output = CryNet_infer_no_seq(CryNet_args)
            current_ca_cif_path = os.path.join(
                current_output_dir, "model_net.cif"
            )
    if parsed_args.sequence_path:
        pruned_file_src = CryNet_output.replace("model_net.cif", "model_prune.cif")
        raw_file_src = CryNet_output.replace("model_net.cif", "model_fix.cif")

        name = os.path.basename(parsed_args.output_dir)
        pruned_file_dst = os.path.join(parsed_args.output_dir, f"{name}.cif")
        raw_file_dst = os.path.join(parsed_args.output_dir, f"{name}_raw.cif")

        os.replace(pruned_file_src, pruned_file_dst)
        os.replace(raw_file_src, raw_file_dst)
        filter_chains(pruned_file_dst,pruned_file_dst,bfactor_threshold=CryNet_args.filter_threshold)
        if CryNet_args.raw_filter:
            filter_chains(raw_file_dst, raw_file_dst, bfactor_threshold=CryNet_args.filter_threshold)
    if parsed_args.fasta_path:
        print(f"---------------------------- hmmer_search ----------------------------")
        new_hits_evalue = config["HMM_search"]["Evalue"]
        new_hits_excel,new_hits_seq = hmmer_search(input_dir = parsed_args.output_dir,fasta_database=parsed_args.fasta_path,raw_fasta=parsed_args.sequence_path,threshold=config["HMM_search"]["confidence_threshold"],cpus=config["HMM_search"]["cpus"],Evalue=new_hits_evalue if parsed_args.sequence_path else min(new_hits_evalue*100,10),total_round=total_CryNet_rounds)
        print(f'The new sequence results found cryo-EM density map, in addition to the input sequence, have been saved to {new_hits_excel}.')
        print(f"The additional sequences have been integrated into {new_hits_seq}, and it is recommended to rerun CryoAtom using this new sequence.")
    if parsed_args.sequence_path:
        pass
    else:
        raw_file_src = CryNet_output
        hmm_profiles_src = os.path.join(os.path.dirname(CryNet_output), "net_hmm_profiles")

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
            shutil.rmtree(UNet_args.output_path, ignore_errors=True)
        for i in range(total_CryNet_rounds):
            shutil.rmtree(
                os.path.join(
                    parsed_args.output_dir, f"CryNet_round_{i + 1}"
                )
            )
    start_time = time.time()-start_time
    # with open(os.path.join(parsed_args.output_dir,'running_time.log'),'w') as f11:
    #     f11.write(str(int(start_time))+'s')

    print("-" * 70)
    print("CryoAtom has completed the construction of the full-atom protein model.")
    print("-" * 70)
    if parsed_args.sequence_path:
        print(f"You can find the final model filtered by the fasta sequence at: {pruned_file_dst}.\n")
    else:
        print(f'You can find the final model without sequence correction at: {pruned_file_dst}.\n')
        print(f'The model contains many uncertain regions. It is recommended to use hmm_search to search for sequences and use it as input to iteratively run CryoAtom.\n')
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
