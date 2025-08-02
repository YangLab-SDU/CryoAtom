import shutil
from pathlib import Path
import os
pkg_dir = os.path.abspath(str(Path(__file__).parent.parent))
import sys
if pkg_dir not in sys.path:
    sys.path.insert(0, pkg_dir)
import argparse
from scipy.spatial import cKDTree
from CryoAtom.utils.protein import get_protein_from_file_path,slice_protein
import numpy as np
from CryoAtom.utils.save_pdb_utils import chain_atom14_to_cif
from collections import defaultdict

def Remove_res(cif_path1,cif_path2,dist_threshold=3):
    if cif_path1 is None:
        protein2 = get_protein_from_file_path(cif_path2)
        chain_atom14_to_cif([protein2.aatype],[protein2.atom14_positions],[protein2.atom14_mask],cif_path2,[protein2.b_factors])
    else:
        protein1 = get_protein_from_file_path(cif_path1)
        protein2 = get_protein_from_file_path(cif_path2)
        position1 = protein1.atom14_positions[:,1]
        position2 = protein2.atom14_positions[:,1]
        tree1 = cKDTree(position1)
        dist,_ = tree1.query(position2,k=1)
        protein2 = slice_protein(protein2,dist>dist_threshold)
        chain_atom14_to_cif([protein2.aatype],[protein2.atom14_positions],[protein2.atom14_mask],cif_path2,[protein2.b_factors])
    return cif_path2
def AlignWithTP(cif_path1,cif_path2,save_pth,AlignScript:str):
    cif_name = os.path.basename(cif_path1)
    lines = os.popen(f'{AlignScript} {cif_path1} {cif_path2} -mm 1 -ter 0 -o {os.path.join(save_pth,cif_name[:-4])}').read().strip().splitlines()
    TMscore1, TMscore2 = 0,0
    tmscores = [r for r in lines if r.startswith("TM-score= ")]
    if len(tmscores) > 1:
        TMscore1 = float(tmscores[0][10:17])
        TMscore2 = float(tmscores[1][10:17])
    save_cif_pth = os.path.join(save_pth,cif_name)
    protein1 = get_protein_from_file_path(save_cif_pth)
    protein2 = get_protein_from_file_path(cif_path2)
    position1 = protein1.atom14_positions[:,1]
    position2 = protein2.atom14_positions[:,1]
    pos_tree2 = cKDTree(position2)
    dist,_ = pos_tree2.query(position1,k=1)
    return TMscore1,np.sum(dist<3)/np.sqrt(len(dist)),save_cif_pth
def AlignWithTPs(CF_raw_pth,template_dir:str,save_pth:str,AlignScript:str,tm_threshold=1):
    template_dir = Path(template_dir)
    save_pth = Path(save_pth)
    save_pth2 = os.path.join(save_pth,'AllStep')
    save_temp_dir = os.path.join(save_pth,'temp')
    os.makedirs(save_pth2, exist_ok=True)
    os.makedirs(save_temp_dir,exist_ok=True)
    name = os.path.basename(save_pth)
    protein2 = get_protein_from_file_path(CF_raw_pth)
    out_raw_pth = os.path.join(save_pth,f"{name}_raw.cif")
    chain_atom14_to_cif([protein2.aatype], [protein2.atom14_positions], [protein2.atom14_mask], out_raw_pth,
                        [protein2.b_factors])
    template_list = os.listdir(template_dir)
    template_list = [r for r in template_list if r.endswith('.cif') or r.endswith('.pdb')]
    template_list = [r for r in template_list if len(get_protein_from_file_path(os.path.join(template_dir,r)).aatype)>30]
    template_list_used = defaultdict(int)
    for ii in range(10000):
        tmscore_arr = []
        template_align_list = []
        for template_cif in template_list:
            if tm_threshold >= 1 and template_list_used[template_cif]>0.5:
                tmscore_arr.append((-1,-1))
                template_align_list.append(template_cif)
            else:
                tmscore1,tmscore2,template_align_cif = AlignWithTP(os.path.join(template_dir,template_cif),out_raw_pth,save_pth=save_temp_dir,AlignScript=AlignScript)
                tmscore_arr.append((tmscore1,tmscore2))
                template_align_list.append(template_align_cif)
        tmscore_arr = np.array(tmscore_arr)
        if tm_threshold<1:
            tmscore_arr_max = np.arange(len(tmscore_arr))
            tmscore_arr_max = tmscore_arr_max[tmscore_arr[:,0]>tm_threshold]
            if len(tmscore_arr_max) < 1:
                break
            tmscore_arr_max = tmscore_arr_max[np.argmax(tmscore_arr[tmscore_arr_max,1])]
        else:
            tmscore_arr_max = np.argmax(tmscore_arr[:,1])
        tmscore_arr = tmscore_arr[:, 0]

        if tmscore_arr[tmscore_arr_max] <tm_threshold and tm_threshold<1:
            break
        elif tm_threshold>=1 and tmscore_arr[tmscore_arr_max]<0:
            break
        else:
            template_align_cif = template_align_list[tmscore_arr_max]
            template_list_used[template_list[tmscore_arr_max]] += 1
            cif_type = template_align_cif[-3:]
            shutil.copy(template_align_cif,os.path.join(save_pth2,f"step{ii}.{cif_type}"))
            Remove_res(template_align_cif,out_raw_pth)

    align_list = os.listdir(save_pth2)
    align_list = [r for r in align_list if r.endswith('.cif') or r.endswith('.pdb')]
    aatype_list = []
    atom14_position_list = []
    atom_14_mask_list = []
    b_factors_list = []
    for align_cif in align_list:
        protein = get_protein_from_file_path(os.path.join(save_pth2,align_cif))
        aatype_list.append(protein.aatype)
        atom14_position_list.append(protein.atom14_positions)
        atom_14_mask_list.append(protein.atom14_mask)
        b_factors_list.append(protein.b_factors)
    out_pth = os.path.join(os.path.dirname(Path(CF_raw_pth)),f"cryoatom_assembly.cif")
    chain_atom14_to_cif(aatype_list,atom14_position_list,atom_14_mask_list,out_pth,b_factors_list)
    return out_pth
def add_args(parser):
    parser.add_argument(
        "--template-dir", "--td","-td", required=True, help="Directory for storing all template structures"
    )
    parser.add_argument(
        "--cryoatom-model", "--c","-c", required=True, help="Models established by CryoAtom"
    )
    parser.add_argument(
        "--tmscore-threshold",
        "--tt",
        "-tt",
        type=float,
        default=1,
        help="The threshold of TM-Score. Template scores greater than this threshold will be reused. That is, if the threshold exceeds 1, each template will be used only once; otherwise, each template can be reused.",
    )
    return parser
def main(args):
    AlignScript = os.path.join(pkg_dir, "CryoAtom", "utils", "USalign")
    work_dir = os.path.join(os.path.dirname(Path(args.cryoatom_model)), "align_result")
    work_dir_exist = os.path.isdir(work_dir)
    print(f"Working directory: {work_dir}\nRunning the cryoatom assemble program...")
    output_cif = AlignWithTPs(CF_raw_pth=args.cryoatom_model, template_dir=args.template_dir, save_pth=work_dir,
                              AlignScript=AlignScript, tm_threshold=args.tmscore_threshold)
    if output_cif and not work_dir_exist and os.path.isdir(work_dir):
        shutil.rmtree(work_dir)
    print(f'done! The file has been saved to: {output_cif}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description='Use templates to fill in the CryoAtom model')
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
