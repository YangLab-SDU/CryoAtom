import numpy as np
import torch
import einops
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Atom import DisorderedAtom
from CryoAtom.utils.residue_constants import restype_3to1,rigid_group_atom_positions, restype_name_to_atom14_names,get_chi_atom_indices, chi_pi_periodic, atom_order,restype_1_to_index
from CryoAtom.utils.affine_utils import affine_from_3_points, get_affine_rot, get_affine, affine_mul_vecs, invert_affine, init_random_affine_from_translation
from CryoAtom.utils.torch_utlis import batched_gather
from CryoAtom.utils.residue_constants import chi_angles_mask as chi_mask
def load_cas_from_structure(stu_fn, all_structs=False, quiet=True):
    """
    MIT License

    Copyright (c) 2022 Kiarash Jamali

    This function comes from: https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/pdb_utils.py

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
    """
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
def load_affines_cas_from_structure(stu_fn, all_structs=False, quiet=True):
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
    ncac_coords = []
    only_ca_coords = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                        ncac_coords.append([residue['N'].get_coord(),residue['CA'].get_coord(),residue['C'].get_coord()])
                    elif residue.has_id('CA'):
                        only_ca_coords.append(residue['CA'].get_coord())
    rigidgroups_gt_frames = torch.zeros((len(ncac_coords)+len(only_ca_coords), 1, 3, 4), dtype=torch.float)
    if len(ncac_coords)>0:
        ncac = torch.from_numpy(np.array(ncac_coords)).float()
        affines = affine_from_3_points(
                ncac[..., 2, :], ncac[..., 1, :], ncac[..., 0, :]
            )
        affines[..., :, 0] = affines[..., :, 0] * (-1)
        affines[..., :, 2] = affines[..., :, 2] * (-1)
        rigidgroups_gt_frames[:len(ncac_coords),0] = affines
    if len(only_ca_coords)>0:
        rigidgroups_gt_frames[len(ncac_coords):, 0] = init_random_affine_from_translation(
            torch.from_numpy(np.array(only_ca_coords)).float()
        )
    return rigidgroups_gt_frames.numpy()
def load_affines_from_structure(stu_fn, all_structs=False, quiet=True):
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
    n_coords = []
    c_coords = []
    sequences = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for model in structure:
            for chain in model:
                sequence = ''
                for residue in chain:
                    if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                        aa_name = restype_3to1.get(residue.get_resname(), 'Z')
                        if aa_name == 'Z':
                            continue
                        else:
                            sequence = sequence + aa_name
                            n_atom = residue['N']
                            ca_atom = residue['CA']
                            c_atom = residue['C']
                            n_coords.append(n_atom.get_coord())
                            ca_coords.append(ca_atom.get_coord())
                            c_coords.append(c_atom.get_coord())
                if sequence != '':
                    sequences.append(sequence)
    n_coords = einops.rearrange(torch.tensor(n_coords), '(b h) d -> b h d', h=1,d=3)
    ca_coords = einops.rearrange(torch.tensor(ca_coords), '(b h) d -> b h d', h=1, d=3)
    c_coords = einops.rearrange(torch.tensor(c_coords), '(b h) d -> b h d', h=1, d=3)
    ncac = torch.cat([n_coords,ca_coords,c_coords],dim=1)
    ncacs = ncac.numpy()
    affines = affine_from_3_points(
            ncac[..., 2, :], ncac[..., 1, :], ncac[..., 0, :]
        )
    affines[..., :, 0] = affines[..., :, 0] * (-1)
    affines[..., :, 2] = affines[..., :, 2] * (-1)
    affines = affines.numpy()
    # affines = einops.rearrange(affines, '(b h) w d -> b h w d', b=1,w=3,d=4)
    return affines,ncacs,sequences

def load_atom14_from_structure(stu_fn, all_structs=False, quiet=True):
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
    atom14_position = []
    atom14_mask = []
    sequences = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for model in structure:
            for chain in model:
                sequence = ''
                for residue in chain:
                    if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                        resname = residue.get_resname()
                        aa_name = restype_3to1.get(resname, 'Z')
                        if aa_name == 'Z':
                            continue
                        else:
                            atom_position = np.zeros((14,3))
                            atom_mask = np.zeros((14))
                            sequence = sequence + aa_name
                            for atom_name in rigid_group_atom_positions[resname]:
                                atom_name = atom_name[0]
                                atom14idx = restype_name_to_atom14_names[resname].index(atom_name)
                                try:
                                    atom_position[atom14idx,:] = residue[atom_name].get_coord()
                                    atom_mask[atom14idx] = 1
                                except KeyError:
                                    continue
                            atom14_position.append(atom_position)
                            atom14_mask.append(atom_mask)
                if sequence != '':
                    sequences.append(sequence)

    atom14_position = np.array(atom14_position)
    atom14_mask = np.array(atom14_mask)
    return atom14_position,sequences,atom14_mask

def atom37_to_torsion_angles(
    aatype,
    all_atom_positions,
    all_atom_mask
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device,dtype=torch.long
    )

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )
    chi_angles_mask = list(chi_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = affine_from_3_points(
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 0, :],
    )
    torsion_frames[..., :, 0] = torsion_frames[..., :, 0] * (-1)
    torsion_frames[..., :, 2] = torsion_frames[..., :, 2] * (-1)
    torsion_frames = get_affine(get_affine_rot(torsion_frames),torsions_atom_pos[...,2,:])
    fourth_atom_rel_pos = affine_mul_vecs(invert_affine(torsion_frames),torsions_atom_pos[...,3,:])
    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )
    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]
    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        chi_pi_periodic,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    return torsion_angles_sin_cos,alt_torsion_angles_sin_cos,torsion_angles_mask
def load_atom37_from_structure(stu_fn, all_structs=False, quiet=True):
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
    atom14_position = []
    atom14_mask = []
    sequences = []
    for model in structure:
        if not quiet:
            print("Model contains", len(model), "chain(s)")
        for model in structure:
            for chain in model:
                sequence = ''
                for residue in chain:
                    if residue.has_id('N') and residue.has_id('CA') and residue.has_id('C'):
                        resname = residue.get_resname()
                        aa_name = restype_3to1.get(resname, 'Z')
                        if aa_name == 'Z':
                            continue
                        else:
                            atom_position = np.zeros((37,3))
                            atom_mask = np.zeros(37)
                            sequence = sequence + aa_name
                            for atom_name in rigid_group_atom_positions[resname]:
                                atom_name = atom_name[0]
                                atomtype = atom_order[atom_name]
                                try:
                                    atom_position[atomtype,:] = residue[atom_name].get_coord()
                                    atom_mask[atomtype] = 1
                                except KeyError:
                                    continue
                            atom14_position.append(atom_position)
                            atom14_mask.append(atom_mask)
                if sequence != '':
                    sequences.append(sequence)
    atom37_position = np.array(atom14_position)
    atom37_position = torch.tensor(atom37_position,dtype=torch.float64)
    sequences = ''.join(sequences)
    aatypes = np.array([restype_1_to_index[r] for r in sequences])
    aatypes = torch.tensor(aatypes,dtype=torch.long)
    atom37_mask = np.array(atom14_mask)
    atom37_mask = torch.tensor(atom37_mask,dtype=torch.float64)
    return atom37_position,aatypes,atom37_mask
def get_torsions(stu_fn):
    atom37_position,aatypes,atom37_mask = load_atom37_from_structure(stu_fn)
    torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask = atom37_to_torsion_angles(aatypes,atom37_position,atom37_mask)
    torsion_angles_sin_cos = torsion_angles_sin_cos.numpy()
    alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos.numpy()
    torsion_angles_mask = torsion_angles_mask.numpy()
    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask