"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from: [https://github.com/3dem/model-angelo/blob/main/model_angelo/gnn/inference.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import os.path
import sys
from collections import namedtuple
import torch
import tqdm
from CryoAtom2.utils.mrc_tools import load_map,make_model_grid
from CryoAtom2.utils.flood_fill import final_results_to_cif
from CryoAtom2.utils.CryoNet_inference_utils import (
    get_neighbour_idxs,
    init_empty_collate_results,
    get_inference_data,
    argmin_random,
    collate_nn_results,
    run_inference_on_data,
    init_protein_from_see_alpha,
    get_final_nn_results,
)
from CryoAtom2.utils.multi_gpu_wrapper import MultiGPUWrapper
from CryoAtom2.utils.torch_utlis import get_device_names
from CryoAtom2.CryoNet.CryoFolder_no_seq import CryoFolder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MRCObject = namedtuple("MRCObject", ["grid", "voxel_size", "global_origin"])

def infer(args):
    output_dir = os.path.dirname(args.output_dir)
    device_names = get_device_names(args.device)
    num_devices = len(device_names)
    model_args = {
        "hidden_features": 384,
        "num_layers_former": 16,
        "num_layers_ipa": 7,
        "attention_heads": 8,
        "attention_features": 72,
    }
    complex = None
    if args.struct.endswith("cif") or args.struct.endswith("pdb"):
        complex = init_protein_from_see_alpha(args.struct)
    if complex is None:
        raise RuntimeError(f"File {args.struct} is not a supported file format.")
    grid_data = None
    if args.map_path.endswith("map") or args.map_path.endswith("mrc"):
        grid, voxel_size, global_origin = load_map(args.map_path, multiply_global_origin=True)
        grid, voxel_size, global_origin = make_model_grid(
            grid,
            voxel_size,
            global_origin,
            target_voxel_size=1.51,
        )
        grid_data = MRCObject(grid,voxel_size,global_origin)
    if grid_data is None:
        raise RuntimeError(
            f"Grid volume file {args.map_path} is not a cryo_em density map file format."
        )
    num_res = len(complex.rigidgroups_gt_frames)
    CryNet_crop_length = args.crop_length if num_res>args.crop_length else num_res
    if CryNet_crop_length < 30:
        raise RuntimeError(f"Please ensure that the number of input residues is greater than 30.")
    collated_results = init_empty_collate_results(
        num_res,
        device="cpu",
    )

    residues_left = num_res
    total_steps = num_res * args.repeat_per_residue
    steps_left_last = total_steps

    pbar = tqdm.tqdm(total=total_steps, file=sys.stdout, position=0, leave=True)
    # Get an initial set of pointers to neighbours for more efficient inference

    init_neighbours = get_neighbour_idxs(complex, k=CryNet_crop_length//3)
    with MultiGPUWrapper(CryoFolder, model_args, args.model_dir, device_names, fp16=False) as wrapper:

        while residues_left > 0:
            idxs = argmin_random(
                collated_results["counts"],
                init_neighbours,
                batch_size=num_devices,
                repeat_per_residue=args.repeat_per_residue,
            )
            data = get_inference_data(complex, grid_data, idxs,crop_length=CryNet_crop_length,num_devices=num_devices)

            results = run_inference_on_data(
                wrapper,
                data,
            )
            for device_id in range(num_devices):
                collated_results, complex = collate_nn_results(
                    collated_results,
                    results[device_id],
                    data[device_id]["indices"],
                    complex,
                    offset=0,
                    end_flag=True if args.aggressive_repeat else args.end_flag,
                    crop_length=CryNet_crop_length,
                    repeat_num=args.repeat_per_residue,
                )
            residues_left = (
                num_res
                - torch.sum(collated_results["counts"] > args.repeat_per_residue - 1).item()
            )
            steps_left = (
                total_steps
                - torch.sum(
                    collated_results["counts"].clip(0, args.repeat_per_residue)
                ).item()
            )

            pbar.update(n=int(steps_left_last - steps_left))
            steps_left_last = steps_left

    pbar.close()

    final_results = get_final_nn_results(collated_results)
    output_path = os.path.join(args.output_dir, "model_net.cif")
    final_results_to_cif(
        final_results = final_results,
        cif_path = output_path,
        prot_mask = complex.prot_mask,
        verbose=True,
        aggressive_pruning=args.aggressive_pruning,
        mask_threshold=args.mask_threshold,
        end_flag=args.end_flag,
    )

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", "--m", required=True, help="The path to the input map")
    parser.add_argument(
        "--struct", "--s", required=True, help="The path to the structure file"
    )
    parser.add_argument("--model-dir", required=True, help="Where the model at")
    parser.add_argument("--output-dir", default=".", help="Where to save the results")
    parser.add_argument("--device", default="cpu", help="Which device to run on")
    parser.add_argument(
        "--crop-length", type=int, default=400, help="How many points per batch"
    )
    parser.add_argument(
        "--repeat-per-residue",
        default=3,
        type=int,
        help="How many times to repeat per residue",
    )
    parser.add_argument(
        "--aggressive-pruning",
        action="store_true",
        help="Only build parts of the model that have a good match with the sequence. "
        + "Will lower recall, but quality of build is higher",
    )
    parser.add_argument(
        "--seq-attention-batch-size",
        type=int,
        default=300,
        help="Lower memory usage by processing the sequence in batches.",
    )
    args = parser.parse_args()
    infer(args)
