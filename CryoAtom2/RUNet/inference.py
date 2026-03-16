import os
import sys
import torch
from scipy.spatial import cKDTree
from CryoAtom2.utils.mrc_tools import load_map,make_model_grid,save_mrc
from CryoAtom2.utils.network_tools import map_segmentation,map_reconstruction
from CryoAtom2.utils.save_pdb_utils import ca_ps_to_pdb
from CryoAtom2.utils.torch_utlis import get_batch_slices,get_device_names
from CryoAtom2.RUNet.RUNet import RUNet
from CryoAtom2.utils.multi_gpu_wrapper import MultiGPUWrapper
from CryoAtom2.utils.pdb_utils import load_cas_c1s_from_structure
import numpy as np
import tqdm
def load_coords_from_getp(stu_fn):
    coords = []
    with open(stu_fn,'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("ATOM"):
            coords.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
    return np.array(coords,dtype=np.float32)
def get_density(grid, coord):
    x, y, z = [int(x) for x in coord]
    n0, n1, n2 = grid.shape
    if x >= n2 or x < 0 or \
       y >= n1 or y < 0 or \
       z >= n0 or z < 0:
        return 0.
    else:
        return grid[z, y, x]
def filter_with_distance(chain, density, r0=3.):
    while True:
        tree = cKDTree(chain)
        keep = [False for _ in range(len(chain))]
        for i in range(len(chain)):
            inds = tree.query_ball_point(chain[i], r0)
            dens = density[inds]

            ii = np.argmax(dens)
            ii = inds[ii]
            keep[ii] = True
        if np.all(keep):
            break

        chain = chain[keep]
        density = density[keep]

    return chain
def get_lattice_meshgrid_np(shape_size, no_shift=False):
    linspace = [np.linspace(
        0.5 if not no_shift else 0,
        shape - (0.5 if not no_shift else 1),
        shape,
    ) for shape in shape_size]
    mesh = np.stack(
        np.meshgrid(linspace[0], linspace[1], linspace[2], indexing="ij"),
        axis=-1,
    )
    return mesh
def Density_LocalSum(grid,threshold=0.6):
    Z,Y,X = grid.shape
    grid_padding = np.zeros((Z+2,Y+2,X+2),dtype=float)
    grid_padding[1:-1,1:-1,1:-1] = grid
    mask = grid_padding > threshold
    indices = np.where(mask)
    sorted_i = np.argsort(1-grid_padding[indices])
    sorted_indices = (a[sorted_i] for a in indices)
    pred = np.copy(grid_padding)
    for ii,jj,kk in zip(*sorted_indices):
        if mask[ii,jj,kk]:
            window_new = np.zeros((3,3,3))
            window_new[1,1,1] = np.sum(pred[ii-1:ii+2,jj-1:jj+2,kk-1:kk+2])
            pred[ii-1:ii+2,jj-1:jj+2,kk-1:kk+2] = window_new
            mask[ii-1:ii+2,jj-1:jj+2,kk-1:kk+2] = False
    return pred[1:-1,1:-1,1:-1]
def grid_to_points(grid, threshold, cluster_threshold, neighbour_distance_threshold):
    """
    MIT License

    Copyright (c) 2022 Kiarash Jamali

    This function comes from: https://github.com/3dem/model-angelo/blob/main/model_angelo/c_alpha/inference.py

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
    """
    lattice = np.flip(get_lattice_meshgrid_np(grid.shape, no_shift=False), -1)

    output_points_before_pruning = np.copy(lattice[grid > threshold, :].reshape(-1, 3))

    points = lattice[grid > threshold, :].reshape(-1, 3)
    probs = grid[grid > threshold]
    # sorted_indices = np.argsort(probs)[::-1]
    # probs = probs[sorted_indices]
    # points = points[sorted_indices]
    for _ in range(3):
        kdtree = cKDTree(np.copy(points))
        n = 0
        new_points = np.copy(points)
        for p in points:
            neighbours = kdtree.query_ball_point(p,cluster_threshold)
            selection = list(neighbours)
            if len(neighbours) > 1 and np.sum(probs[selection]) > 0:
                keep_idx = np.argmax(probs[selection])
                prob_sum = np.sum(probs[selection])

                new_points[selection[keep_idx]] = (
                    np.sum(probs[selection][..., None] * points[selection], axis=0)
                    / prob_sum
                )
                probs[selection] = 0
                probs[selection[keep_idx]] = prob_sum

            n += 1

        points = new_points[probs > 0].reshape(-1, 3)
        probs = probs[probs > 0]

    kdtree = cKDTree(np.copy(points))
    for point_idx, point in enumerate(points):
        d, _ = kdtree.query(point, 2)
        if d[1] > neighbour_distance_threshold:
            points[point_idx] = np.nan

    points = points[~np.isnan(points).any(axis=-1)].reshape(-1, 3)

    output_points = points
    return output_points, output_points_before_pruning



def infer(args):
    os.makedirs(args.output_path, exist_ok=True)
    model_output_dir = os.path.join(args.output_path, "output_ca_p_points.cif")
    device_names = get_device_names(args.device)
    num_devices = len(device_names)
    model_args = {}
    if args.map_path.endswith("map") or args.map_path.endswith("mrc"):
        grid_np, voxel_size, global_origin = load_map(args.map_path)
        if args.mask_path:
            assert args.mask_path.endswith("map") or args.mask_path.endswith("mrc")
            mask_np,b1,b2 = load_map(args.mask_path)
        else:
            mask_np = np.ones(grid_np.shape)
        grid_np = grid_np*mask_np
        grid_np, voxel_size, global_origin = make_model_grid(
            np.copy(grid_np), voxel_size, global_origin, target_voxel_size=1.51
        )
    else:
        raise RuntimeError(f"File {args.map_path} is not a cryo-em density map file format.")
    grid_np = (grid_np - np.mean(grid_np)) / np.std(grid_np)
    grid = (grid_np).astype(np.float32)
    batch_size = int(args.batch_size)
    windows_size = args.windows_size
    stride = args.stride
    shape = np.array(grid_np.shape[-3:])
    total_batch_num = np.prod(np.ceil(shape/stride))

    pbar = tqdm.tqdm(
        total=total_batch_num,
        file=sys.stdout,
        position=0,
        leave=True,
    )
    class_num = 4
    with MultiGPUWrapper(RUNet, model_args, args.log_dir, device_names) as wrapper:
        with torch.no_grad():
            if np.all(shape>windows_size):
                segmentation = map_segmentation(torch.from_numpy(grid), stride=stride, windows_size=windows_size)
                segmentation = torch.stack(segmentation, dim=0)
                segmentation = segmentation[:, None]
                grid_batches = get_batch_slices(segmentation.shape[0], batch_size*num_devices)
                out_segmentation = torch.zeros((segmentation.shape[0],class_num)+segmentation.shape[-3:], device=torch.device("cpu"))
                for grid_batch in grid_batches:
                    meta_batches = get_batch_slices(len(grid_batch), batch_size)
                    meta_batch_list = [{"V0":segmentation[grid_batch][mb],"run_iters":3} for mb in meta_batches]
                    meta_batch_out = wrapper(meta_batch_list)
                    for ii,mb in enumerate(meta_batches):
                        out_segmentation[[grid_batch[jj] for jj in mb]] = meta_batch_out[ii].softmax(dim=1).detach().cpu()
                    pbar.update(len(grid_batch))
                out_segmentation = out_segmentation.numpy()
                pred = map_reconstruction(out_segmentation, (class_num,)+grid.shape, stride=stride, windows_size=windows_size)
            else:
                meta_batch_list = [{"V0":torch.from_numpy(grid)[None][None],"run_iters":3}]
                meta_batch_out = wrapper(meta_batch_list)
                pred = torch.softmax(meta_batch_out[0],dim=1)[0].detach().cpu().numpy()
                pbar.update(1)
    pbar.close()
    atom_grid = {'CA':pred[1],'P':pred[3]}
    atom_threshs = {'CA':args.ca_threshold,'P':args.p_threshold}
    atom_r0 = {'CA':1.4,'P':3}
    atom_positions = {}
    # ca_grid = pred[1]
    # p_grid = pred[3]
    script = args.getp_path
    for atom_type in ['P']:
        atom_mrc_pth = os.path.join(args.output_path, f"{atom_type}.mrc")
        atom_pdb_pth = os.path.join(args.output_path, f"{atom_type}.pdb")
        atom_log_pth = os.path.join(args.output_path, f"temp.log")
        save_mrc(atom_grid[atom_type],voxel_size,global_origin,atom_mrc_pth)
        atom_thresh = atom_threshs[atom_type]
        lines = os.popen(f'{script} --in {atom_mrc_pth} --out {atom_pdb_pth} --thresh {atom_thresh} --nt 4 >> {atom_log_pth}').read()
        if os.path.getsize(atom_pdb_pth) == 0:
            atom_positions[atom_type] = np.array([]).reshape(-1,3)
        else:
            atom_position = load_coords_from_getp(atom_pdb_pth)
            atom_dens = np.asarray([get_density(atom_grid[atom_type], (atom_position[i] - global_origin) / voxel_size) for i in range(len(atom_position))])
            atom_positions[atom_type] = filter_with_distance(atom_position, atom_dens, r0=atom_r0[atom_type])
    output_ca_points, output_ca_points_before_pruning = grid_to_points(
        atom_grid["CA"],threshold=atom_threshs['CA'],cluster_threshold=1.1,neighbour_distance_threshold=6/np.min(voxel_size)
    )
    atom_positions['CA'] = output_ca_points * voxel_size[None] + global_origin[None]
    ca_ps_to_pdb(model_output_dir,atom_positions['CA'],atom_positions['P'])

    # output_ca_points, output_ca_points_before_pruning = grid_to_points(
    #     ca_grid,threshold=args.threshold,cluster_threshold=1.1,neighbour_distance_threshold=6/np.min(voxel_size)
    # )
    # output_p_points, output_p_points_before_pruning = grid_to_points(
    #     p_grid,threshold=args.threshold-0.1,cluster_threshold=2.1,neighbour_distance_threshold=6/np.min(voxel_size)
    # )
    # ca_ps_to_pdb(
    #     os.path.join(args.output_path, "output_ca_p_points_before_pruning.cif"),
    #     output_ca_points_before_pruning * voxel_size[None] + global_origin[None],
    #     output_p_points_before_pruning * voxel_size[None] + global_origin[None],
    # )
    # output_file_path = os.path.join(args.output_path, "output_ca_p_points.cif")
    # ca_ps_to_pdb(
    #     output_file_path,
    #     output_ca_points * voxel_size[None] + global_origin[None],
    #     output_p_points * voxel_size[None] + global_origin[None],
    # )

    return model_output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-path", "--v", required=True, help="input cryo-em density map"
    )
    parser.add_argument(
        "--mask-path", "--m", help="input cryo-em mask map"
    )
    parser.add_argument(
        "--output-path",
        "--o",
        required=True,
        help="The C-alpha atoms ouput path",
    )
    parser.add_argument(
        "--device",
        "-d",
        "--d",
        help="compute device, pick one of {cpu, cuda:number}. "
             "Default set to use cuda.",
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        "--batch-size", type=int, default=6, help="Batch size for inference"
    )
    parser.add_argument(
        "--stride", type=int, default=50, help="The stride for inference"
    )
    parser.add_argument("--windows-size", type=int, default=65, help="The windows for inference")
    parser.add_argument(
        "--ca_threshold",
        type=float,
        default=0.6,
        help="Probability threshold for inference",
    )
    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.6,
        help="Probability threshold for inference",
    )
    parser.add_argument("--log-dir",type=str,help="The model load dir")
    args = parser.parse_args()

    infer(
        args,
    )