"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/affine_utils.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import torch
from CryoAtom.utils.torch_utlis import is_ndarray,shared_cat
from torch import nn
def get_affine(rot_matrix, shift):
    is_torch = torch.is_tensor(rot_matrix) and torch.is_tensor(shift)
    is_numpy = is_ndarray(rot_matrix) and is_ndarray(shift)

    if is_torch or is_numpy:
        if len(rot_matrix.shape) == len(shift.shape):
            return shared_cat((rot_matrix, shift), dim=-1, is_torch=is_torch)
        elif len(rot_matrix.shape) == len(shift.shape) + 1:
            return shared_cat((rot_matrix, shift[..., None]), dim=-1, is_torch=is_torch)
        else:
            raise ValueError(
                f"get_affine does not support rotation matrix of shape {rot_matrix.shape}"
                f"and shift of shape {shift.shape} "
            )
    else:
        raise ValueError(
            f"get_affine does not support different types for rot_matrix and shift, ie one is a numpy array, "
            f"the other is a torch tensor "
        )


def get_affine_translation(affine):
    return affine[..., :, -1]
def get_affine_rot(affine):
    return affine[..., :3, :3]
def invert_affine(affine):
    inv_rots = get_affine_rot(affine).transpose(-1, -2)
    t = torch.einsum("...ij,...j->...i", inv_rots, affine[..., :, -1])
    inv_shift = -t
    return get_affine(inv_rots, inv_shift)
def affine_mul_vecs(affine, vecs):
    num_unsqueeze_dims = len(vecs.shape) - len(affine.shape) + 1
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    return torch.einsum(
        "...ij, ...j-> ...i", get_affine_rot(affine), vecs
    ) + get_affine_translation(affine)
def vecs_to_local_affine(affine, vecs):
    return affine_mul_vecs(invert_affine(affine), vecs)

def grid_sampler_normalize(coord, size, align_corners=False):
    if align_corners:
        return (2 / (size - 1)) * coord - 1
    else:
        return ((2 * coord + 1) / size) - 1


def sample_centered_cube(
    grid,
    rotation_matrices,
    shifts,
    cube_side=10,
    align_corners=True,
):
    assert len(grid.shape) == 5
    bz, cz, szz, szy,szx = grid.shape
    sz = torch.tensor([szx,szy,szz])
    align_d = 1 if align_corners else 0
    scale_mult = (
        (
            torch.Tensor(
                [
                    cube_side,
                    cube_side,
                    cube_side,
                ]
            )
            - align_d
        )
        / (sz - align_d)
    ).to(grid.device)
    center_shift_vector = -torch.Tensor(
        [[cube_side // 2, cube_side // 2, cube_side // 2]]
    ).to(shifts.device)
    center_shift_vector = torch.einsum(
        "...ij, ...j-> ...i", rotation_matrices, center_shift_vector.expand(len(rotation_matrices),3)
    )

    rotation_matrices = rotation_matrices * scale_mult[None][None]
    shifts = (
        grid_sampler_normalize(
            shifts + center_shift_vector, sz.to(shifts.device), align_corners=align_corners
        )
        + rotation_matrices.sum(-1)
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = nn.functional.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + 3 * (cube_side,),
        align_corners=align_corners,
    )

    return nn.functional.grid_sample(grid.detach(), cc, align_corners=align_corners)

def get_z_to_w_rotation_matrix(w):
    """
    Special case of get_a_to_b_rotation matrix for when you are converting from
    the Z axis to some vector w. Algorithm comes from
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    _w = nn.functional.normalize(w, p=2, dim=-1)
    # (1, 0, 0) cross _w
    v2 = -_w[..., 2]
    v3 = _w[..., 1]
    # (1, 0, 0) dot _w
    c = _w[..., 0]
    # The result of I + v_x + v_x_2 / (1 + c)
    # [   1 - (v_2^2 + v_3 ^ 2) / (1 + c),                   -v_3,                    v_2]
    # [                               v_3,    1 - v_3^2 / (1 + c),    v_2 * v_3 / (1 + c)]
    # [                              -v_2,    v_2 * v_3 / (1 + c),    1 - v_2^2 / (1 + c)]
    R = torch.zeros(*w.shape[:-1], 3, 3).to(w.device)
    v2_2, v3_2 = ((v2 ** 2) / (1 + c)), ((v3 ** 2) / (1 + c))
    v2_v3 = v2 * v3 / (1 + c)
    R[..., 0, 0] = 1 - (v2_2 + v3_2)
    R[..., 0, 1] = -v3
    R[..., 0, 2] = v2
    R[..., 1, 0] = v3
    R[..., 1, 1] = 1 - v3_2
    R[..., 1, 2] = v2_v3
    R[..., 2, 0] = -v2
    R[..., 2, 1] = v2_v3
    R[..., 2, 2] = 1 - v2_2
    return R
def sample_centered_rectangle(
    grid,
    rotation_matrices,
    shifts,
    rectangle_length=10,
    rectangle_width=3,
    align_corners=True,
):
    assert len(grid.shape) == 5
    bz, cz, szz, szy, szx = grid.shape
    sz = torch.tensor([szx, szy, szz])
    align_d = 1 if align_corners else 0
    scale_mult = (
        (
            torch.Tensor(
                [
                    rectangle_width,
                    rectangle_width,
                    rectangle_length,
                ]
            )
            - align_d
        )
        / (sz - align_d)
    ).to(grid.device)
    center_shift_vector = -torch.Tensor(
        [[rectangle_width // 2, rectangle_width // 2, rectangle_width // 2]]
    ).to(shifts.device)
    center_shift_vector = torch.einsum(
        "...ij, ...j-> ...i", rotation_matrices, center_shift_vector.expand(len(rotation_matrices),3)
    )
    rotation_matrices = rotation_matrices * scale_mult[None][None]
    shifts = (
        grid_sampler_normalize(
            shifts + center_shift_vector, sz.to(shifts.device), align_corners=align_corners
        )
        + rotation_matrices.sum(-1)
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = nn.functional.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + (rectangle_length, rectangle_width, rectangle_width),
        align_corners=align_corners,
    )
    return nn.functional.grid_sample(grid.detach(), cc, align_corners=align_corners)


def sample_centered_rectangle_along_vector(
    batch_grids,
    batch_vectors,
    batch_origin_points,
    rectangle_length=10,
    rectangle_width=3,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_vectors = [batch_vectors]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, vectors, origin_points) in zip(
        batch_grids, batch_vectors, batch_origin_points
    ):
        rotation_matrices = get_z_to_w_rotation_matrix(vectors)
        rectangle = sample_centered_rectangle(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )
        if marginalization_dims is not None:
            rectangle = rectangle.sum(dim=marginalization_dims)
        output.append(rectangle)
    output = torch.cat(output, dim=0)
    return output
def sample_centered_cube_rot_matrix(
    batch_grids,
    batch_rot_matrices,
    batch_origin_points,
    cube_side=10,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_rot_matrices = [batch_rot_matrices]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, rotation_matrices, origin_points) in zip(
        batch_grids, batch_rot_matrices, batch_origin_points
    ):
        cube = sample_centered_cube(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            cube_side=cube_side,
        )
        if marginalization_dims is not None:
            cube = cube.sum(dim=marginalization_dims)
        output.append(cube)
    output = torch.cat(output, dim=0)
    return output
def rots_from_two_vecs(e1_unnormalized, e2_unnormalized):
    e1 = nn.functional.normalize(e1_unnormalized, p=2, dim=-1)
    c = torch.einsum("...i,...i->...", e2_unnormalized, e1)[..., None]  # dot product
    e2 = e2_unnormalized - c * e1
    e2 = nn.functional.normalize(e2, p=2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)
def init_random_affine_from_translation(translation):
    v, w = torch.rand_like(translation), torch.rand_like(translation)
    rot = rots_from_two_vecs(v, w)
    return get_affine(rot, translation)
def affine_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    rotation = rots_from_two_vecs(
        e1_unnormalized=origin - point_on_neg_x_axis,
        e2_unnormalized=point_on_xy_plane - origin,
    )
    return get_affine(rotation, origin)
def affine_from_tensor4x4(m):
    assert m.shape[-1] == 4 == m.shape[-2]
    return get_affine(m[..., :3, :3], m[..., :3, -1])
def fill_rotation_matrix(xx, xy, xz, yx, yy, yz, zx, zy, zz):
    R = torch.zeros(*xx.shape, 3, 3).to(xx.device)
    R[..., 0, 0] = xx
    R[..., 0, 1] = xy
    R[..., 0, 2] = xz

    R[..., 1, 0] = yx
    R[..., 1, 1] = yy
    R[..., 1, 2] = yz

    R[..., 2, 0] = zx
    R[..., 2, 1] = zy
    R[..., 2, 2] = zz
    return R
def affine_mul_rots(affine, rots):
    num_unsqueeze_dims = len(rots.shape) - len(affine.shape)
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    rotation = affine[..., :3, :3] @ rots
    return get_affine(rotation, get_affine_translation(affine))
def affine_composition(a1, a2):
    """
    Does the operation a1 o a2
    """
    rotation = get_affine_rot(a1) @ get_affine_rot(a2)
    translation = affine_mul_vecs(a1, get_affine_translation(a2))
    return get_affine(rotation, translation)

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
