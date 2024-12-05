import numpy as np
import sigpy as sp
from math import ceil
from .factors import calc_factors, generate_transform_grids
from .transforms import RigidTransform

#TODO include translations parameters as well
def generate_transforms(num_shots, rotations, random=False):
    transforms = np.zeros((num_shots, 6), dtype=float)
    for axis, degree in enumerate(rotations):
        if (degree == 0):
            continue
        if random:
            rads = (np.random.rand(num_shots) - 0.5) * degree * np.pi / 180
        else:
            rads = np.linspace(-0.5, 0.5, num_shots, endpoint=True) * degree * np.pi / 180
        transforms[:, axis+3] = rads
    return transforms - np.mean(transforms, axis=0)

#TODO refactor this function for more sample schemes and axis selection
def generate_sampling_mask(num_shots, img_shape):
    mask = np.zeros((num_shots, *img_shape), dtype=bool)
    partition_size = img_shape[-1] // num_shots
    for shot in range(num_shots):
        mask[shot, :,:, shot*partition_size:(shot+1)*partition_size] = 1
        #mask[shot, :,:, shot::num_shots] = 1 
    return mask

def _normalize_axes(axes, ndim):
    if axes is None:
        return tuple(range(ndim))
    else:
        return tuple(a % ndim for a in sorted(axes))

def resample(x, subdivisions, axes=None, is_kspace=False):
    factor = 2 ** subdivisions
    axes = _normalize_axes(axes, x.ndim)
    idx = [slice(None)] * x.ndim
    for axis in axes:
        mid = x.shape[axis] // 2
        extent = x.shape[axis] // factor
        hi = ceil(extent / 2)
        lo = extent // 2
        idx[axis] = slice(mid-lo, mid+hi)

    if not is_kspace:
        return sp.ifft(sp.fft(x, axes=(-3,-2,-1))[*idx], axes=(-3,-2,-1))
    return x[*idx]

def generate_corrupt_kspace(img, mps, mask, transforms):
    num_shots = len(transforms)
    kgrid, _, _, rkgrid = generate_transform_grids(img.shape)
    corr_ksp = np.zeros(mps.shape, dtype=np.complex128)
    for shot_idx in range(num_shots):
        factors_trans, factors_tan, factors_sin = calc_factors(transforms[shot_idx:shot_idx+1], kgrid, rkgrid)
        T = RigidTransform(img.shape, factors_trans, factors_tan, factors_sin)
        corr_ksp += np.sum(mask[shot_idx] * sp.fft(mps[:, np.newaxis] * (T * img), axes=[-3,-2,-1]), axis=1)
    return corr_ksp