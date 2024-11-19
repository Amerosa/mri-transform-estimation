import numpy as np
import sigpy as sp

from .factors import calc_factors, make_grids
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

def downsample_4d(array, factor):
    xs = array.shape[-3] // factor
    ys = array.shape[-2] // factor
    zs = array.shape[-1] // factor
    x_start = (array.shape[1] - xs) // 2
    y_start = (array.shape[2] - ys) // 2
    z_start = (array.shape[3] - zs) // 2
    return array[:, x_start:x_start+xs, y_start:y_start+ys,:]

def generate_corrupt_kspace(img, mps, num_shots, rotations):
    kgrid, kkgrid, rgrid, rkgrid = make_grids(img.shape)
    transforms = generate_transforms(num_shots, rotations)
    mask = generate_sampling_mask(num_shots, img.shape)
    corr_ksp = np.zeros(mps.shape, dtype=np.complex64)
    for shot_idx in range(num_shots):
        factors_trans, factors_tan, factors_sin = calc_factors(transforms[shot_idx:shot_idx+1], kgrid, rkgrid)
        T = RigidTransform(img.shape, factors_trans, factors_tan, factors_sin)
        corr_ksp += np.sum(mask[shot_idx] * sp.fft(mps[:, np.newaxis] * (T * img), axes=[-3,-2,-1]), axis=1)
    return corr_ksp, transforms, mask