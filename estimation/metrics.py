from .transforms import RigidTransform
from .factors import calc_factors
import sigpy as sp

def objective_all_shot(img, ksp, mps, mask, transforms, kgrid, rkgrid):
    num_shots = len(transforms)
    obj = 0.0
    for shot in range(num_shots):
        obj += objective_one_shot(img, ksp, mps, mask[shot], transforms[shot:shot+1], kgrid, rkgrid)
    return obj

def objective_one_shot(img, ksp, mps, mask, transform, kgrid, rkgrid):
    factors_trans, factors_tan, factors_sin = calc_factors(transform, kgrid, rkgrid)
    T = RigidTransform(img.shape, factors_trans, factors_tan, factors_sin)
    resid  = (mask * sp.fft(mps[:, None] * (T * img), axes=[-3,-2,-1])) 
    resid -= (mask * ksp[:, None])
    xp = sp.get_array_module(img)
    return (xp.sum(xp.abs(resid)**2) / xp.size(ksp))