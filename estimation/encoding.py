import sigpy as sp
import numpy as np
from .transforms import *

def AlignedSense(input, mps, masks, factors_trans, factors_tan, factors_sin, shot_batch_size=None):

    img_shape = input.shape
    num_shots = len(masks)
    if shot_batch_size is None:
        shot_batch_size = num_shots

    if shot_batch_size < num_shots:
        num_shot_batches = (num_shots + shot_batch_size - 1) // shot_batch_size
        E = sp.linop.Vstack(
            [
                AlignedSense(
                    input,
                    mps,
                    masks[s * shot_batch_size: (s+1) * shot_batch_size],
                    factors_trans[s * shot_batch_size: (s+1) * shot_batch_size],
                    [f[s * shot_batch_size: (s+1) * shot_batch_size] for f in factors_tan],
                    [f[s * shot_batch_size: (s+1) * shot_batch_size] for f in factors_sin]
                )
                for s in range(num_shot_batches)
            ],
            axis=1,
        )
    
    T = RigidTransform(img_shape, factors_trans, factors_tan, factors_sin)
    S = sp.linop.Multiply(T.oshape, mps[:, np.newaxis])
    F = sp.linop.FFT(S.oshape, axes=(-3, -2, -1))
    A = sp.linop.Multiply(F.oshape, masks)
    E = A * F * S * T
    return E

def GradAlignedSense(input, partial_idx, mps, masks, factors_trans, factors_tan, factors_sin, grad_factors, shot_batch_size=None):

    img_shape = input.shape
    num_shots = len(masks)
    if shot_batch_size is None:
        shot_batch_size = num_shots

    if shot_batch_size < num_shots:
        num_shot_batches = (num_shots + shot_batch_size - 1) // shot_batch_size
        E = sp.linop.Vstack(
            [
                GradAlignedSense(
                    input,
                    partial_idx,
                    mps,
                    masks[s * shot_batch_size: (s+1) * shot_batch_size],
                    factors_trans[s * shot_batch_size: (s+1) * shot_batch_size],
                    [f[s * shot_batch_size: (s+1) * shot_batch_size] for f in factors_tan],
                    [f[s * shot_batch_size: (s+1) * shot_batch_size] for f in factors_sin],
                    {'trans': [gf[s * shot_batch_size: (s+1) * shot_batch_size] for gf in grad_factors['trans']],
                     'tan':   [gf[s * shot_batch_size: (s+1) * shot_batch_size] for gf in grad_factors['tan']],
                     'sin':   [gf[s * shot_batch_size: (s+1) * shot_batch_size] for gf in grad_factors['sin']]
                    }
                )
                for s in range(num_shot_batches)
            ],
            axis=1,
        )
    
    T = GradRigidTransform(partial_idx, img_shape, factors_trans, factors_tan, factors_sin, grad_factors)
    S = sp.linop.Multiply(T.oshape, mps[:, np.newaxis])
    F = sp.linop.FFT(S.oshape, axes=(-3, -2, -1))
    A = sp.linop.Multiply(F.oshape, masks)
    E = A * F * S * T
    E.repr_str = f'GradAlignedSense wrt Partial: {partial_idx}'
    return E

def HessAlignedSense(input, partial_first, partial_second, mps, masks, factors_trans, factors_tan, factors_sin, grad_factors, hess_factors, shot_batch_size=None):

    img_shape = input.shape
    num_shots = len(masks)
    if shot_batch_size is None:
        shot_batch_size = num_shots

    if shot_batch_size < num_shots:
        num_shot_batches = (num_shots + shot_batch_size - 1) // shot_batch_size
        E = sp.linop.Vstack(
            [
                HessAlignedSense(
                    input,
                    partial_first,
                    partial_second,
                    mps,
                    masks[s * shot_batch_size: (s+1) * shot_batch_size],
                    factors_trans[s * shot_batch_size: (s+1) * shot_batch_size],
                    [f[s * shot_batch_size: (s+1) * shot_batch_size] for f in factors_tan],
                    [f[s * shot_batch_size: (s+1) * shot_batch_size] for f in factors_sin],
                    {'trans': [gf[s * shot_batch_size: (s+1) * shot_batch_size] for gf in grad_factors['trans']],
                     'tan':   [gf[s * shot_batch_size: (s+1) * shot_batch_size] for gf in grad_factors['tan']],
                     'sin':   [gf[s * shot_batch_size: (s+1) * shot_batch_size] for gf in grad_factors['sin']]
                    },
                    {'trans': [ [hf[s * shot_batch_size: (s+1) * shot_batch_size] for hf in partials] for partials in hess_factors['trans']],
                     'tan':   [hf[s * shot_batch_size: (s+1) * shot_batch_size] for hf in hess_factors['tan']],
                     'sin':   [hf[s * shot_batch_size: (s+1) * shot_batch_size] for hf in hess_factors['sin']]
                    }
                    
                )
                for s in range(num_shot_batches)
            ],
            axis=1,
        )
    
    T = HessRigidTransform(partial_first, partial_second, img_shape, factors_trans, factors_tan, factors_sin, grad_factors, hess_factors)
    S = sp.linop.Multiply(T.oshape, mps[:, np.newaxis])
    F = sp.linop.FFT(S.oshape, axes=(-3, -2, -1))
    A = sp.linop.Multiply(F.oshape, masks)
    E = A * F * S * T
    E.repr_str = f'Hess Translation wrt Partial: {partial_first}|{partial_second}'
    return E