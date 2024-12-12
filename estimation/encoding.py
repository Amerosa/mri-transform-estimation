import sigpy as sp
import numpy as np
from .transforms import RigidTransform, GradRigidTransform
from .factors import calc_factors

def Encoding(params, mask, mps, kgrid, rkgrid, ishape=None, comm=None, shot_chunk_size=None):

    num_shots = params.shape[0]
    if ishape is None:
        ishape = mps.shape[1:]
        img_dim = mps.ndim - 1
    else:
        img_dim = len(ishape)
    
    
    if shot_chunk_size is None:
        shot_chunk_size = num_shots

    if shot_chunk_size < num_shots:
        num_shot_chunks = num_shots // shot_chunk_size
        A = sp.linop.Add(
            [
                Encoding(params[s * shot_chunk_size : (s+1) * shot_chunk_size],
                         mask[s * shot_chunk_size : (s+1) * shot_chunk_size],
                         mps,
                         kgrid,
                         rkgrid,
                         ishape=ishape,
                         comm=comm
                        )
                        for s in range(num_shot_chunks)
            ]
        )

        if comm is not None:
            C = sp.linop.AllReduce(A.oshape, comm, in_place=True)
            A = C * A

        return A
    
    device = sp.get_device(params)
    with device:
        ftr, ft, fs = calc_factors(params, kgrid, rkgrid)

    T = RigidTransform(ishape, ftr, ft,fs)
    R = sp.linop.Reshape((T.oshape[0], 1, *T.oshape[1:]), T.oshape)
    S = sp.linop.Multiply(R.oshape, mps)
    F = sp.linop.FFT(S.oshape, axes=range(-img_dim, 0))
    M = sp.linop.Multiply(F.oshape, mask[:, None])
    Sum = sp.linop.Sum(M.oshape, axes=(0,))
    A = Sum * M * F * S * R * T

    if comm is not None:
        C = sp.linop.AllReduce(A.oshape, comm, in_place=True)
        A = C * A

    A.repr_str = f"[Encoding Model {shot_chunk_size} Chunks]"
    return A

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