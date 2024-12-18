import sigpy as sp
from .transforms import RigidTransform
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
            C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
            A = A * C

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
        C = sp.linop.AllReduceAdjoint(ishape, comm, in_place=True)
        A = A * C
    
    A.repr_str = f"[Encoding Model {shot_chunk_size} Chunks]"
    return A