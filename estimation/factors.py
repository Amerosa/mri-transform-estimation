import numpy as xp
import sigpy as sp

def calc_factors(parameters, kgrid, rkgrid):
    xp = sp.get_device(parameters).xp
    trans = xp.expand_dims(parameters[:, :3], axis=(-1, -2, -3)) 
    rots  = xp.expand_dims(parameters[:, 3:], axis=(-1, -2, -3)) #[shots, rot, x, y, z] rot is scalar here xyz are all 1
    factors_trans = xp.exp( -1j * (trans[:, 0] * kgrid[0] + trans[:, 1] * kgrid[1] + trans[:, 2] * kgrid[2])).astype(xp.complex64) #[shots, x, y, z]
    tan_op = 1j * xp.tan(rots/2)
    sin_op = -1j * xp.sin(rots)
    factors_tan = []
    factors_sin = []
    for idx in range(3):
        factors_tan.append(xp.exp(tan_op[:, idx] * rkgrid[0][idx]).astype(xp.complex64))
        factors_sin.append(xp.exp(sin_op[:, idx] * rkgrid[1][idx]).astype(xp.complex64))

    return factors_trans, factors_tan, factors_sin

def calc_derivative_factors(parameters, kgrid, kkgrid, rkgrid, factors_trans, factors_tan, factors_sin):
    xp = sp.get_device(parameters).xp
    trans = xp.expand_dims(parameters[:, :3], axis=(-1, -2, -3))
    rots  = xp.expand_dims(parameters[:, 3:], axis=(-1, -2, -3))
    tan_theta = xp.tan(rots)
    tan_half_theta = xp.tan(rots/2)
    tan_quad = (1 + tan_half_theta ** 2) / 2
    cos_theta = xp.cos(rots)

    grad_factors = {}
    hess_factors = {}
    grad_factors['trans'] = [-1j * kgrid[i] * factors_trans for i in range(3)]
    grad_factors['tan'] = [1j * tan_quad[:, i] * rkgrid[0][i] * factors_tan[i] for i in range(3)]
    grad_factors['sin'] = [-1j * cos_theta[:, i] * rkgrid[1][i] * factors_sin[i] for i in range(3)]

    hess_factors['trans'] = [[-1 * kkgrid[i][j] * factors_trans for j in range(3)] for i in range(3)]
    hess_factors['tan'] = [(tan_half_theta[:, i] + 1j * tan_quad[:, i] * rkgrid[0][i]) * grad_factors['tan'][i] for i in range(3)]
    hess_factors['sin'] = [ -1 * (tan_theta[:, i] + 1j * cos_theta[:, i] * rkgrid[1][i]) * grad_factors['sin'][i] for i in range(3)]

    for gf in grad_factors.values():
        for factor in gf:
            factor = factor.astype(xp.complex64)

    return grad_factors, hess_factors

def make_grids(shape, device=sp.cpu_device):
    xp = device.xp
    s = xp.array(shape)
    lo = -(s//2)
    hi = -(s//-2)

    kgrid = [ (axis - shape[i] // 2) * 2 * xp.pi / shape[i] for i, axis in enumerate(xp.ogrid[0:s[0], 0:s[1], 0:s[2]])]
    k1, k2, k3 = kgrid

    r1 = xp.arange(shape[0]) if shape[0] == 1 else (xp.arange(shape[0]) - shape[0] //2 )
    r2 = xp.arange(shape[1]) if shape[1] == 1 else (xp.arange(shape[1]) - shape[1] //2 )
    r3 = xp.arange(shape[2]) if shape[2] == 1 else (xp.arange(shape[2]) - shape[2] //2 )
    r1 = r1.reshape(-1,1,1)
    r2 = r2.reshape(1,-1,1)
    r3 = r3.reshape(1,1,-1)

    rgrid  = [r1, r2, r3]
    kkgrid = [[kgrid[i] * kgrid[j] for j in range(3)] for i in range(3)]
    rkgrid = [[k2 * r3, k3 * r1, k1 * r2], [k3 * r2, k1 * r3, k2 * r1]]

    return kgrid, kkgrid, rgrid, rkgrid