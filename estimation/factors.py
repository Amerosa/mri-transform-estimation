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

    #factors_tan = [xp.fft.ifftshift(f, axes=(-3,-2,-1)) for f in factors_tan]
    #factors_sin = [xp.fft.ifftshift(f, axes=(-3,-2,-1)) for f in factors_sin]
    #factors_trans = xp.fft.ifftshift(factors_trans, axes=(-3,-2,-1))
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

    """
    for i in range(3):
        grad_factors['trans'][i] = xp.fft.ifftshift(grad_factors['trans'][i], axes=(-3,-2,-1))
        grad_factors['tan'][i] = xp.fft.ifftshift(grad_factors['tan'][i], axes=(-3,-2,-1))
        grad_factors['sin'][i] = xp.fft.ifftshift(grad_factors['sin'][i], axes=(-3,-2,-1))

        hess_factors['tan'][i] = xp.fft.ifftshift(hess_factors['tan'][i], axes=(-3,-2,-1))
        hess_factors['sin'][i] = xp.fft.ifftshift(hess_factors['sin'][i], axes=(-3,-2,-1))

        for j in range(3):
            hess_factors['trans'][i][j] = xp.fft.ifftshift(hess_factors['trans'][i][j], axes=(-3,-2,-1))
    """

    return grad_factors, hess_factors

def make_grids(shape, device=sp.cpu_device):
    xp = device.xp
    s = xp.array(shape)
    lo = -(s//2)
    hi = -(s//-2)
    k1 = (xp.linspace(lo[0], hi[0], s[0]) * 2 * xp.pi / s[0]).reshape(-1, 1, 1)
    k2 = (xp.linspace(lo[1], hi[1], s[1]) * 2 * xp.pi / s[1]).reshape(1, -1, 1)
    k3 = (xp.linspace(lo[2], hi[2], s[2]) * 2 * xp.pi / s[2]).reshape(1, 1, -1)

    r1 = xp.arange(shape[0]) if shape[0] == 1 else (xp.arange(shape[0]) - shape[0] //2 )
    r2 = xp.arange(shape[1]) if shape[1] == 1 else (xp.arange(shape[1]) - shape[1] //2 )
    r3 = xp.arange(shape[2]) if shape[2] == 1 else (xp.arange(shape[2]) - shape[2] //2 )
    r1 = r1.reshape(-1,1,1)
    r2 = r2.reshape(1,-1,1)
    r3 = r3.reshape(1,1,-1)


    kgrid  = [k1, k2, k3]
    rgrid  = [r1, r2, r3]
    kkgrid = [[kgrid[i] * kgrid[j] for j in range(3)] for i in range(3)]
    rkgrid = [[k2 * r3, k3 * r1, k1 * r2], [k3 * r2, k1 * r3, k2 * r1]]

    return kgrid, kkgrid, rgrid, rkgrid