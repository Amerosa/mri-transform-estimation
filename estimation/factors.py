import numpy as np

def calc_factors(parameters, kgrid, rkgrid):
    trans = np.expand_dims(parameters[:, :3], axis=(-1, -2, -3)) 
    rots  = np.expand_dims(parameters[:, 3:], axis=(-1, -2, -3)) #[shots, rot, x, y, z] rot is scalar here xyz are all 1
    factors_trans = np.exp( -1j * (trans[:, 0] * kgrid[0] + trans[:, 1] * kgrid[1] + trans[:, 2] * kgrid[2])).astype(np.complex64) #[shots, x, y, z]
    tan_op = 1j * np.tan(rots/2)
    sin_op = -1j * np.sin(rots)
    factors_tan = []
    factors_sin = []
    for idx in range(3):
        factors_tan.append(np.exp(tan_op[:, idx] * rkgrid[0][idx]).astype(np.complex64))
        factors_sin.append(np.exp(sin_op[:, idx] * rkgrid[1][idx]).astype(np.complex64))

    #factors_tan = [np.fft.ifftshift(f, axes=(-3,-2,-1)) for f in factors_tan]
    #factors_sin = [np.fft.ifftshift(f, axes=(-3,-2,-1)) for f in factors_sin]
    #factors_trans = np.fft.ifftshift(factors_trans, axes=(-3,-2,-1))
    return factors_trans, factors_tan, factors_sin

def calc_derivative_factors(parameters, kgrid, kkgrid, rkgrid, factors_trans, factors_tan, factors_sin):
    trans = np.expand_dims(parameters[:, :3], axis=(-1, -2, -3))
    rots  = np.expand_dims(parameters[:, 3:], axis=(-1, -2, -3))
    tan_theta = np.tan(rots)
    tan_half_theta = np.tan(rots/2)
    tan_quad = (1 + tan_half_theta ** 2) / 2
    cos_theta = np.cos(rots)

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

def make_grids(shape):
    s = np.array(shape)
    lo = -(s//2)
    hi = -(s//-2)
    k1 = (np.linspace(lo[0], hi[0], s[0]) * 2 * np.pi / s[0]).reshape(-1, 1, 1)
    k2 = (np.linspace(lo[1], hi[1], s[1]) * 2 * np.pi / s[1]).reshape(1, -1, 1)
    k3 = (np.linspace(lo[2], hi[2], s[2]) * 2 * np.pi / s[2]).reshape(1, 1, -1)

    r1 = np.linspace(0, s[0], s[0]).reshape(-1, 1, 1) - hi[0]
    r2 = np.linspace(0, s[1], s[1]).reshape(1, -1, 1) - hi[1]
    r3 = np.linspace(0, s[2], s[2]).reshape(1, 1, -1) - hi[2]
    if s[2] == 1:
        r3 += hi[2]
    
    kgrid  = [k1, k2, k3]
    rgrid  = [r1, r2, r3]
    kkgrid = [[kgrid[i] * kgrid[j] for j in range(3)] for i in range(3)]
    rkgrid = [[k2 * r3, k3 * r1, k1 * r2], [k3 * r2, k1 * r3, k2 * r1]]

    return kgrid, kkgrid, rgrid, rkgrid