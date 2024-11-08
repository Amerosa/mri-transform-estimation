# -*- coding: utf-8 -*-
#from . import sinc_transforms as stforms
import numpy as np
import scipy.fft as fft
import sigpy as sp

def error_fit(x, T, S, A, y, kgrid, rkgrid):
    et = stforms.precomp_sinc_transforms(kgrid, [], rkgrid, T)
    x = stforms.sinc_rigid_transform(x, et)
    #Temporary fix that needs to be refactored in the future
    x = x * S[np.newaxis]
    x = fft.fftn(x, axes=(-3,-2))
    x *= A
    y = y[np.newaxis] * A
    x -= y
    return np.sum(np.real(x*np.conj(x)))

def make_transforms(params, device=sp.cpu_device):
    xp = device.xp
    #if len(params) == 1:
    #    p = xp.array(params).reshape(1, -1)
    with device:
        p = xp.array(params).astype(xp.float64)              
        p[:, 3:] *= xp.pi / 180
    
    return p

def downsample_4d(array, factor):
    xs = array.shape[-3] // factor
    ys = array.shape[-2] // factor
    zs = array.shape[-1] // factor
    x_start = (array.shape[1] - xs) // 2
    y_start = (array.shape[2] - ys) // 2
    z_start = (array.shape[3] - zs) // 2
    return array[:, x_start:x_start+xs, y_start:y_start+ys,:]