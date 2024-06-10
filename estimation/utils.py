# -*- coding: utf-8 -*-
from . import sinc_transforms as stforms
import numpy as np
import scipy.fft as fft

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