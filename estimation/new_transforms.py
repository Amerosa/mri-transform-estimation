#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:15:59 2024

@author: Giuseppe Grossi
"""
from functools import partial, reduce 
import numpy as np

import scipy.fft as fft

# u which is one array
# u' which is three arrays
# u'' which is [2, 3] array

#v_tan v_tan' v_tan'' which are each 3 arrays

def precomp_factors(kgrid, kkgrid, rkgrid, T, direct=True, compute_grads=False):
    #rotations degress is a 6x1 np array
    #index 0-2 are the translations adn 3-5 are rotations, 6 params in total
    theta = T[3:]
    tan_theta2 = np.tan(theta/2)
    if direct:
        t = -1j * T[:3]
        tan_theta2j = 1j * tan_theta2
        sin_theta = -1j * np.sin(theta)
    else:
        t = 1j * T[:3]
        tan_theta2j = -1j * tan_theta2
        sin_theta = 1j * np.sin(theta)
        
    kgrid = [np.atleast_3d(ele) for ele in kgrid]

    v_tans = np.exp(np.expand_dims(tan_theta2j, (-1,-2,-3) ) * np.atleast_3d(rkgrid[0][0]))#(3d coordinates by num of transforms)
    v_sins = np.exp(np.expand_dims(sin_theta, (-1,-2,-3))  * np.atleast_3d(rkgrid[1][0]))

    u = np.exp(t[0]*kgrid[0] + t[1]*kgrid[1] + t[2]*kgrid[2])
    
    perm = np.array([[1,3,2], [2,1,3]]) - 1
    #Keeping the permutation indices in human readable from to align wiht paper
    #Subtracting one to make it zero based indexing
    for i, precomp in enumerate(v_tans):
        v_tans[i,...] = fft.ifftshift(precomp, perm[0,i])
        
    for i, precomp in enumerate(v_sins):
            v_tans[i,...] = fft.ifftshift(precomp, perm[1,i])
    
    u = fft.ifftshift(u)
    
    return u, v_tans, v_sins

def precomp_deriv_factors():
    if compute_grads:
        tan_theta = np.tan(theta)
        tan_theta_cuad = (1+ tan_theta * tan_theta) / 2
        cos_theta = np.cos(theta)
        
    
    if compute_grads:
        etg["tans"] = [tan_theta_cuad[i] * 1j * np.atleast_3d(rkgrid[0][i]) for i in range(3)]
        etg["sins"] = [cos_theta[i] * -1j * np.atleast_3d(rkgrid[1][i]) for i in range(3)]
        
        eth["tans"] = [tan_theta2[i] + etg["tans"][i] for i in range(3)]
        eth["sins"] = [-tan_theta[i] + etg["sins"][i] for i in range(3)]
        
        for k, sinusoid in enumerate(["tans" , "sins"]):
            for i in range(3):
                etg[sinusoid][i] *= et[sinusoid][i]
                eth[sinusoid][i] *= etg[sinusoid][i]
                
                etg[sinusoid][i] = fft.ifftshift(np.atleast_3d(etg[sinusoid][i]), axes=perm[k,i])
                eth[sinusoid][i] = fft.ifftshift(np.atleast_3d(eth[sinusoid][i]), axes=perm[k,i])
                
            if compute_grads:
                etg["trans"] = [-1j * kgrid[i] * et["trans"] for i in range(3)]
                etg["trans"] = [fft.ifftshift(ele, axes=img_axis) for ele in etg["trans"]]
                
                eth["trans"] = [-kkgrid[i] * et["trans"] for i in range(6)]
                eth["trans"] = [fft.ifftshift(ele, axes=img_axis) for ele in eth["trans"]]
        

def precomp_second_deriv_factors():
    pass

#u, v_tans, v_sins = precomp_factors()

def calc_inidices(index):
    return (index-1) % 3

def shear(factors, axis, img):
    img = fft.fft(img, axis=axis)
    img = img * factors
    img = fft.ifft(img, axis=axis)
    return img

def rotation_base(index, v_tan, v_sin, img):
    l_plus_one = calc_inidices(index+1)
    l_plus_two = calc_inidices(index+2)
    
    img = shear(v_tan, l_plus_one, img)
    img = shear(v_sin, l_plus_two, img)
    img = shear(v_tan, l_plus_one, img)

    return img

def rotation_base_d(index, img):
    l_plus_one = calc_inidices(index+1)
    l_plus_two = calc_inidices(index+2)
    
    v_tan, dv_tan, _ = v_tans[index-1]
    v_sin, dv_sin, _ = v_sins[index-1]
    
    img = fft.fft(img, axis=l_plus_one)
    
    summand_1 = img * dv_tan
    summand_1 = fft.ifft(summand_1, axis=l_plus_one)
    summand_1 = shear(v_sin,l_plus_two, summand_1)
    summand_1 = fft.fft(summand_1, axis=l_plus_one)
    summand_1 = summand_1 * v_tan
    
    summand_2 = img * v_tan
    summand_2 = fft.ifft(summand_2, axis=l_plus_one)
    summand_2 = shear(dv_sin,l_plus_two, summand_2)
    summand_2 = fft.fft(summand_2, axis=l_plus_one)
    summand_2 = summand_2 * v_tan
    
    summand_3 = img * dv_tan
    summand_3 = fft.ifft(summand_3, axis=l_plus_one)
    summand_3 = shear(v_sin,l_plus_two, summand_3)
    summand_3 = fft.fft(summand_3, axis=l_plus_one)
    summand_3 = summand_3 * dv_tan
    
    img = img * (summand_1 + summand_2 + summand_3)
    img = fft.ifft(img, axis=l_plus_one)

    return img

def rotation_base_dd(index, img):
    l_plus_one = calc_inidices(index+1)
    l_plus_two = calc_inidices(index+2)
    
    v_tan, dv_tan, ddv_tan = v_tans[index-1]
    v_sin, dv_sin, ddv_sin = v_sins[index-1]
    
    
    img = fft.fft(img, axis=l_plus_one)
    
    summand_1 = img * dv_tan
    summand_1 = fft.ifft(summand_1, axis=l_plus_one)
    summand_1 = shear(v_sin,l_plus_two, summand_1)
    summand_1 = fft.fft(summand_1, axis=l_plus_one)
    summand_1 = 2*summand_1 * dv_tan
    
    summand_2 = img * dv_tan
    summand_2 = fft.ifft(summand_2, axis=l_plus_one)
    summand_2 = shear(dv_sin,l_plus_two, summand_2)
    summand_2 = fft.fft(summand_2, axis=l_plus_one)
    summand_2 = 2 * summand_2 * v_tan
    
    summand_3 = img * v_tan
    summand_3 = fft.ifft(summand_3, axis=l_plus_one)
    summand_3 = shear(dv_sin,l_plus_two, summand_3)
    summand_3 = fft.fft(summand_3, axis=l_plus_one)
    summand_3 = 2 * summand_3 * dv_tan
    
    summand_4 = img * ddv_tan
    summand_4 = fft.ifft(summand_4, axis=l_plus_one)
    summand_4 = shear(v_sin,l_plus_two, summand_4)
    summand_4 = fft.fft(summand_4, axis=l_plus_one)
    summand_4 = summand_4 * v_tan
    
    summand_5 = img * v_tan
    summand_5 = fft.ifft(summand_5, axis=l_plus_one)
    summand_5 = shear(ddv_sin,l_plus_two, summand_5)
    summand_5 = fft.fft(summand_5, axis=l_plus_one)
    summand_5 = summand_5 * v_tan
    
    summand_6 = img * v_tan
    summand_6 = fft.ifft(summand_6, axis=l_plus_one)
    summand_6 = shear(v_sin,l_plus_two, summand_6)
    summand_6 = fft.fft(summand_6, axis=l_plus_one)
    summand_6 = summand_6 * ddv_tan

    img = img * (summand_1 + summand_2 + summand_3 +summand_4 +
                 summand_5 + summand_6)
    img = fft.ifft(img, axis=l_plus_one)

def translation_base(factors, img):
    img = fft.fftn(img)
    img = img * factors
    img = fft.ifftn(img)

    return img

    
first_rotation  = partial(rotation_base, index=1)
second_rotation = partial(rotation_base, index=2)
third_rotation  = partial(rotation_base, index=3)

first_rotation_deriv   = partial(rotation_base_d, index=1)
second_rotation_deriv  = partial(rotation_base_d, index=2)
third_rotation_deriv   = partial(rotation_base_d, index=3)

first_rotation_second_deriv     = partial(rotation_base_dd, index=1)
second_rotation_second_deriv    = partial(rotation_base_dd, index=2)
third_rotation_second_deriv     = partial(rotation_base_dd, index=3)

#translation = partial(translation_base, u)

def translation(factors):
    return partial(translation_base, factors)

def first_ord_translation(index, img):
    return partial(translation_base, first_partial_u[index])

def second_ord_translation(first_index, second_index, img):
    return partial(translation_base, second_partial_u[first_index][second_index])


def compose(img, *functions):
    return reduce(lambda x, f: f(x), reversed(functions), img)

def transform(u, img):
    """Where z=(z1, z2,z3,z4,z5,z6) the first 3 being translation params
    and the last three being the rotation params for each axis"""
    return compose(img, 
                  [translation(u), 
                   first_rotation, 
                   second_rotation, 
                   third_rotation])
    
def jacobian_transform(index, img):
    if index in [1, 2, 3]:
        return compose(img, 
                       [first_ord_translation(index),
                        first_rotation,
                        second_rotation,
                        third_rotation])
    elif index == 4:
        return compose(img, 
                       [translation,
                        first_rotation_deriv,
                        second_rotation,
                        third_rotation])
    elif index == 5:
        return compose(img, 
                       [first_ord_translation,
                        first_rotation,
                        second_rotation_deriv,
                        third_rotation])
    elif index == 6:
        return compose(img, 
                       [first_ord_translation,
                        first_rotation,
                        second_rotation,
                        third_rotation_deriv])
    else:
        raise Exception("Index for calculating Jacobian Transform is wrong") 
    
    
def hessian_transform(first_index, second_index, img):
    if 1 <= first_index <= second_index <= 3:
        return compose(img, 
                       [second_ord_translation(first_index, second_index),
                        first_rotation,
                        second_rotation,
                        third_rotation])
    elif 1 <= first_index <= 3 and second_index == 4:
        return compose(img, 
                       [first_ord_translation(first_index),
                        first_rotation_deriv,
                        second_rotation,
                        third_rotation])
    elif 1 <= first_index <= 3 and second_index == 5:
        return compose(img, 
                       [first_ord_translation(first_index),
                        first_rotation,
                        second_rotation_deriv,
                        third_rotation])
    elif 1 <= first_index <= 3 and second_index == 6:
        return compose(img, 
                       [first_ord_translation(first_index),
                        first_rotation,
                        second_rotation,
                        third_rotation_deriv])
    elif first_index == second_index == 4:
        return compose(img,
                       [translation,
                        first_rotation_second_deriv,
                        second_rotation,
                        third_rotation])
    elif first_index == 4 and second_index == 5:
        return compose(img,
                       [translation,
                        first_rotation_deriv,
                        second_rotation_deriv,
                        third_rotation])
    elif first_index == 4 and second_index == 6:
        return compose(img,
                       [translation,
                        first_rotation_deriv,
                        second_rotation,
                        third_rotation_deriv])
    elif first_index == second_index == 5:
        return compose(img,
                       [translation,
                        first_rotation,
                        second_rotation_second_deriv,
                        third_rotation])
    elif first_index == 5 and second_index == 6:
        return compose(img,
                       [translation,
                        first_rotation,
                        second_rotation_deriv,
                        third_rotation_deriv])
    elif first_index == second_index == 6:
        return compose(img,
                       [translation,
                        first_rotation,
                        second_rotation,
                        third_rotation_second_deriv])
    else:
        raise Exception("Incorrect Combination of indices for Hessian computation!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

