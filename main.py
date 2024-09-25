# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:18:37 2024

@author: giuseppe
"""
import argparse
import twixtools
from sigpy import backend  
from estimation.algos import JointEstimation
import numpy as np
import sigpy as sp
import time
import sigpy.plot as pl

from estimation import * 

def get_kspaces(filename):
    multi_twix = twixtools.read_twix(filename, parse_pmu=False)
    
    # map the twix data to twix_array objects
    mapped = twixtools.map_twix(multi_twix)
    mapped_img_data = mapped[-1]['image']
    mapped_refscan_data = mapped[-1]['refscan']
    
    # make sure that we later squeeze the right dimensions:
    print(f'Img Data non singleton dims : {mapped_img_data.non_singleton_dims}')
    print(f'RefScan Data non singleton dims : {mapped_refscan_data.non_singleton_dims}')
    
    # remove 2x oversampling and zero pad to ensure same shape
    mapped_img_data.flags['remove_os'] = True
    mapped_img_data.flags['zf_missing_lines'] = True
    mapped_refscan_data.flags['remove_os'] = True
    mapped_refscan_data.flags['zf_missing_lines'] = True
    
    
    image_ksp = mapped_img_data[:].squeeze()
    refscan_ksp = mapped_refscan_data[:].squeeze()
    print(f'Dimensions of image k-space is {image_ksp.shape}')
    print(f'Dimensions of refscan k-space is {refscan_ksp.shape}')
    
    #Rearrange so that shape follows format of [nc, col, par, line]
    image_ksp = np.transpose(image_ksp, (2, 3, 0, 1))
    refscan_ksp = np.transpose(refscan_ksp, (2, 3, 0, 1))
    print(f'New dimenions are [nc, kx, ky, kz] -> {refscan_ksp.shape}')

    return image_ksp, refscan_ksp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Clean image for experiment")
    parser.add_argument("output", help="Output file for the reconstructed image")
    parser.add_argument("maps", help="File for sensitivity maps if already generated")
    parser.add_argument("--bins", type=int, help="Number of linescans we transform estimate at once")
    parser.add_argument("--bin_size", type=int, help="Size of bins")
    args = parser.parse_args()
        
    #kspace, refscan = get_kspaces(kspace_file)
    image = np.load(args.input)
    device = sp.Device(-1)
    device.use()
    

    mps = np.load(args.maps)

    xp = device.xp
    image = backend.to_device(image, device)
    mps = backend.to_device(mps, device)

    kgrid, kkgrid, rgrid, rkgrid = make_grids(image.shape)
    transforms = make_transforms([[0,0,0,5,0,0], [0,0,0,-5,0,0]])
    factors_trans, factors_tan, factors_sin = calc_factors(transforms, kgrid, rkgrid)
    num_shots = 2
    mask_shape = [num_shots] + [*image.shape]
    masks = np.zeros(mask_shape)
    masks[0,:,:,0::2] = 1
    masks[1,:,:,1::2] = 1
    masks.shape

    E = AlignedSense(image, mps, masks, factors_trans, factors_tan, factors_sin)
    corr_kspace = np.sum(E * image, axis=1)
    #print(corr_kspace.shape)

    T = RigidTransform(image.shape, factors_trans, factors_tan, factors_sin)
    S = sp.linop.Multiply(corr_kspace.shape[1:], mps)
    F = sp.linop.FFT(corr_kspace.shape, axes=(-1,-2,-3))
    #pl.ImagePlot( (S.H*F.H * F * S * T *small_img)[...,0], z=0)
    #pl.ImagePlot( (S.H * F.H*E*small_img)[...,0], z=0)
    corr_img = S.H * F.H * corr_kspace
    corr_img = xp.sum(corr_img, axis=0)
    #print(corr_img.shape)
    #pl.ImagePlot(masks, z=0)
    pl.ImagePlot((T * image), z=0, title='Applied transforms for motion states')
    pl.ImagePlot(corr_img, 'Motion corrupted image')


    alg = JointEstimation(mps, masks, corr_kspace, kgrid, kkgrid, rgrid, rkgrid)
    print('Starting Recon ...')
    while not alg.done():
        start = time.perf_counter()
        alg.update()
        print(f'Iteration: {alg.iter} Transforms: {alg.transforms[:, 3:] * 180 / np.pi}')
        print(f'Time Elapsed is {time.perf_counter() - start} s ... Error: {alg.xerr}')
        
        
    xp.save(args.output, alg.x)