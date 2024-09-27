# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:18:37 2024

@author: giuseppe
"""
import argparse
import numpy as np
import sigpy as sp
import time
import sigpy.plot as pl

from estimation import * 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Clean image for experiment")
    parser.add_argument("output", help="Output file for the reconstructed image")
    parser.add_argument("maps", help="File for sensitivity maps if already generated")
    parser.add_argument("--bins", type=int, help="Number of linescans we transform estimate at once")
    parser.add_argument("--bin_size", type=int, help="Size of bins")
    parser.add_argument("--img_recon_iter", type=int, help="Number of iterations for image recon subproblem")
    parser.add_argument("--t_est_iter", type=int, help="Number of iterations for transform estimation subproblem")
    args = parser.parse_args()
        
    #kspace, refscan = get_kspaces(kspace_file)
    image = np.load(args.input)
    device = sp.Device(-1)
    device.use()
    
    mps = np.load(args.maps)

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

    #T = RigidTransform(image.shape, factors_trans, factors_tan, factors_sin)
    #S = sp.linop.Multiply(corr_kspace.shape[1:], mps)
    #F = sp.linop.FFT(corr_kspace.shape, axes=(-1,-2,-3))
    #pl.ImagePlot( (S.H*F.H * F * S * T *small_img)[...,0], z=0)
    #pl.ImagePlot( (S.H * F.H*E*small_img)[...,0], z=0)
    #corr_img = S.H * F.H * corr_kspace
    #corr_img = xp.sum(corr_img, axis=0)
    #print(corr_img.shape)
    #pl.ImagePlot(masks, z=0)
    #pl.ImagePlot((T * image), z=0, title='Applied transforms for motion states')
    #pl.ImagePlot(corr_img, title='Motion corrupted image')


    alg = JointEstimation(mps, masks, corr_kspace, kgrid, kkgrid, rgrid, rkgrid, img_recon_iter=args.img_recon_iter, t_est_iter=args.t_est_iter)
    print(f'Starting Recon ...')
    print(f'Experiment | CG iter {args.img_recon_iter}, Newtons iter {args.t_est_iter}')
    while not alg.done():
        start = time.perf_counter()
        alg.update()
        print(f'Iteration: {alg.iter} Transforms: {alg.transforms[:, 3:] * 180 / np.pi}')
        print(f'Time Elapsed is {time.perf_counter() - start} s ... Error: {alg.xerr}')
                
    np.save(args.output, alg.x)