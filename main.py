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
import matplotlib.pyplot as plt

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
    xp = device.xp
    
    mps = np.load(args.maps)

    kgrid, kkgrid, rgrid, rkgrid = make_grids(image.shape, device)
    transforms = make_transforms([[0,0,0,2.5,0,0], [0,0,0,-2.5,0,0]])
    factors_trans, factors_tan, factors_sin = calc_factors(transforms, kgrid, rkgrid)
    num_shots = 2
    mask_shape = [num_shots] + [*image.shape]
    masks = np.zeros(mask_shape)
    masks[0,:, :, 0::2] = 1
    masks[1,:, :, 1::2] = 1
    masks.shape

    E = AlignedSense(image, mps, masks, factors_trans, factors_tan, factors_sin)
    corr_kspace = np.sum(E * image, axis=1)

    max_norm = np.max(np.abs(np.fft.ifftn(corr_kspace, axes=(-3,-2,-1))))
    corr_kspace /= max_norm

    recon = MotionCorruptedImageRecon(corr_kspace, mps, masks, tol=1e-6).run()
    pl.ImagePlot(recon * max_norm)
"""     gt = image / max_norm
    max_diff = []
    x_err = []
    obj_err = []
    t_err = []
    init = np.array([[-1.79575717e-04,  8.46043284e-06, -1.74125792e-04,
         4.34900082e-02,  2.10716159e-05, -4.37268242e-05],
       [ 1.79575717e-04, -8.46043284e-06,  1.74125792e-04,
        -4.34900082e-02, -2.10716159e-05,  4.37268242e-05]])
    alg = JointEstimation(mps, masks, corr_kspace, kgrid, kkgrid, rgrid, rkgrid, transforms=init, tol=1e-6, img_recon_iter=3, t_est_iter=1, max_iter=100)
    print(f'Starting Recon ...')
    print(f'Experiment | CG iter {args.img_recon_iter}, Newtons iter {args.t_est_iter}')
    while not alg.done():
        start = time.perf_counter()
        alg.update()
        #print(f'Iteration: {alg.iter} Transforms: {alg.transforms[:, 3:] * 180 / np.pi}')
        #print(f'Time Elapsed is {time.perf_counter() - start} s ... Error: {alg.xerr}')
        max_diff.append(alg.xerr.item())
        
        
        xaux = alg.x * max_norm - image
        x_err.append( np.sum(np.real(xaux * np.conj(xaux))) / (image.shape[0] * image.shape[1] * image.shape[2]) )

        factors_trans, factors_tan, factors_sin = calc_factors(alg.transforms, kgrid, rkgrid)
        E = AlignedSense(alg.x, mps, masks, factors_trans, factors_tan, factors_sin)
        oberr = (E*alg.x) - (masks * corr_kspace[:, None])
        obj_err.append(np.sum(np.real(oberr * np.conj(oberr))).item())

        t_err.append(np.max(np.abs(alg.transforms - transforms)).item())

        print(f'Iteration: {alg.iter} Transforms: {alg.transforms[:, :3]} {alg.transforms[:, 3:] * 180 / np.pi}')
        print(f'Time Elapsed is {time.perf_counter() - start} s ... Max diff: {alg.xerr} ... Obj err: {obj_err[-1]} ... xerr: {x_err[-1]} ... terr: {t_err[-1]}')
        interm = './data/synth/' + str(alg.iter) + '.png'
        mid_slice = int(image.shape[0]//2) 
        plt.imsave(interm, np.abs(alg.x[mid_slice,:,:]), cmap='gray')

    #pl.LinePlot(np.array(alg.FT_err), mode='l')            
    
    
    
    fig, axs = plt.subplots(2,2)
    fig.suptitle('Errors for 10 degrees 2 Shots')
    axs[0, 0].plot(np.log10(obj_err))
    axs[0, 0].set_title('Objective Func Log10')
    axs[0, 1].plot(max_diff, 'tab:orange')
    axs[0, 1].set_title('Max Voxel Diff')
    axs[1, 0].plot(x_err, 'tab:green')
    axs[1, 0].set_title('Image error to GT')
    axs[1, 1].plot(t_err, 'tab:red')
    axs[1, 1].set_title('Transform error to GT')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    
    plt.show()
    pl.ImagePlot((alg.x*max_norm))
    np.save(args.output, alg.x) """