# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:18:37 2024

@author: giuseppe
"""
import argparse
import numpy as np
import sigpy as sp
import sigpy.plot as pl
import os

import matplotlib.pyplot as plt

from estimation.utils import generate_corrupt_kspace, generate_sampling_mask, resample
from estimation.factors import generate_transform_grids
from estimation.joint_recon import MotionCorruptedImageRecon
from estimation.image_solver import ImageEstimation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="Clean image for experiment")
    parser.add_argument("mps", help="File for sensitivity maps if already generated")
    parser.add_argument("out", help="Output dir for the reconstructed image")
    parser.add_argument('transforms', help="File for the simulated transform parameters")
    args = parser.parse_args()

    #Experiment Setup | Synthetic data sample to be used
    img = np.load(args.img)
    mps = np.load(args.mps)
    transforms = np.load(args.transforms)
    #from sigpy.mri import sim
    #from scipy.ndimage import gaussian_filter
    #img_shape = [32, 32,32]
    #mps_shape = [10, 32, 32, 32]
    #img = sp.shepp_logan(img_shape)
    #img = gaussian_filter(img, 1)
    #mps = sim.birdcage_maps(mps_shape)
    num_shots = len(transforms)
    
    norm = np.linalg.norm(img)
    img /= norm
    mask = generate_sampling_mask(num_shots, img.shape)
    ksp = generate_corrupt_kspace(img, mps, mask, transforms)

    ksp_in = resample(ksp, 2, axes=[-3,-2], is_kspace=True)
    mps_in = resample(mps, 2, axes=[-3,-2], is_kspace=False)
    mask_in = resample(mask, 2, axes=[-3,-2], is_kspace=True)
    ground_truth = resample(img, 2, axes=[-3,-2])
    
    rss = np.sqrt(np.sum(np.abs(mps_in)**2, axis=0))
    M = rss > (np.max(rss) * 0.1)
    p = 1 / (np.sum(np.real(mps_in * mps_in.conj()), axis=0) + 1e-3)
    P = sp.linop.Multiply(mps_in.shape[1:], p)

    recon_four, estimates_ratio_four, objs, t_iter = MotionCorruptedImageRecon(ksp_in, 
                                                                 mps_in, 
                                                                 mask_in, 
                                                                 img.shape,
                                                                 constraint=M, 
                                                                 P=P, 
                                                                 tol=0, 
                                                                 max_joint_iter=4_000, 
                                                                 device=sp.Device(0), 
                                                                 save_objective_values=True).run()
    
    np.save(os.path.join(args.out, "objectives-resolution-lvl2"), objs)
    np.save(os.path.join(args.out, "params-resolution-lvl2"), t_iter)

    #SECOND LEVEL RESOLUTION PYRAMID
    #REDUCING BY A RATIO OF 2
    ksp_in = resample(ksp, 1, axes=[-3,-2], is_kspace=True)
    mps_in = resample(mps, 1, axes=[-3,-2], is_kspace=False)
    mask_in = resample(mask, 1, axes=[-3,-2], is_kspace=True)
    ground_truth = resample(img, 1, axes=[-3,-2])

    rss = np.sqrt(np.sum(np.abs(mps_in)**2, axis=0))
    M = rss > (np.max(rss) * 0.1)
    p = 1 / (np.sum(np.real(mps_in * mps_in.conj()), axis=0) + 1e-3)
    P = sp.linop.Multiply(mps_in.shape[1:], p)

    recon_two, estimates_ratio_two, objs, t_iter = MotionCorruptedImageRecon(ksp_in, 
                                                               mps_in, 
                                                               mask_in, 
                                                               img.shape, 
                                                               transforms=estimates_ratio_four, 
                                                               constraint=M, 
                                                               P=P, 
                                                               tol=0, 
                                                               max_joint_iter=1_000, 
                                                               device=sp.Device(0), 
                                                               save_objective_values=True).run()

    np.save(os.path.join(args.out, "objectives-resolution-lvl1"), objs)
    np.save(os.path.join(args.out, "params-resolution-lvl1"), t_iter)
    #FINAL LEVEL OF RESOLUTION PYRAMID
    #FULL RESOLUTION IMAGE ESTIMATION ONLY
    rss = np.sqrt(np.sum(np.abs(mps)**2, axis=0))
    M = rss > (np.max(rss) * 0.1)
    p = 1 / (np.sum(np.real(mps * mps.conj()), axis=0) + 1e-3)
    P = sp.linop.Multiply(mps.shape[1:], p)

    kgrid, _, rgrid, rkgrid = generate_transform_grids(img.shape, device=sp.Device(-1))

    final_recon = ImageEstimation(ksp, mps, mask, estimates_ratio_two.get(), kgrid, rkgrid, constraint=M, P=P, tol=1e-12, max_iter=100, device=sp.Device(-1)).run()

    error_mask = img - final_recon
    corrupted_img = np.sum(mps.conj() * sp.ifft(ksp, axes=[-3,-2,-1]), axis=0)
    mid = final_recon.shape[0] // 2 
    figure = np.concatenate([np.abs(img[mid]), np.abs(corrupted_img[mid]), np.abs(final_recon[mid]), np.abs(error_mask[mid])], axis=-1)

    np.save(os.path.join(args.out, "comparison_fig"), figure)
    np.save(os.path.join(args.out, "recon"), final_recon)