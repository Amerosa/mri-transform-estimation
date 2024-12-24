# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:18:37 2024

@author: giuseppe
"""
import numpy as np
import sigpy as sp
import os
import yaml
import sys

from estimation.utils import resample
from estimation.factors import generate_transform_grids
from estimation.joint_recon import MotionCorruptedImageRecon
from estimation.image_solver import ImageEstimation

if __name__ == '__main__':

    # Load config file
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    #Experiment Setup | Synthetic data sample to be used
    ksp = np.load(config["ksp"])
    mps = np.load(config["mps"])
    mask = np.load(config["sampling_mask"])

    num_shots = len(mask)
    img_shape = ksp.shape[1:]

    import sigpy.config
    sigpy.config.nccl_enabled = False # not sure if its working atm
    comm = sp.Communicator()

    ksp_in = resample(ksp, 2, axes=[-3,-2], is_kspace=True)
    mps_in = resample(mps, 2, axes=[-3,-2], is_kspace=False)
    mask_in = resample(mask, 2, axes=[-3,-2], is_kspace=True)
    
    rss = np.sqrt(np.sum(np.abs(mps_in)**2, axis=0))
    M = rss > (np.max(rss) * 0.1)
    p = 1 / (np.sum(np.real(mps_in * mps_in.conj()), axis=0) + 1e-3)
    P = sp.linop.Multiply(mps_in.shape[1:], p)

    recon_four, estimates_ratio_four, objs, t_iter = MotionCorruptedImageRecon(ksp_in, 
                                                                 mps_in, 
                                                                 mask_in, 
                                                                 img_shape,
                                                                 constraint=M, 
                                                                 P=P, 
                                                                 tol=0, 
                                                                 max_joint_iter=config["parameters"]["level_2_iter"], 
                                                                 device=sp.Device(0), 
                                                                 save_objective_values=True,
                                                                 shot_chunk_size=config["parameters"]["shot_chunk_size"],
                                                                 comm=comm).run()
    estimates_ratio_four = comm.gatherv(estimates_ratio_four, root=0)
    if comm.rank == 0:
        estimates_ratio_four = estimates_ratio_four.reshape((-1,6))
        xp = sp.get_array_module(estimates_ratio_four)
        xp.save(os.path.join(config["output_dir"], "estimates_level_2_resolution"), t_iter)
        xp.save(os.path.join(config["output_dir"], "objective_level_2_resolution"), objs)
    else:
        estimates_ratio_four = np.zeros((num_shots, 6), dtype=float)
    comm.bcast(estimates_ratio_four, root=0)

    #SECOND LEVEL RESOLUTION PYRAMID
    #REDUCING BY A RATIO OF 2
    ksp_in = resample(ksp, 1, axes=[-3,-2], is_kspace=True)
    mps_in = resample(mps, 1, axes=[-3,-2], is_kspace=False)
    mask_in = resample(mask, 1, axes=[-3,-2], is_kspace=True)

    rss = np.sqrt(np.sum(np.abs(mps_in)**2, axis=0))
    M = rss > (np.max(rss) * 0.1)
    p = 1 / (np.sum(np.real(mps_in * mps_in.conj()), axis=0) + 1e-3)
    P = sp.linop.Multiply(mps_in.shape[1:], p)

    recon_two, estimates_ratio_two, objs, t_iter = MotionCorruptedImageRecon(ksp_in, 
                                                               mps_in, 
                                                               mask_in, 
                                                               img_shape, 
                                                               transforms=estimates_ratio_four, 
                                                               constraint=M, 
                                                               P=P, 
                                                               tol=0, 
                                                               max_joint_iter=config["parameters"]["level_1_iter"], 
                                                               device=sp.Device(0), 
                                                               save_objective_values=True,
                                                               shot_chunk_size=config["parameters"]["shot_chunk_size"],
                                                               comm=comm).run()

    estimates_ratio_two = comm.gatherv(estimates_ratio_two, root=0)
    if comm.rank == 0:
        estimates_ratio_two = estimates_ratio_two.reshape((-1,6))
        xp = sp.get_array_module(estimates_ratio_two)
        xp.save(os.path.join(config["output_dir"], "estimates_level_1_resolution"), t_iter)
        xp.save(os.path.join(config["output_dir"], "objective_level_1_resolution"), objs)
    else:
        estimates_ratio_two = np.zeros((num_shots, 6), dtype=float)
    comm.bcast(estimates_ratio_two, root=0)
    
    #FINAL LEVEL OF RESOLUTION PYRAMID
    #FULL RESOLUTION IMAGE ESTIMATION ONLY
    rss = np.sqrt(np.sum(np.abs(mps)**2, axis=0))
    M = rss > (np.max(rss) * 0.1)
    p = 1 / (np.sum(np.real(mps * mps.conj()), axis=0) + 1e-3)
    P = sp.linop.Multiply(mps.shape[1:], p)

    kgrid, _, rgrid, rkgrid = generate_transform_grids(img_shape, device=sp.Device(-1))

    final_recon = ImageEstimation(ksp, mps, mask, estimates_ratio_two.get(), kgrid, rkgrid, constraint=M, P=P, tol=1e-12, max_iter=10, device=sp.Device(-1), shot_chunk_size=1).run()

    np.save(os.path.join(config["output_dir"], "final_recon"), final_recon)