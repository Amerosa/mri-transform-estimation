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
from estimation.factors import generate_transform_grids
from estimation.joint_recon import MotionCorruptedImageRecon
from estimation.image_solver import ImageEstimation

def sampling_mask(bins, img_shape, img_axis=1):
    num_shots = img_shape[img_axis] // bins
    sampling_mask = np.zeros((num_shots, *img_shape), dtype=bool)
    
    for shot in range(num_shots):
        idx = [shot] + [slice(None)] * len(img_shape)
        idx[1+img_axis] = slice(shot * bins, (shot+1) * bins)
        sampling_mask[tuple(idx)] = 1
    return sampling_mask

if __name__ == '__main__':

    # Load config file
    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    #Experiment Setup | Synthetic data sample to be used
    ksp = np.load(config["ksp"])
    mps = np.load(config["mps"])
    if config["transform"] is not None:
        init_transforms = np.load(config["transform"])
    else:
        init_transforms = None

    experiment_params = config["parameters"]
    num_channels = len(ksp)
    img_shape = ksp.shape[1:]
    bin_size = experiment_params["bin_size"]
    bin_axis = experiment_params["bin_axis"]

    #Axis indicies are "Par": 0, "Lin": 1, "Col": 2

    mask = sampling_mask(bin_size, img_shape, img_axis=bin_axis)
    num_shots = len(mask)
    device = sp.Device(experiment_params["device"])
    
    
    #Subsample input data based of resolution pyramid level if needed
    #ie level 2 gives a reduced size of 2^2 which is a factor of 4, 1 gives 2^1 = 2, etc...
    #Maps and images need to be fft first since we crop the middle square of ksapce
    #Kspace and boolean mask just need to be cropped
    if experiment_params["est_transform"]:
        resolution_factor = 2 ** experiment_params["res_scale"]
        low_resolution_shape = []
        for i in range(len(img_shape)):
            if img_shape[i] % (resolution_factor) != 0:
                raise ValueError(f"Image shape must be divisible by {resolution_factor}")
            if i != bin_axis:
                low_resolution_shape.append(img_shape[i] // resolution_factor)
            else:
                low_resolution_shape.append(img_shape[i])
        

        ksp = sp.resize(ksp, [num_channels] + low_resolution_shape)
        mask = sp.resize(mask, [num_shots] + low_resolution_shape)
        mps = sp.ifft(sp.resize(sp.fft(mps, axes=range(-3)), [num_channels] + low_resolution_shape))

        import sigpy.config
        sigpy.config.nccl_enabled = False # not sure if its working atm
        comm = sp.Communicator()

        rss = np.sqrt(np.sum(np.abs(mps)**2, axis=0))
        M = rss > (np.max(rss) * 0.1)
        p = 1 / (np.sum(np.real(mps * mps.conj()), axis=0) + 1e-3)
        P = sp.linop.Multiply(mps.shape[1:], p)


        kgrid, kkgrid, rgrid, rkgrid = generate_transform_grids(img_shape, low_resolution_shape, device)



        recon, estimates, objective, iter_data = MotionCorruptedImageRecon( ksp, mps, mask,
                                                                            kgrid, kkgrid, rgrid, rkgrid,
                                                                            transforms=init_transforms,
                                                                            constraint=M, P=P, tol=0, 
                                                                            max_joint_iter=experiment_params["max_iter"], 
                                                                            device=device,
                                                                            shot_chunk_size=experiment_params["shot_chunk_size"],
                                                                            comm=comm).run()
        
        iter_data_fname = f"iter_data_level_{experiment_params['res_scale']}_resolution"
        objective_fname = f"objective_level_{experiment_params['res_scale']}_resolution"
        recon_fname     = f"recon_level_{experiment_params['res_scale']}_resolution"
        estimates_fname = f"estimates_level_{experiment_params['res_scale']}_resolution"

        estimates = comm.gatherv(estimates, root=0)
        if comm.rank == 0:
            estimates = estimates.reshape((-1,6))
            xp = sp.get_array_module(estimates)
            xp.save(os.path.join(config["output_dir"], iter_data_fname), iter_data)
            xp.save(os.path.join(config["output_dir"], objective_fname), objective)
            xp.save(os.path.join(config["output_dir"], recon_fname), recon)
            xp.save(os.path.join(config["output_dir"], estimates_fname), estimates)
        else:
            estimates = np.zeros((num_shots, 6), dtype=float)
        comm.bcast(estimates, root=0)

    else:
    #FULL RESOLUTION IMAGE ESTIMATION ONLY
        rss = np.sqrt(np.sum(np.abs(mps)**2, axis=0))
        M = rss > (np.max(rss) * 0.1)
        p = 1 / (np.sum(np.real(mps * mps.conj()), axis=0) + 1e-3)
        P = sp.linop.Multiply(mps.shape[1:], p)

        kgrid, _, rgrid, rkgrid = generate_transform_grids(img_shape, device=device)
        init_transforms = sp.to_device(init_transforms, device)
        final_recon = ImageEstimation(ksp, mps, mask, init_transforms, kgrid, rkgrid, 
                                      constraint=M, P=P, 
                                      tol=1e-6, max_iter=100, device=device, shot_chunk_size=experiment_params["shot_chunk_size"]).run()

        xp = sp.get_array_module(final_recon)
        xp.save(os.path.join(config["output_dir"], "final_recon"), final_recon)