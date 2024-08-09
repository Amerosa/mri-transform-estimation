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
    parser.add_argument("--input", help="Raw dat input file")
    parser.add_argument("--output", help="Output file for the reconstructed image")
    parser.add_argument("--maps", help="File for sensitivity maps if already generated")
    parser.add_argument("--save_maps", help="Save the estimated maps to file for resuse")
    parser.add_argument("--device_id", help="Integer for the device, -1 for cpu other for gpu")
    args = parser.parse_args()
    
    kspace_file = args.input
    dst_file = args.output
    
    kspace, refscan = get_kspaces(kspace_file)
    device = sp.Device(args.device_id)
    device.use()
    
    if args.maps is None:
        #Estimate maps from the low res refscan
        maps = sp.mri.app.EspiritCalib(refscan, device=device)
    else:
        maps = np.load(args.maps)

    if args.save_maps is not None:
        np.save(args.save_maps, maps)

    xp = device.xp
    kspace = backend.to_device(kspace, device)
    maps = backend.to_device(maps, device)

    alg = JointEstimation(kspace, maps, 3, 1, 100)
    while not alg.done():
        print(f'Join Iteration number {alg.iter + 1 }')
        alg.update()
    xp.save(alg.recon_image, dst_file)