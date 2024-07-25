# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:18:37 2024

@author: giuseppe
"""

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.alg as alg
import sigpy.linop as linop
import sigpy.plot as pl
import argparse
import twixtools

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

class RigidTransform:
    def __init__(self, parameters, img_shape):
        self.parameters = parameters
        self.img_shape = img_shape
        
        self.k1, self.k2, self.k3 = self.generate_kgrid()
        self.r1, self.r2, self.r3 = self.generate_rgrid()
        self.rkgrid_tan, self.rkgrid_sin = self.generate_rkgrid()
        
        self.factors_tan = None
        self.factors_sin = None
        self.factors_u   = None
        self.update_factors(self.parameters)

    def Shear(self, factors, axis):
        return linop.Compose([linop.IFFT(self.img_shape, axes=(axis,)),
                              linop.Multiply(self.img_shape, factors),
                              linop.FFT(self.img_shape, axes=(axis,))])
    
    def Rotation(self, index):
        axis_one = ((index + 1) % 3) - 3
        axis_two = ((index + 2) % 3) - 3
        return linop.Compose([self.Shear(self.factors_tan[index], axis_one),
                              self.Shear(self.factors_sin[index], axis_two),
                              self.Shear(self.factors_tan[index], axis_one)])
    
    def Translation(self):
        return linop.Compose([linop.IFFT(self.img_shape, axes=(-1,-2,-3)),
                              linop.Multiply(self.img_shape, self.factors_u),
                              linop.FFT(self.img_shape, axes=(-1,-2,-3))])
    
    def Transform(self):
        U = self.Translation()
        R_one = self.Rotation(0)
        R_two = self.Rotation(1)
        R_three = self.Rotation(2)
        return U * R_one * R_two * R_three
    
    def generate_kgrid(self):
        sx, sy, sz = self.img_shape
        sxt = (-(sx//2), -(sx//-2)) if sx > 1 else (0,1)
        syt = (-(sy//2), -(sy//-2)) if sy > 1 else (0,1)
        szt = (-(sz//2), -(sz//-2)) if sz > 1 else (0,1)
        k1, k2, k3 = np.mgrid[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1]].astype(np.float64)
        k1 *= (2 * np.pi / sx) 
        k2 *= (2 * np.pi / sy) 
        k3 *= (2 * np.pi / sz) 
        return k1, k2, k3
    
    def generate_rgrid(self):
        sx, sy, sz = self.img_shape
        r1, r2, r3 = np.mgrid[:sx, :sy, :sz].astype(np.float64)
        centroid = tuple(map(lambda x: -(x//-2), self.img_shape))
        r1 -= centroid[0]
        r2 -= centroid[1]
        r3 -= centroid[2]
        return r1, r2, r3
    
    def generate_rkgrid(self):
        return ([self.k2 * self.r3, self.k3 * self.r1, self.k1 * self.r2],
            [self.k3 * self.r2, self.k1 * self.r3, self.k2 * self.r1] )
        
    def update_factors(self, parameters):
        #TODO accomodate for shape of [params, shots]
        t1, t2, t3  = parameters[:3]
        self.factors_u = t1 * self.k1 + t2 * self.k2 + t3 * self.k3
        self.factors_u = np.exp(-1j * self.factors_u)
        
        tan_rot = 1j * np.tan(parameters[3:]/2)
        sin_rot = -1j * np.sin(parameters[3:])
        
        self.factors_tan = [np.exp(tan_rot[idx] * self.rkgrid_tan[idx]) for idx in range(3)]
        self.factors_sin = [np.exp(sin_rot[idx] * self.rkgrid_sin[idx]) for idx in range(3)]
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Raw dat input file")
    parser.add_argument("output", help="Output file for the reconstructed image")
    parser.add_argument("--maps", help="File for sensitivity maps if already generated")
    parser.add_argument("--save_maps", help="Save the estimated maps to file for resuse")
    args = parser.parse_args()
    
    kspace_file = args.input
    dst_file = args.output
    
    kspace, refscan = get_kspaces(kspace_file)
    
    device = sp.Device(0)
    device.use()

    if args.maps is None:
        #Estimate maps from the low res refscan
        maps = mr.app.EspiritCalib(refscan, device=device)
    else:
        maps = np.load(args.maps)

    if args.save_maps is not None:
        np.save(args.save_maps, maps)
    
    img_shape = kspace.shape[1:]
    mask = np.where(kspace > 0, 1, 0)
    parameters = np.array([0,0,0,0,0,0], dtype=np.float64)
    parameters[3:] *= np.pi / 180
    rigid_transform = RigidTransform(parameters, img_shape)
    
    T = rigid_transform.Transform()
    S = linop.Multiply(img_shape, maps)
    F = linop.FFT(kspace.shape, axes=(-1,-2,-3))
    A = linop.Multiply(kspace.shape, mask)
    FWD = A * F * S * T
    BCK = T.H * S.H * F.H * A.H
    
    output = np.zeros(img_shape, dtype=np.complex64)
    sp.app.LinearLeastSquares(FWD, kspace, output, tol=1e-8).run()
    np.save(dst_file, output)