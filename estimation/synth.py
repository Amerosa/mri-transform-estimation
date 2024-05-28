#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:15:59 2024

@author: Giuseppe Grossi
"""
import numpy as np
from . import sinc_transforms as st

def synthesize_T(num_shots, theta, rand_motion=True):
    transform = np.empty((len(num_shots), len(theta)))
    for (i, j), ele in np.ndenumerate(transform):
        ele = np.zeros((1,1,1,1,num_shots[i],6), dtype=np.float32)
        if len(num_shots) == 1 and num_shots[i] == 2 and not rand_motion:
            ele[0,0,0,0,0,3] = theta[j]*np.pi/180
        else:
            ele[...,:,3] = np.pi * theta[j] * (np.random.rand(1,1,1,1,num_shots[i]) - 0.5) / 180
        elem = np.mean(ele, axis=4)
        ele = ele - elem 

    return transform

def generate_spectral_grid(N):
    kGrid = [np.arange(-np.floor(N[m] / 2), np.ceil(N[m] / 2), dtype=np.float32) for m in range(3)]
    kGrid[0] = kGrid[0].reshape((len(kGrid[0]), 1))
    kGrid[1] = kGrid[1].reshape((1, len(kGrid[1])))
    #kGrid[2] = kGrid[2].reshape((1, 1, len(kGrid[2])))
    for m in range(3):
        kGrid[m] = 2 * np.pi * kGrid[m] / N[m]
    return kGrid

def generate_spectral_spectral_grid(kGrid):
    p1 = [0, 1, 2, 1, 2, 2]
    p2 = [0, 1, 2, 0, 0, 1]
    kkGrid = [kGrid[i] * kGrid[j] for i,j in zip(p1, p2)]
    return kkGrid

def generate_spatial_grid(N):
    cent = np.ceil(np.array(N) / 2).astype(int)
    rGrid = [np.arange(1, N[m]+1, dtype=np.float32) - cent[m] for m in range(3)]
    rGrid[0] = rGrid[0].reshape((len(rGrid[0]), 1))
    rGrid[1] = rGrid[1].reshape((1, len(rGrid[1])))
    #rGrid[2] = rGrid[2].reshape((1, 1, len(rGrid[2])))
    return rGrid

def generate_spatio_spectral_grid(rGrid, kGrid):
    per = np.array([[0,2,1], [1,0,2]])
    rkGrid = []
    for n in range(2):
        rkGrid.append([None] * 3)
        for m in range(3):
            rkGrid[n][m] = rGrid[per[1-n, m]] * kGrid[per[n, m]]
    return rkGrid

def generateGrids(N):
    kGrid = generate_spectral_grid(N)
    kkGrid = generate_spectral_spectral_grid(kGrid)
    rGrid = generate_spatial_grid(N)
    rkGrid = generate_spatio_spectral_grid(rGrid, kGrid)
    return kGrid, kkGrid, rGrid, rkGrid

def generate_filter(img_shape, kgrid):
    kk = np.empty((128,128,2))
    kk[...,0] = np.repeat(kgrid[0], img_shape[1], axis=1)
    kk[...,1] = np.repeat(kgrid[1], img_shape[0], axis=0)
    kkang = np.arctan(kk[...,1] / kk[...,0])
    kkrad = np.sqrt(np.sum(np.power(kk,2), axis=2))
    W = np.float32(kkrad < np.pi) 
    W = np.fft.ifftshift(W)
    return W
    
def generate_encoding(img_shape, num_shots):
    enc_mthd = dict.fromkeys(["LinSeq", "RadSeq", "LinPar", "LinSqu", "Random"], 
                #np.zeros( (img_shape[0], img_shape[1], 1, 1, num_shots), dtype=np.float32))
                np.zeros( (num_shots, 1, img_shape[0], img_shape[1], 1), dtype=np.float32))
    
    for method in enc_mthd.keys():
        
        if method == "LinPar":
            for shot in range(num_shots):
                #enc_mthd[method][:, shot::num_shots, 0, 0, shot] = 1
                enc_mthd[method][shot, 0, :, shot::num_shots, 0] = 1
                enc_mthd[method] = np.fft.ifftshift(enc_mthd[method], axes=(-3,-2))
    return enc_mthd
    
def synthesize_transforms(num_shots, thetas, rand=True):
    results = [None] * len(thetas)
    for i, theta in enumerate(thetas):
        transform = np.zeros((6, num_shots, 1, 1, 1, 1), dtype=np.single)
        if num_shots == 2 and rand == False:
            transform[3, 0, 0, 0, 0, 0] = theta*np.pi/180
        
        shots_mean = np.mean(transform, axis=1, keepdims=True)  
        results[i] = transform - shots_mean
    return results

def synthesize_Y(ground_truth, ground_transforms, S, methods, kgrid, kkgrid, rkgrid ):
    results = []
    for T in ground_transforms:
        et = st.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, T, direct=True)
        y = st.sinc_rigid_transform(ground_truth, et, direct=True)
        y = y * S                  #(shots 32 128 128 1)
        y = np.fft.fftn(y, axes=(-3, -2))
        d = {}
        for method, A in methods.items():
            d[method] = np.sum(y * A, axis=0)
        results.append(d)
    return results

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        