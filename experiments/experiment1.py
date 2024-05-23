#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:23:10 2024

@author: goresky
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import synth 
import compression as comp
import solvers as slv


data = scipy.io.loadmat('xGT.mat')
sensitivity = data['S']
ground_truth = data['xGT']
print(f'Sensitivity matrix is of size: {sensitivity.shape}, and Ground Truth is of size: {ground_truth.shape}')
print(f'Both have data type of {sensitivity.dtype} and {ground_truth.dtype} respectively')

ground_truth = np.atleast_3d(ground_truth)

#Data Synthesis
img_shape = ground_truth.shape

#Transform Generation
thetas = [2, 5, 10, 20]
#thetas = [10]
num_shots = 2
ground_transforms = synth.synthesize_transforms(num_shots, thetas, rand=False)
#returns a list of length number of thetas and each ele is of shape (6 2 1 1 1 1)

#grid generation
kgrid, kkgrid, rgrid, rkgrid = synth.generateGrids(img_shape)

#Encoding Methods
encoding_methods = synth.generate_encoding(img_shape, num_shots)
iso_filter = synth.generate_filter(img_shape, kgrid)
iso_filter = np.atleast_3d(iso_filter)
#Apply iso filter
ground_truth = np.fft.ifftn(iso_filter * np.fft.fftn(ground_truth,axes=(-3,-2)), axes=(-3,-2))

#Coil Array compression
S, _ = comp.coilArrayCompression(sensitivity, None, 0.99, 0)

#Data gen
y = synth.synthesize_Y(ground_truth, ground_transforms, S, encoding_methods, kgrid, kkgrid, rkgrid)
#y is a list of lenth theta with dictionaries of 5 methods as each ele

#RECONSUTRCTION

#Solver parameters
iter_joint     = 10_000
iter_conj_grad = 3
iter_newtons   = 1
winic          = 1
tolerance       = 1e-8

#Precon for GC
SH = np.conj(S)
reg = 0.001
precond = np.sum(np.real(SH*S), axis=0)
precond = np.power((precond + reg), -1)

#Spatial maks used to contrain the solution
M = np.ones(img_shape)

should_estimate_t = True
x_estimate = []
t_estimate = []
for v in range(len(thetas)):
    results = {}
    #for method, A in encoding_methods.items():
    for _ in range(1):
        method = "LinPar"
        A = encoding_methods[method]
        #Initialization
        x = np.zeros(ground_truth.shape) #(128 128 1)
        T = np.zeros(ground_transforms[v].shape) #(6 2 1 1 1 1)
        shape_T = T.shape
        w = winic*np.ones(shape_T[1:]) #shape(2 1 1 1 1)
        flag_w = np.zeros(shape_T[1:])
        flagw_prev = np.zeros(shape_T[1:])
        
        #Precomputations
        #y or measure is going to be shape (shot coil 128 128 1)
        yX = np.fft.ifftn(y[v][method], axes=(-3,-2))
        max_normalize = np.max(np.abs(yX))
        yIn = y[v][method] / max_normalize
        
        print(f'Solving for {num_shots} shots, theta={thetas[v]} and encoding {method}')
        
        for n in range(iter_joint):
            print(f'Solving iteration number: {n}', end='\n')
            xant = x
            x = slv.solve_x(x, yIn, M, T, S, SH, precond, A, kgrid, kkgrid, rkgrid, iter_conj_grad, 1)
            x=x[0]#need to see why extra dim is being added in solve x
            #Solve for T
            if should_estimate_t:
                flagw_prev = flag_w
                T, x, w, flag_w = slv.solve_t(x,yIn,M,T,S,A,kgrid,kkgrid,rgrid,rkgrid,iter_newtons,w,flag_w)
                w[w<1e-4] *= 2
                w[w>1e16] /= 2
                
            #Measure error
            xaux = x*max_normalize - ground_truth
            
            if should_estimate_t:
                pass
            
            #Check convergence
            xant = np.squeeze(x) - np.squeeze(xant)
            xant = np.real(xant * np.conj(xant))
            xant = np.max(xant)
            print(f'difference is: {xant}', end='\n')
            if xant < tolerance and np.sum(flagw_prev != 1) > 0:
                print(f'convergence has been reached at iteration {n+1}!')
                break
            elif (n == iter_joint-1):
                print('Convergence has NOT been reached!')
        results[method] = x * max_normalize
    x_estimate.append(results)
    if should_estimate_t:
        t_estimate.append(T)
        
            
        