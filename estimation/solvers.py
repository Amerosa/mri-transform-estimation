#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:15:59 2024

@author: Giuseppe Grossi
"""

#import numpy as np
import numpy as np
import scipy.fft as fft

from . import sinc_transforms as stform
from . import isense as sense



def solve_x(X, Y, M, T, S, SH, precond, A, kgrid, kkgrid, rkgrid, n, block_size):
    
    #x -> 128 128 1
    #y -> shots coils 128 128 1 
    #M -> 128 128
    #S -> 128 128 1 coils
    #Sh -> S
    #precond is number
    #A shot 1 128 128 1
    #T 6 2 1 1 1 1 intially all zeros
    
    nx = np.atleast_3d(X).shape #(128, 128 1)
    nt = T.shape #Assums this is (num shots transforms)
    ny = Y.shape #(shots coils 128 128 1)
    ns = S.shape #(coils 128 128 1)
    
    num_runs = 1 

    et_dir = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, T, direct=True)
    et_inv = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, T, direct=False)
    
    #SENSE undersampling parameters, only for factors lower than 3
    FOV = np.ones((2,2), dtype=np.single)
    iFOV = np.ones((2,2), dtype=np.single)
    for i in range(-3, -1, 1):
        if ns[i] != ny[i]:
            disc = 3*ny[i] - ns[i]
            iFOV[i, 0] = np.floor(disc/2)
            iFOV[i, 1] = np.ceil(disc/2)
            over = ns[i] - ny[i]
            FOV[i, 0] = np.floor(over/2)
            FOV[i, 1] = np.ceil(over/2)
            
    
    #balanace high frequ harmonics in recon for iso resolution
    
    #Sampling amtrix A is of shape (128 128 1 1 shot)
    Atot = 1 - np.sum(A, axis=0)
    #nt shape is going to be 6 2 1 1 1 1 
    Atot = Atot/nt[1] #(1 128 128 1) div on number of shots
    A = A + Atot #(2 1 128 128 1) + (1 128 128 1)
    
    
    y_end = np.zeros(nx, dtype=np.complex128)
    
    for s in range(num_runs):
        #Inverse transform pre calculations
        
        #Apply sampling inverse matrix
        yS = Y * A
        
        #Apply inverse fourier transform matrix
        yS = np.fft.ifftn(yS, axes=(-3,-2))
        #Apply inverse coils sense matrix
        for m in range(-3, -1, 1):
            yS = sense.isense(yS, m, ns[m], ny[m], iFOV[m+3,:])
        #(2 11 128 128 1)
        yS = np.sum(yS * SH, axis=1, keepdims=True) # sum across coils to mage one image
        #Apply inverse transform matrix
        y_end = y_end + stform.sinc_rigid_transform(yS, et_inv, direct=False) #FIX THIS
        
    y = y_end
    y = M * y
    
    Ap = apply_cg(X, nx, num_runs, et_dir, et_inv, S, ns, ny, FOV, A, iFOV, SH, M)
    r = y - Ap
    z = precond * r
    p = z
    rs_old = np.sum(np.conj(z) * r)
    
    
    for _ in range(n):
        Ap = apply_cg(p, nx, num_runs, et_dir, et_inv, S, ns, ny, FOV, A, iFOV, SH, M)
        al = np.conj(rs_old) / np.sum(np.conj(p) * Ap)
        X = X + al*p
        r = r - al*Ap
        z = precond * r
        rs_new = np.sum(np.conj(z) * r)
        be = rs_new / rs_old
        p = z + be*p
        rs_old = rs_new
        if np.sqrt(np.abs(rs_new)) < 1e-10:
            break
    return X
        
def apply_cg(X, nx, num_runs, et_dir, et_inv, S, ns, ny, FOV, A, iFOV, SH, M):
    x_end = np.zeros(nx)
    for s in range(num_runs):
        #Tx
        xS = stform.sinc_rigid_transform(X, et_dir, direct=True)
        #STx
        xS = xS * S 
        #FSTx
        for m in range(-3, -1, 1):
            xS = sense.sense(xS, m, ns[m], ny[m], FOV[m+3,:])
        
        xS = np.fft.fftn(xS, axes=(-3,-2))
        
        #A'AFSTx
        xS = xS * A
        
        #F'A'AFSTx
        xS = np.fft.ifftn(xS, axes=(-3,-2))
        
        for m in range(-3, -1, 1):
            xS = sense.isense(xS, m, ns[m], ny[m], iFOV[m+3,:])
        
        #S'F'A'AFSTx
        xS = np.sum(xS * SH, axis=1, keepdims=True)
        
        #T'S'F'A'AFSTx
        x_end = x_end + stform.sinc_rigid_transform(xS, et_inv, direct=False)
    
    X = x_end
    return X * M
        
    
def solve_t(x, y, M, T, S, A, kgrid, kkgrid, rgrid, rkgrid, num_iter, w, flag_w, mean_t=1):
    div_weight_regularizer = 1.2
    mul_weight_regularizer = 2.0
    winic = w
    flag_winic = flag_w
    t_up = T
    
    shape_x = np.atleast_3d(x).shape
    shape_y = y.shape
    shape_s = S.shape
    shape_t = T.shape
    
    #       Second order derivatives
    sods = np.array([[1, 2, 3, 1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 5, 4, 5, 6],
              [1, 2, 3, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 6, 6, 4, 5, 6]])
    num_hessians = sods.shape[1]

    #SENSE undersampling parameters, only for factors lower than 3
    FOV = np.ones((2,2), dtype=np.single)
    for i in range(-3, -1, 1):
        if shape_s[i] != shape_y[i]:
            over = shape_s[i] - shape_y[i]
            FOV[i, 0] = np.floor(over/2)
            FOV[i, 1] = np.ceil(over/2)
            
    dHe = np.zeros((num_hessians, shape_t[1])) #(21 x shots=2)
    dH  = np.zeros((shape_t[0], shape_t[1]))  #(6 x shots=2)
    E_prev = np.zeros((shape_t[1:]))
    E = np.zeros((shape_t[1:]))
    
    
    num_loads = 1
    for _ in range(num_iter):
        x_prev = x
        
        #Update the Weights
        w[flag_w == 2] = w[flag_w == 2] / div_weight_regularizer
        w[flag_w == 1] = w[flag_w == 1] * mul_weight_regularizer
        
        for s in range(num_loads):
            et, etg, eth = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, T, compute_grads=True)
            xT, xB = stform.sinc_rigid_transform(x, et, aux_imgs=True)
            xT = forward_application(xT, S, shape_y, FOV)
            xT = xT - y
            xT = xT * A
            xT_conj = np.conj(xT)
            E_prev[:, 0, 0, 0, 0] = np.sum(np.real(xT * xT_conj), axis=tuple(range(1,5)))
            
            G, GB, GC = stform.sinc_rigid_transform_gradient(xB, et, etg)
            G_conj = []
            for theta_index in range(shape_t[0]):
                G[theta_index] = forward_application(G[theta_index], S, shape_y, FOV)
                G[theta_index] = G[theta_index] * A
                G_conj.append(np.conj(G[theta_index]))
                
            for h_index in range(num_hessians):
                GG = stform.sinc_rigid_transform_hessian(xB, GB, GC, et, etg, eth, h_index)
                GG = forward_application(GG, S, shape_y, FOV)
                GG = GG * A
                GG = np.real(GG * xT_conj)
                GG = GG + np.real(G[sods[0, h_index]-1] * G_conj[sods[1, h_index]-1])
                
                dHe[h_index, :] = np.sum(GG, axis=tuple(range(1,5)))
                
            
            for theta_index in range(shape_t[0]):
                G[theta_index] = np.real( G[theta_index] * xT_conj)
                dH[theta_index, :] = np.sum(G[theta_index], axis=tuple(range(1,5)))
            
        MHe = 1000*np.eye(shape_t[0])
        for s in range(shape_t[1]):
            for k in range(num_hessians):
                if sods[0, k] == sods[1, k]:
                    MHe[sods[0, k]-1, sods[1,k]-1] = dHe[k, s] + w[s,0,0,0,0]
                else:
                    MHe[sods[0, k]-1, sods[1,k]-1] = dHe[k, s]
                    MHe[sods[1, k]-1, sods[0,k]-1] = dHe[k, s]
            dH[:, s] = np.linalg.lstsq(MHe, dH[:,s], rcond=None)[0]
        
        t_up = T - np.reshape(dH, (6,2,1,1,1,1))
        t_ang = t_up[3:,...]
        
        while np.any(t_ang > np.pi):
            t_ang[t_ang > np.pi] -= 2*np.pi
        while np.any(t_ang < -np.pi):
            t_ang[t_ang < -np.pi] += 2*np.pi
        t_up[3:, ...] = t_ang
        for m in range(3):
            t_tra = t_up[m,...]
            if shape_x[m] > 1:
                while np.any(t_tra > np.max(rgrid[m])):
                    t_tra[t_tra > np.max(rgrid[m])] -= shape_x[m]
                while np.any(t_tra < np.min(rgrid[m])):
                    t_tra[t_tra < np.min(rgrid[m])] += shape_x[m]
            t_up[m,...] = t_tra
            
         
        for s in range(num_loads):
            et = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, t_up)
            xT = stform.sinc_rigid_transform(x, et)
            xT = forward_application(xT, S, shape_y, FOV)
            xT = xT - y
            xT = xT * A
            xT_conj = np.conj(xT)
            E[:,0,0,0,0] = np.sum(np.real(xT * xT_conj), axis=tuple(range(1,5)))
        
        flag_w[E < E_prev] = 2
        flag_w[E >= E_prev] = 1
        for s in range(6):
            TauxA = T[s,...]
            TauxB = t_up[s,...]
            TauxA[flag_w == 2] = TauxB[flag_w == 2]
            T[s,...] = TauxA
            
        if mean_t:
            t_med = np.mean(T, axis=1, keepdims=True) #along shots
            et_dir = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, t_med)
            x = stform.sinc_rigid_transform(x_prev, et_dir)
            x = M * x
            T = T - t_med
        
    return T, x, w, flag_w
        
                
def forward_application(x, S, shape_y, FOV):
    x = x * S
    for i in range(-3, -1, 1):
        x = sense.sense(x, i, S.shape[i], shape_y[i], FOV[i+3, :])
    x = fft.fftn(x, axes=(-3,-2))
    return x

def solve_newandimproved_t(x, y, M, T, S, A, kgrid, kkgrid, rgrid, rkgrid, num_iter, w, flag_w, mean_t=1):
    div_weight_regularizer = 1.2
    mul_weight_regularizer = 2.0
    winic = w
    flag_winic = flag_w
    t_up = T
    
    shape_x = np.atleast_3d(x).shape
    shape_y = y.shape
    shape_s = S.shape
    shape_t = T.shape
    
    #       Second order derivatives
    sods = np.array([[1, 2, 3, 1, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 5, 4, 5, 6],
              [1, 2, 3, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 5, 6, 6, 4, 5, 6]])
    num_hessians = sods.shape[1]

    #SENSE undersampling parameters, only for factors lower than 3
    FOV = np.ones((2,2), dtype=np.single)
    for i in range(-3, -1, 1):
        if shape_s[i] != shape_y[i]:
            over = shape_s[i] - shape_y[i]
            FOV[i, 0] = np.floor(over/2)
            FOV[i, 1] = np.ceil(over/2)
            
    dHe = np.zeros((num_hessians, shape_t[1])) #(21 x shots=2)
    dH  = np.zeros((shape_t[0], shape_t[1]))  #(6 x shots=2)
    E_prev = np.zeros((shape_t[1:]))
    E = np.zeros((shape_t[1:]))
    
    
    num_loads = 1
    for _ in range(num_iter):
        x_prev = x
        
        #Update the Weights
        w[flag_w == 2] = w[flag_w == 2] / div_weight_regularizer
        w[flag_w == 1] = w[flag_w == 1] * mul_weight_regularizer
        
        for s in range(num_loads):
            et, etg, eth = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, T, compute_grads=True)
            xT, xB = stform.sinc_rigid_transform(x, et, aux_imgs=True)
            xT = forward_application(x, S, shape_y, FOV)
            xT = xT - y
            xT = xT * A
            xT_conj = np.conj(xT)
            E_prev[:, 0, 0, 0, 0] = np.sum(np.real(xT * xT_conj), axis=tuple(range(1,5)))
            
            G, GB, GC = stform.sinc_rigid_transform_gradient(xB, et, etg)
            G_conj = []
            for theta_index in range(shape_t[0]):
                G[theta_index] = forward_application(G[theta_index], S, shape_y, FOV)
                G[theta_index] = G[theta_index] * A
                G_conj.append(np.conj(G[theta_index]))
                
            for h_index in range(num_hessians):
                GG = stform.sinc_rigid_transform_hessian(xB, GB, GC, et, etg, eth, h_index)
                GG = forward_application(GG, S, shape_y, FOV)
                GG = GG * A
                GG = np.real(GG * xT_conj)
                GG = GG + np.real(G[sods[0, h_index]-1] * G_conj[sods[1, h_index]-1])
                
                dHe[h_index, :] = np.sum(GG, axis=tuple(range(1,5)))
                
            
            for theta_index in range(shape_t[0]):
                G[theta_index] = np.real( G[theta_index] * xT_conj)
                dH[theta_index, :] = np.sum(G[theta_index], axis=tuple(range(1,5)))
            
        MHe = 1000*np.eye(shape_t[0])
        for s in range(shape_t[1]):
            for k in range(num_hessians):
                if sods[0, k] == sods[1, k]:
                    MHe[sods[0, k]-1, sods[1,k]-1] = dHe[k, s] + w[s,0,0,0,0]
                else:
                    MHe[sods[0, k]-1, sods[1,k]-1] = dHe[k, s]
                    MHe[sods[1, k]-1, sods[0,k]-1] = dHe[k, s]
            dH[:, s] = np.linalg.lstsq(MHe, dH[:,s])[0]
        
        t_up = T - np.reshape(dH, (6,2,1,1,1,1))
        t_ang = t_up[3:,...]
        
        while np.any(t_ang > np.pi):
            t_ang[t_ang > np.pi] -= 2*np.pi
        while np.any(t_ang < np.pi):
            t_ang[t_ang < np.pi] += 2*np.pi
        t_up[3:, ...] = t_ang
        for m in range(3):
            t_tra = t_up[m,...]
            if shape_x[m-3] > 1:
                while np.any(t_tra > np.squeeze(rgrid[m])[-1]):
                    t_tra[t_tra > np.squeeze(rgrid[m])[-1]] -= shape_x[m-3]
                while np.any(t_tra < np.squeeze(rgrid[m])[0]):
                    t_tra[t_tra < np.squeeze(rgrid[m])[0]] += shape_x[m-3]
            t_up[m,...] = t_tra
            
         
        for s in range(num_loads):
            et = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, t_up)
            xT, _ = stform.sinc_rigid_transform(x, et)
            xT = xT - y
            xT = xT * A
            xT_conj = np.conj(xT)
            E_prev[:,0,0,0,0] = np.sum(np.real(xT * xT_conj), axis=tuple(range(1,5)))
        
        flag_w[E < E_prev] = 2
        flag_w[E >= E_prev] = 2
        for s in range(6):
            TauxA = T[s,...]
            TauxB = t_up[s,...]
            TauxA[flag_w == 2] = TauxB[flag_w == 2]
            T[s,...] = TauxA
            
        if mean_t:
            t_med = np.mean(T, axis=1, keepdims=True) #along shots
            et_dir = stform.precomp_sinc_transforms(kgrid, kkgrid, rkgrid, t_med)
            x = stform.sinc_rigid_transform(x_prev, et_dir)
            x = M * x
            T = T - t_med
        
    return T, x, w, flag_w   
    
    
    
    
    
    