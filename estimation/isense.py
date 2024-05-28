#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:31:48 2024

@author: goresky
"""
import numpy as np



def sense(x, m, ns, ny, FOV):
    perm = list(range(x.ndim))
    perm[0] = m
    perm[m] = 1
    
    if ns != ny:
        if m != 0:
            x = np.transpose(x, axes=perm)
        
        xb = x[FOV[1]:-FOV[0],...]
        xb += x[-FOV[0]:,...]
        xb += x[:FOV[1],...]
        x = xb
        if m != 0:
            x = np.transpose(x, axes=perm)
    return x

def isense(x, m, ns, ny, FOV):
    perm = list(range(x.ndim))
    perm[0] = m
    perm[m] = 1
    
    if ns != ny:
        if m != 0:
            x = np.transpose(x, axes=perm)
        
        if ns <= 3*ny:
            x = np.concatenate( x[FOV[0]:,...], x, x[:-FOV[1],...], axis=0)
        else:
            raise ValueError("The reconstruction code doesn't work for SENSE factor larger than 3")
            
        if m != 1:
            x = np.transpose(x, axes=perm) 
    return x