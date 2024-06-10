#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:15:59 2024

@author: Giuseppe Grossi
"""
import numpy as np
import scipy.fft as fft

def precomp_sinc_transforms(kgrid, kkgrid, rkgrid, T, direct=True, compute_grads=False):
    #Shape of T should just be (6 2 1 1 1 1) flat array technically
    #The T will be a transform for a specific shot
    #index 0-2 are the translations adn 3-5 are rotations, 6 params in total
    img_axis = (-3, -2, -1)
    theta = T[3:,...]
    tan_theta2 = np.tan(theta/2)
    if direct:
        t = -1j * T[:3,...]
        tan_theta2j = 1j * tan_theta2
        sin_theta = -1j * np.sin(theta)
    else:
        t = 1j * T[:3,...]
        tan_theta2j = -1j * tan_theta2
        sin_theta = 1j * np.sin(theta)
        
    if compute_grads:
        tan_theta = np.tan(theta)
        tan_theta_cuad = (1+ tan_theta * tan_theta) / 2
        cos_theta = np.cos(theta)
        
        
    #should be 1x3 and 1x3      
   # print(f"tan_theta2j has shape {tan_theta2j.shape} and sin_theta has shape {sin_theta.shape}")
    kgrid = [np.atleast_3d(ele) for ele in kgrid]
    kkgrid = [np.atleast_3d(ele) for ele in kkgrid]
    #Tans from from 1-3 
    #(123 128) ()
    et  = {}
    etg = {}
    eth = {}
    et["tans"]  = [np.exp(tan_theta2j[i] * np.atleast_3d(rkgrid[0][i])) for i in range(3)]
    et["sins"]  = [np.exp(sin_theta[i] * np.atleast_3d(rkgrid[1][i])) for i in range(3)]
    et["trans"] = np.exp(t[0]*kgrid[0] + t[1]*kgrid[1] + t[2]*kgrid[2])

    perm = np.array([[0,2,1], [1,0,2]]) - 3 #we are shifting the index b/c our image is stored in the last 3 dims
    #we store the image in the last dim because its easier for boradcasting the arrays when mul


    if compute_grads:
        etg["tans"] = [tan_theta_cuad[i] * 1j * np.atleast_3d(rkgrid[0][i]) for i in range(3)]
        etg["sins"] = [cos_theta[i] * -1j * np.atleast_3d(rkgrid[1][i]) for i in range(3)]
        
        eth["tans"] = [tan_theta2[i] + etg["tans"][i] for i in range(3)]
        eth["sins"] = [-tan_theta[i] + etg["sins"][i] for i in range(3)]
        
        for k, sinusoid in enumerate(["tans" , "sins"]):
            for i in range(3):
                etg[sinusoid][i] *= et[sinusoid][i]
                eth[sinusoid][i] *= etg[sinusoid][i]
                
                etg[sinusoid][i] = fft.ifftshift(np.atleast_3d(etg[sinusoid][i]), axes=perm[k,i])
                eth[sinusoid][i] = fft.ifftshift(np.atleast_3d(eth[sinusoid][i]), axes=perm[k,i])
        

    #(1 1 1 1 2 3)
    #(128 1)    (1 128) -> (3 2 128 128 1)
    #1 3 2 tans // 2 1 3 sins
    
    #perm = [[1,3,2], [2,1,3]]
    for i, precomp in enumerate(et["tans"]):
        et["tans"][i] = fft.ifftshift(np.atleast_3d(precomp), axes=perm[0,i])
    
    for i, precomp in enumerate(et["sins"]):
        et["sins"][i] = fft.ifftshift(np.atleast_3d(precomp), axes=perm[1,i])
    #et["tans"] = [fft.ifftshift(x) for x in et["tans"]]
    #et["sins"] = [fft.ifftshift(x) for x in et["sins"]]
    
    
    if compute_grads:
        etg["trans"] = [-1j * kgrid[i] * et["trans"] for i in range(3)]
        etg["trans"] = [fft.ifftshift(ele, axes=img_axis) for ele in etg["trans"]]
        
        eth["trans"] = [-kkgrid[i] * et["trans"] for i in range(6)]
        eth["trans"] = [fft.ifftshift(ele, axes=img_axis) for ele in eth["trans"]]
        
    et["trans"] = fft.ifftshift(et["trans"], axes=img_axis)
    
    if compute_grads:
        return et, etg, eth
    else:
        return et

def rotation_as_shear(I, v_tan, v_sin, fourier_dims, reverse=False):
    assert len(fourier_dims) == 3
    #print(I.shape, v_tan.shape, v_sin.shape)
    v_tan = np.atleast_3d(v_tan)
    v_sin = np.atleast_3d(v_sin)
    #I.T shape -> (1 1 1 128 128)
    #V.T shape -> (2 1 1 128 128)
    #               (2 1 1 128 128)
    # transpose   (128 128 1 1 2)
        #First shear in dim 1-2-1
    fourier_dims = [ele - 3 for ele in fourier_dims]
    I = fft.fft(I, axis=fourier_dims[0])
    IB = I
    I = (I * v_tan) #(128 128 1) * (2 1 128 128 1)
    I = fft.ifft(I, axis=fourier_dims[0])

        #Second shear in dim 3-1-3
    I = fft.fft(I, axis=fourier_dims[1])
    I = (I * v_sin)
    I = fft.ifft(I, axis=fourier_dims[1])

        #Third shear in dim 2-3-2
    I = fft.fft(I, axis=fourier_dims[2])
    I = (I * v_tan)
    if reverse:
        I = np.sum(I, axis=0)
    I = fft.ifft(I, axis=fourier_dims[2])

    return I, IB

def direct_transform(I, et, aux_imgs):
    IB = []
    I, aux = rotation_as_shear(I, et["tans"][0], et["sins"][0], (0,1,0))
    IB.append(aux)
    #plt.imshow(np.abs(I[0,0,:,:,0]), cmap='gray')
    #plt.show()
    I, aux = rotation_as_shear(I, et["tans"][1], et["sins"][1], (2,0,2))
    IB.append(aux)
    #plt.imshow(np.abs(I[0,0,:,:,0]), cmap='gray')
    #plt.show()
    I, aux = rotation_as_shear(I, et["tans"][2], et["sins"][2], (1,2,1))
    IB.append(aux)
    #plt.imshow(np.abs(I[0,0,:,:,0]), cmap='gray')
    #plt.show()

    #Translation part
    I = fft.fftn(I, axes=(-3,-2,-1))
    IB.append(I)
    I = (I * np.atleast_3d(et["trans"]))
    I = fft.ifftn(I, axes=(-3,-2,-1))
    
    #print(I.shape)
    if aux_imgs:
        return I, IB
    else:
        return I

def inverse_transform(I, et):
    #print("Preforming inverse Transform")
    #Back-translation
    I = fft.fftn(I, axes=(-3,-2,-1))
    I = (I * et["trans"])
    I = fft.ifftn(I, axes=(-3,-2,-1))

    #Back-rotations
    I, _ = rotation_as_shear(I, et["tans"][2], et["sins"][2], (1,2,1))
    I, _ = rotation_as_shear(I, et["tans"][1], et["sins"][1], (2,0,2))
    I, _ = rotation_as_shear(I, et["tans"][0], et["sins"][0], (0,1,0), reverse=True)
    #There is a sum inehre on the 5th dim not sure why
    return I

def sinc_rigid_transform(I, et, direct=True, aux_imgs=False):
    I = np.atleast_3d(I)
    if direct:
        return direct_transform(I, et, aux_imgs)
    else:
        return inverse_transform(I, et)


def sinc_rigid_transform_gradient(aux_images, et, etg):
    
    GB = []
    GC = [] 
    G  = []
    img_axes = (-3, -2 , -1) #last three axes are the image and we fft over this alot
    
    # et["trans"] = np.asarray( et["trans"] )
    # et["tans"]  = [np.asarray(ele) for ele in et["tans"]]
    # et["sins"]  = [np.asarray(ele) for ele in et["sins"]]
    
    # etg["trans"]  = [np.asarray(ele) for ele in etg["trans"]]
    # etg["tans"]  = [np.asarray(ele) for ele in etg["tans"]]
    # etg["sins"]  = [np.asarray(ele) for ele in etg["sins"]]
    
    # aux_images = [np.asarray(ele) for ele in aux_images]
    # #translation params
    for i in range(3):
        x = aux_images[3] * etg["trans"][i]
        x = np.asarray(x)
        x = fft.ifftn(x, axes=img_axes)
        G.append(x)

    #First Rotation
    x = [None] * 2 
    x[0] = aux_images[0] * et["tans"][0]
    x[1] = aux_images[0] * etg["tans"][0]
    
    # x[0] = np.asarray(x[0])
    # x[1] = np.asarray(x[1])
    
    for i in range(2):
        x[i] = fft.ifft(x[i], axis=-3)
        x[i] = fft.fft(x[i], axis=-2)
    
    x[1] = (etg["sins"][0] * x[0]) + (et["sins"][0] * x[1])
    x[0] = x[0] * et["sins"][0]
    for i in range(2):
        x[i] = fft.ifft(x[i], axis=-2)
        x[i] = fft.fft(x[i], axis=-3)
        
    x[0] = (etg["tans"][0] * x[0]) + (et["tans"][0] * x[1])
    x[0] = fft.ifft(x[0], axis=-3)
    x[0] = fft.fft(x[0], axis=-1)
    GC.append(x[0])
    x[0] = x[0] * et["tans"][1]
    x[0] = fft.ifft(x[0], axis=-1)
    x[0] = fft.fft(x[0], axis=-3)
    x[0] = x[0] * et["sins"][1]
    x[0] = fft.ifft(x[0], axis=-3)
    x[0] = fft.fft(x[0], axis=-1)
    x[0] = x[0] * et["tans"][1]
    x[0] = fft.ifft(x[0], axis=-1)
    x[0] = fft.fft(x[0], axis=-2)
    GC.append(x[0])
    
    x[0] = x[0] * et["tans"][2]
    x[0] = fft.ifft(x[0], axis=-2)
    x[0] = fft.fft(x[0], axis=-1)
    x[0] = x[0] * et["sins"][2]
    x[0] = fft.ifft(x[0], axis=-1)
    x[0] = fft.fft(x[0], axis=-2)
    x[0] = x[0] * et["tans"][2]
    
    x[0] = fft.fftn(x[0], axes=(-3,-1))
    GB.append(x[0])
    x[0] = x[0] * et["trans"]
    x[0] = fft.ifftn(x[0], axes=img_axes)
    G.append(x[0])
    
    #Second rotation
    x[0] = aux_images[1] * et["tans"][1]
    x[1] = aux_images[1] * etg["tans"][1]
    for i in range(2):
        x[i] = fft.ifft(x[i], axis=-1)
        x[i] = fft.fft(x[i], axis=-3)
    
    x[1] = (etg["sins"][1] * x[0]) + (et["sins"][1] * x[1])
    x[0] = x[0] * et["sins"][1]
    
    for i in range(2):
        x[i] = fft.ifft(x[i], axis=-3)
        x[i] = fft.fft(x[i], axis=-1)
    
    x[0] = (etg["tans"][1] * x[0]) + (et["tans"][1] * x[1])
    
    x[0] = fft.ifft(x[0], axis=-1)
    x[0] = fft.fft(x[0], axis=-2)
    GC.append(x[0])
    x[0] = x[0] * et["tans"][2]
    x[0] = fft.ifft(x[0], axis=-2)
    x[0] = fft.fft(x[0], axis=-1)
    x[0] = x[0] * et["sins"][2]
    x[0] = fft.ifft(x[0], axis=-1)
    x[0] = fft.fft(x[0], axis=-2)
    x[0] = x[0] * et["tans"][2]
    x[0] = fft.fftn(x[0], axes=img_axes)
    GB.append(x[0])
    x[0] = x[0] * et["trans"]
    x[0] = fft.ifftn(x[0], axes=img_axes)
    G.append(x[0])
    
    #Thrid Rotation
    x[0] = aux_images[2] * et["tans"][2]
    x[1] = aux_images[2] * etg["tans"][2]
    for i in range(2):
        x[i] = fft.ifft(x[i], axis=-2)
        x[i] = fft.fft(x[i], axis=-1)
    x[1] = (etg["sins"][2] * x[0]) + (et["sins"][2] * x[1])
    x[0] = x[0] * et["sins"][2]
    for i in range(2):
        x[i] = fft.ifft(x[i], axis=-1)
        x[i] = fft.fft(x[i], axis=-2)
    x[0] = (etg["tans"][2] * x[0]) + (et["tans"][2] * x[1])
    x[0] = fft.fftn(x[0], axes=(-3,-1))
    GB.append(x[0])
    x[0] = x[0] * et["trans"]
    x[0] = fft.ifftn(x[0], axes=img_axes)
    G.append(x[0])
    
    # G = [np.asnumpy(ele) for ele in G]
    # GB = [np.asnumpy(ele) for ele in GB]
    # GC = [np.asnumpy(ele) for ele in GC]
    
    
    
    return G, GB, GC
    
    
def sinc_rigid_transform_hessian(aux_images, GB, GC, et, etg, eth, h_index):
    H = []
    x = [None] * 6 
    img_axes = (-3, -2 , -1)
    
    # et["trans"] = np.asarray( et["trans"] )
    # et["tans"]  = [np.asarray(ele) for ele in et["tans"]]
    # et["sins"]  = [np.asarray(ele) for ele in et["sins"]]
    
    # etg["trans"]  = [np.asarray(ele) for ele in etg["trans"]]
    # etg["tans"]  = [np.asarray(ele) for ele in etg["tans"]]
    # etg["sins"]  = [np.asarray(ele) for ele in etg["sins"]]
    
    # eth["trans"]  = [np.asarray(ele) for ele in eth["trans"]]
    # eth["tans"]  = [np.asarray(ele) for ele in eth["tans"]]
    # eth["sins"]  = [np.asarray(ele) for ele in eth["sins"]]
    
    # aux_images = [np.asarray(ele) for ele in aux_images]
    
    
    # GB = [np.asarray(ele) for ele in GB]
    # GC = [np.asarray(ele) for ele in GC]
    
    #translation parameters            
    if h_index in list(range(6)):
        x[0] = aux_images[3] * eth["trans"][h_index]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]
    
    
    #Translation-Roation Parameters
    #Fist Rotation
    for n in range(3):
        for m in range(3):
            if (5 + 3*n + m + 1) == h_index:
                x[0] = GB[n] * etg["trans"][m]
                x[0] = fft.ifftn(x[0], axes=img_axes)
                H = x[0]
            
    
    if h_index == 15:
        x[0] = GC[0] * et["tans"][1]
        x[1] = GC[0] * etg["tans"][1]
        for i in range(2):
            x[i] = fft.ifft(x[i], axis=-1)
            x[i] = fft.fft(x[i], axis=-3)
            
        x[1] = (etg["sins"][1] * x[0]) + (et["sins"][1] * x[1])
        x[0] = et["sins"][1] * x[0]
        for i in range(2):
            x[i] = fft.ifft(x[i], axis=-3)
            x[i] = fft.fft(x[i], axis=-1)
        x[0] = (etg["tans"][1] * x[0]) + (et["tans"][1] * x[1])
        x[0] = fft.ifft(x[0], axis=-1)
        x[0] = fft.fft(x[0], axis=-2)
        x[0] = x[0] * et["tans"][2]
        x[0] = fft.ifft(x[0], axis=-2)
        x[0] = fft.fft(x[0], axis=-1)
        x[0] = x[0] * et["sins"][2]
        x[0] = fft.ifft(x[0], axis=-1)
        x[0] = fft.fft(x[0], axis=-2)
        x[0] = x[0] * et["tans"][2]
        
        x[0] = fft.fftn(x[0], axes=(-3,-1))
        x[0] = x[0] * et["trans"]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]
    
    if h_index == 16:
        x[1] = GC[1] * etg["tans"][2]
        x[0] = GC[1] * et["tans"][2]
        for i in range(2):
            x[i] = fft.ifft(x[i], axis=-2)
            x[i] = fft.fft(x[i], axis=-1)
        x[1] = (etg["sins"][2] *  x[0]) + (et["sins"][2] * x[1])
        x[0] = et["sins"][2] * x[0]
        for i in range(2):
            x[i] = fft.ifft(x[i], axis=-1)
            x[i] = fft.fft(x[i], axis=-2)
        x[0] = (etg["tans"][2] * x[0]) + (et["tans"][2] * x[1])
        x[0] = fft.fftn(x[0], axes=(-3, -1))
        x[0] = x[0] * et["trans"]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]
        
     
    if h_index == 17:
        x[1] = GC[2] * etg["tans"][2]
        x[0] = GC[2] * et["tans"][2]
        for i in range(2):
            x[i] = fft.ifft(x[i], axis=-2)
            x[i] = fft.fft(x[i], axis=-1)
        x[1] = (etg["sins"][2] *  x[0]) + (et["sins"][2] * x[1])
        x[0] = et["sins"][2] * x[0]
        for i in range(2):
            x[i] = fft.ifft(x[i], axis=-1)
            x[i] = fft.fft(x[i], axis=-2)
        x[0] = (etg["tans"][2] * x[0]) + (et["tans"][2] * x[1])
        x[0] = fft.fftn(x[0], axes=(-3, -1))
        x[0] = x[0] * et["trans"]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]
        
    if h_index == 18:
        x[0] = aux_images[0] * et["tans"][0]
        x[1] = aux_images[0] * eth["tans"][0]
        x[2] = aux_images[0] * etg["tans"][0]
        x[3] = aux_images[0] * etg["tans"][0]
        for i in range(4):
            x[i] = fft.ifft(x[i], axis=-3)
            x[i] = fft.fft(x[i], axis=-2)
        x[1] = (eth["sins"][0] * x[0]) + (et["sins"][0] * x[1])
        x[4] = (etg["sins"][0] * x[0]) + (et["sins"][0] * x[2])
        x[5] = (etg["sins"][0] * x[0]) + (et["sins"][0] * x[3])
        x[0] = (et["sins"][0] * x[0]) 
        x[2] = (etg["sins"][0] * x[2]) 
        x[3] = (etg["sins"][0] * x[3]) 
        for i in range(6):
            x[i] = fft.ifft(x[i], axis=-2)
            x[i] = fft.fft(x[i], axis=-3)
        
        x[0] = (eth["tans"][0] * x[0]) + (et["tans"][0] * x[1])
        x[1] = (etg["tans"][0] * x[4]) + (et["tans"][0] * x[2])
        x[2] = (etg["tans"][0] * x[5]) + (et["tans"][0] * x[3])
        x[0] = x[0] + x[1] + x[2]
        x[0] = fft.ifft(x[0], axis=-3)
        
        x[0] = fft.fft(x[0], axis=-1)
        x[0] = x[0] * et["tans"][1]
        x[0] = fft.ifft(x[0], axis=-1)
        x[0] = fft.fft(x[0], axis=-3)
        x[0] = x[0] * et["sins"][1]
        x[0] = fft.ifft(x[0], axis=-3)
        x[0] = fft.fft(x[0], axis=-1)
        x[0] = x[0] * et["tans"][1]
        x[0] = fft.ifft(x[0], axis=-1)
        
        x[0] = fft.fft(x[0], axis=-2)
        x[0] = x[0] * et["tans"][2]
        x[0] = fft.ifft(x[0], axis=-2)
        x[0] = fft.fft(x[0], axis=-1)
        x[0] = x[0] * et["sins"][2]
        x[0] = fft.ifft(x[0], axis=-1)
        x[0] = fft.fft(x[0], axis=-2)
        x[0] = x[0] * et["tans"][2]
        
        x[0] = fft.fftn(x[0], axes=(-3,-1))
        x[0] = x[0] * et["trans"]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]
        
    if h_index == 19:
        x[0] = aux_images[1] * et["tans"][1]
        x[1] = aux_images[1] * eth["tans"][1]
        x[2] = aux_images[1] * etg["tans"][1]
        x[3] = aux_images[1] * etg["tans"][1]
        for i in range(4):
            x[i] = fft.ifft(x[i], axis=-1)
            x[i] = fft.fft(x[i], axis=-3)
        x[1] = (eth["sins"][1] * x[0]) + (et["sins"][1] * x[1])
        x[4] = (etg["sins"][1] * x[0]) + (et["sins"][1] * x[2])
        x[5] = (etg["sins"][1] * x[0]) + (et["sins"][1] * x[3])
        x[0] = (et["sins"][1] * x[0]) 
        x[2] = (etg["sins"][1] * x[2]) 
        x[3] = (etg["sins"][1] * x[3]) 
        for i in range(6):
            x[i] = fft.ifft(x[i], axis=-3)
            x[i] = fft.fft(x[i], axis=-1)
        
        x[0] = (eth["tans"][1] * x[0]) + (et["tans"][1] * x[1])
        x[1] = (etg["tans"][1] * x[4]) + (et["tans"][1] * x[2])
        x[2] = (etg["tans"][1] * x[5]) + (et["tans"][1] * x[3])
        x[0] = x[0] + x[1] + x[2]
        
        x[0] = fft.ifft(x[0], axis=-1)
        
        x[0] = fft.fft(x[0], axis=-2)
        x[0] = x[0] * et["tans"][2]
        x[0] = fft.ifft(x[0], axis=-2)
        x[0] = fft.fft(x[0], axis=-1)
        x[0] = x[0] * et["sins"][2]
        x[0] = fft.ifft(x[0], axis=-1)
        x[0] = fft.fft(x[0], axis=-2)
        x[0] = x[0] * et["tans"][2]
        
        x[0] = fft.fftn(x[0], axes=(-3,-1))
        x[0] = x[0] * et["trans"]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]
        
    if h_index == 20:
        x[0] = aux_images[2] * et["tans"][2]
        x[1] = aux_images[2] * eth["tans"][2]
        x[2] = aux_images[2] * etg["tans"][2]
        x[3] = aux_images[2] * etg["tans"][2]
        for i in range(4):
            x[i] = fft.ifft(x[i], axis=-2)
            x[i] = fft.fft(x[i], axis=-1)
        x[1] = (eth["sins"][2] * x[0]) + (et["sins"][2] * x[1])
        x[4] = (etg["sins"][2] * x[0]) + (et["sins"][2] * x[2])
        x[5] = (etg["sins"][2] * x[0]) + (et["sins"][2] * x[3])
        x[0] = (et["sins"][2] * x[0]) 
        x[2] = (etg["sins"][2] * x[2]) 
        x[3] = (etg["sins"][2] * x[3]) 
        for i in range(6):
            x[i] = fft.ifft(x[i], axis=-1)
            x[i] = fft.fft(x[i], axis=-2)
        
        x[0] = (eth["tans"][2] * x[0]) + (et["tans"][2] * x[1])
        x[1] = (etg["tans"][2] * x[4]) + (et["tans"][2] * x[2])
        x[2] = (etg["tans"][2] * x[5]) + (et["tans"][2] * x[3])
        x[0] = x[0] + x[1] + x[2]
        
        x[0] = fft.fftn(x[0], axes=(-3,-1))
        x[0] = x[0] * et["trans"]
        x[0] = fft.ifftn(x[0], axes=img_axes)
        H = x[0]

    #H = np.asnumpy(H)
    return H

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


        

