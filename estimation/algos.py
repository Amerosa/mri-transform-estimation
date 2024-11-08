import sys, os
sys.path.append('../estimation')
import sigpy as sp
import numpy as np
from .factors import *
from .encoding import *
from .utils import *

class TransformEstimation(sp.alg.Alg):
    def __init__(self, mps, masks, transforms, image, kspace, kgrid, kkgrid, rgrid, rkgrid, winic, max_iter):
        self.mps = mps
        self.masks = masks
        self.transforms = transforms
        self.img = image
        self.img_shape = image.shape
        self.kspace = kspace
        self.kgrid = kgrid
        self.kkgrid = kkgrid
        self.rgrid = rgrid
        self.rkgrid = rkgrid
        self.winic = winic
        self.device = sp.get_device(image)
        self.decreasing_err = False

        xp = self.device.xp
        with self.device:
            self.gradient = xp.empty((len(transforms), 6))
            self.hessian = xp.empty((len(transforms),6,6))
            self.image_grads = xp.empty((6, *self.img_shape))
        super().__init__(max_iter)

    
    def _update(self):
        xp = self.device.xp
        with self.device:

            z_prev = self.transforms
            #Avoid numerical instability
            self.winic = xp.where(self.winic < 1e-4, self.winic * 2, self.winic)
            self.winic = xp.where(self.winic > 1e16, self.winic / 2, self.winic)

            #First forward pass to evaluate at f(z)
            factors_trans, factors_tan, factors_sin = calc_factors(self.transforms, self.kgrid, self.rkgrid)
            grad_factors, hess_factors = calc_derivative_factors(self.transforms, self.kgrid, self.kkgrid, self.rkgrid, factors_trans, factors_tan, factors_sin)
            
            E = AlignedSense(self.img, self.mps, self.masks, factors_trans, factors_tan, factors_sin) #[channels, shots, kx, ky, kz]
            w  = (E * self.img) - (self.masks * self.kspace[:, xp.newaxis])
            w_conj = xp.conj(w)
            error_prev = xp.sum(xp.real(w * w_conj), axis=(0,2,3,4)) #each shot will have its own error    

            #Calculating the first order partial derivated 6 in total

            
            for l in range(6):
                gE = GradAlignedSense(self.img, l, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, shot_batch_size=None)
                wl = gE * self.img #[coils, shots, kx, ky, kz]
                self.gradient[:, l] = xp.sum(xp.real(w_conj * wl) , axis=(0,2,3,4)) #Sum along every dimensions except the shots
            
            #Calculating the second order partials for the Hessian matrix
            #Only the 22 partials of the upper right triangle are calculated
            #then copied to the bottom under the diagonal
            
            rows, cols = xp.triu_indices(6)
            rows = rows.tolist()
            cols = cols.tolist()
            for l, m in zip(rows, cols):
                gl = GradAlignedSense(self.img, l, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, shot_batch_size=None) * self.img
                gm = GradAlignedSense(self.img, m, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, shot_batch_size=None) * self.img
                he = HessAlignedSense(self.img, l, m, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, hess_factors, shot_batch_size=None) * self.img
                self.hessian[:,l,m] = xp.sum(xp.real(xp.conj(gl) * gm + w_conj * he), axis=(0,2,3,4))
                
            #Copy the upper triangle to the bottom under the diagonal
            lower_idxs = xp.tril_indices(6, k=-1)
            for shot in range(len(self.transforms)):
                self.hessian[shot][lower_idxs]  = self.hessian[shot].T[lower_idxs]
            #print(self.hessian)
            
            #for shot in range(len(self.transforms)):
            #    self.hessian[shot] = self.gradient[None, shot].T @ self.gradient[None, shot]

            #Winic adjustment for (27) of paper
            self.hessian += self.winic[:,xp.newaxis,xp.newaxis] * xp.eye(6)

            #Newton's step, lstsq is solving for H^-1 * G
            #Can only solve one by one for each shot
            z_next = xp.zeros((len(self.transforms),6))
            for idx in range(len(self.transforms)):
                z_next[idx] = z_prev[idx] - xp.linalg.lstsq(self.hessian[idx], self.gradient[idx], rcond=None)[0]

            #Restrict ranges so as to not overshoot newton's method
            #Angles
            z_next[:, 3:][z_next[:, 3:] < -xp.pi] += 2 * xp.pi
            z_next[:, 3:][z_next[:, 3:] >  xp.pi] -= 2 * xp.pi

            #Translations
            #min = xp.min(self.rigid_transform.rgrid, axis=(-3,-2,-1))
            #max = xp.max(self.rigid_transform.rgrid, axis=(-3,-2,-1))
            #max_idx = (z_next[:3] > max).nonzero()
            #min_idx = (z_next[:3] < min).nonzero()
            #z_next[max_idx] -= xp.array(self.image.shape)[max_idx]
            #z_next[min_idx] += xp.array(self.image.shape)[min_idx]

            factors_trans, factors_tan, factors_sin = calc_factors(z_next, self.kgrid, self.rkgrid)
            E = AlignedSense(self.img, self.mps, self.masks, factors_trans, factors_tan, factors_sin) #[channels, shots, kx, ky, kz]
            w = (E * self.img) - (self.masks * self.kspace[:, xp.newaxis])
            error_next = xp.sum(xp.real(xp.conj(w) * w), axis=(0,2,3,4))

            self.winic = xp.where(error_next > error_prev, self.winic*2, self.winic/1.2)

            if (error_next < error_prev).any():
                self.decreasing_err = True
                idxs = (error_next < error_prev).nonzero()
                self.transforms[idxs] = z_next[idxs]
            else:
                self.decreasing_err = False

            z_med = xp.mean(self.transforms, axis=0, keepdims=True)
            factors_trans, factors_tan, factors_sin = calc_factors(z_med, self.kgrid, self.rkgrid)
            T = RigidTransform(self.img.shape, factors_trans, factors_tan, factors_sin)
            self.img = xp.sum(T * self.img, axis=0)
            self.transforms -= z_med

    def compute_image_grads(self, factors_trans, factors_tan, factors_sin, grad_factors, cache):
        """ Cache holds the three results after third, second and third rotations respectively"""
        self.image_grads[0] = GradTranslation(0, self.img_shape, grad_factors) * cache[2]
        self.image_grads[1] = GradTranslation(1, self.img_shape, grad_factors) * cache[2]
        self.image_grads[2] = GradTranslation(2, self.img_shape, grad_factors) * cache[2]
        R0   = Rotation(0, self.img_shape, factors_tan, factors_sin)
        R1   = Rotation(1, self.img_shape, factors_tan, factors_sin)
        D_R0 = GradRotation(0, self.img_shape, factors_tan, factors_sin, grad_factors)
        D_R1 = GradRotation(1, self.img_shape, factors_tan, factors_sin, grad_factors)
        D_R2 = GradRotation(2, self.img_shape, factors_tan, factors_sin, grad_factors)
        U    = Translation(self.img_shape, factors_trans)
        self.image_grads[3] =  U * D_R0 * cache[1]
        self.image_grads[4] =  U * R0 * D_R1 * cache[0]
        self.image_grads[5] =  U * R0 * R1 * D_R2 * cache[0]

class LevenbergMarquardt(sp.alg.Alg):
    def __init__(self, mps, masks, transforms, image, kspace, kgrid, kkgrid, rgrid, rkgrid, winic, max_iter, shot_batch_size=1):
        self.mps = mps
        self.masks = masks
        self.transforms = transforms
        self.img = image
        self.img_shape = image.shape
        self.kspace = kspace
        self.kgrid = kgrid
        self.kkgrid = kkgrid
        self.rgrid = rgrid
        self.rkgrid = rkgrid
        self.winic = winic
        self.device = sp.get_device(image)
        self.decreasing_err = False
        self.shot_batch_size = shot_batch_size
        self.num_shots = len(transforms)

        super().__init__(max_iter)
    
    def _update(self):
        xp = self.device.xp
        with self.device:
            batch_iter = -(self.num_shots//-self.shot_batch_size)

            for batch_idx in range(batch_iter):
                shot_idx = slice(batch_idx * self.shot_batch_size, (batch_idx+1) * self.shot_batch_size)

                factors_trans, factors_tan, factors_sin = calc_factors(self.transforms[shot_idx], self.kgrid, self.rkgrid)
                grad_factors, _ = calc_derivative_factors(self.transforms[shot_idx], self.kgrid, self.kkgrid, self.rkgrid, factors_trans, factors_tan, factors_sin)
                T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
                S = sp.linop.Multiply(T.oshape, self.mps[:, xp.newaxis])
                F = sp.linop.FFT(S.oshape, axes=(-3,-2,-1))
                A = sp.linop.Multiply(F.oshape, self.masks[shot_idx])
                
                E = A * F * S * T
                diff = (E * self.img) - (self.masks[shot_idx] * self.kspace[:, xp.newaxis]) 
                energy_prev = xp.sum(xp.real(diff * diff.conj()), axis=(0,2,3,4))

                gradient = xp.zeros((self.shot_batch_size, 6))
                #6 T(z) wrt axis each one producing a [ns x vol] shape
                partial_transforms = xp.zeros((self.shot_batch_size, 6, *self.img_shape), dtype=xp.complex64) 
                for p_idx in range(6):
                    GT = GradRigidTransform(p_idx, self.img_shape, factors_trans, factors_tan, factors_sin, grad_factors)
                    partial_transforms[:, p_idx] = GT * self.img
                    partial_encoding = A * F * S * partial_transforms[:, p_idx] #[coils, shots, kx, ky, kz]
                    gradient[:, p_idx] = xp.sum(xp.real(partial_encoding * diff.conj()) , axis=(0,2,3,4))

                #Now we want to estiamte the hessian with our partial transfomrs which is just J.T*J 
                hessian = xp.zeros((self.shot_batch_size, 6, 6))
                for shot in range(len(hessian)):
                    for i in range(6):
                        for j in range(6):
                            hessian[shot, i, j] = xp.sum(xp.real(partial_transforms[shot, i].conj() * partial_transforms[shot, j]))
                hessian += self.winic[shot_idx,xp.newaxis,xp.newaxis] * xp.eye(6)

                z_next = xp.zeros((self.shot_batch_size,6))
                for shot in range(len(z_next)):
                    z_next[shot] = self.transforms[(batch_idx*self.shot_batch_size)+shot] - xp.linalg.lstsq(hessian[shot], gradient[shot], rcond=None)[0]

                factors_trans, factors_tan, factors_sin = calc_factors(z_next, self.kgrid, self.rkgrid)
                T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
                E = A * F * S * T
                diff = (E * self.img) - (self.masks[shot_idx] * self.kspace[:, xp.newaxis])
                energy_next = xp.sum(xp.real(diff * diff.conj()), axis=(0,2,3,4))

                updated_winic = xp.where(energy_next > energy_prev, self.winic*2, self.winic/1.2)
                sp.copyto(self.winic, updated_winic)

                if (energy_next < energy_prev).any():
                    self.decreasing_err = True
                    z_next = xp.where(energy_next < energy_prev, z_next, self.transforms[shot_idx])
                    #self.transforms[shot_idx] = z_next
                    sp.copyto(self.transforms[shot_idx], z_next)
                else:
                    self.decreasing_err = False           
