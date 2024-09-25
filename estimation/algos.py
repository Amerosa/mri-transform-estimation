import sigpy as sp
import numpy as np
from .factors import *
from .encoding import *

class TransformEstimation(sp.alg.Alg):
    def __init__(self, mps, masks, transforms, image, kspace, kgrid, kkgrid, rgrid, rkgrid, winic, max_iter):
        self.mps = mps
        self.masks = masks
        self.transforms = transforms
        self.img = image
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
            w  = (E * self.img) - self.kspace[:, xp.newaxis]
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
            
            for l, m in xp.nditer(xp.triu_indices(6)):
                gl = GradAlignedSense(self.img, l, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, shot_batch_size=None) * self.img
                gm = GradAlignedSense(self.img, m, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, shot_batch_size=None) * self.img
                he = HessAlignedSense(self.img, l, m, self.mps, self.masks, factors_trans, factors_tan, factors_sin, grad_factors, hess_factors, shot_batch_size=None) * self.img
                self.hessian[:,l,m] = xp.sum(xp.real(xp.conj(gl) * gm + w_conj * he), axis=(0,2,3,4))
                
            #Copy the upper triangle to the bottom under the diagonal
            lower_idxs = xp.tril_indices(6, k=-1)
            for shot in range(len(self.transforms)):
                self.hessian[shot][lower_idxs]  = self.hessian[shot].T[lower_idxs]
            
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
            w = (E * self.img) - self.kspace[:, xp.newaxis]
            error_next = xp.sum(xp.real(xp.conj(w) * w), axis=(0,2,3,4))

            self.winic = xp.where(error_next >= error_prev, self.winic*2, self.winic/1.2)

            self.decreasing_err = (error_next < error_prev).all()
            if (error_next < error_prev).any():
                idxs = (error_next < error_prev).nonzero()
                self.transforms[idxs] = z_next[idx]


            z_med = xp.mean(self.transforms, axis=0, keepdims=True)
            factors_trans, factors_tan, factors_sin = calc_factors(z_med, self.kgrid, self.rkgrid)
            T = RigidTransform(self.img.shape, factors_trans, factors_tan, factors_sin)
            self.img = xp.sum(T * self.img, axis=0)
            self.transforms = z_next - z_med

class JointEstimation(sp.alg.Alg):
    def __init__(self, mps, masks, kspace, kgrid, kkgrid, rgrid, rkgrid, tol=1e-6, max_iter=100):
        self.x = np.zeros(mps.shape[1:], dtype=np.complex64)
        self.num_shots = len(masks)
        self.transforms = np.zeros((self.num_shots, 6), dtype=np.float64)
        self.mps = mps
        self.masks = masks
        self.kspace = kspace
        
        self.winic = np.ones(self.num_shots)
        self.kgrid = kgrid
        self.kkgrid = kkgrid
        self.rgrid = rgrid
        self.rkgrid = rkgrid

        p = 1 / (np.sum(np.abs(mps)**2, axis=0) + 0.001)
        self.P = sp.linop.Multiply(mps.shape[1:], p)
        self.xerr = np.infty
        self.tol = tol
        self.decreasing_err = False
        super().__init__(max_iter)

    def _update(self):

        prev_x = self.x.copy()
        factors_trans, factors_tan, factors_sin = calc_factors(self.transforms, self.kgrid, self.rkgrid)
        E = AlignedSense(self.x, self.mps, self.masks, factors_trans, factors_tan, factors_sin, shot_batch_size=None)
        #need to modify kspace to have an extra dim for the shots
        sp.app.LinearLeastSquares(E, np.repeat(self.kspace[:, np.newaxis], self.num_shots, 1), self.x, P=self.P, max_iter=3, show_pbar=False).run()
        
        t_est_alg = TransformEstimation(self.mps, self.masks, self.transforms, self.x, self.kspace, self.kgrid, self.kkgrid, self.rgrid, self.rkgrid, self.winic, max_iter=1)
        while not t_est_alg.done():
            t_est_alg.update()
        self.transforms = t_est_alg.transforms
        self.x = t_est_alg.img
        self.winic = t_est_alg.winic
        self.decreasing_err = t_est_alg.decreasing_err
        self.xerr = self.x - prev_x
        self.xerr = np.real(self.xerr * np.conj(self.xerr))
        self.xerr = np.max(self.xerr)

    def _done(self):
        return (self.iter >= self.max_iter) or (self.xerr <= self.tol and self.decreasing_err == True)