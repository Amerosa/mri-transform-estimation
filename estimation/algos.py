from sigpy.alg import Alg
from sigpy import backend
from sigpy.mri.linop import Sense
from sigpy.mri.app import _estimate_weights
from sigpy.app import LinearLeastSquares, App
from estimation.linop import RigidTransform
from sigpy.linop import Multiply
import time 
import sigpy.plot as pl
import numpy as np
class TransformEstimation(Alg):
    def __init__(self, A, transform_states, image, kspace, max_iter):
        self.A = A
        self.rigid_transform = transform_states
        self.image = image
        self.kspace = kspace
        self.winic = 1
        self.device = backend.get_device(image)
        super().__init__(max_iter)

    
    def _update(self):
        xp = self.device.xp
        with self.device:
            
            #Avoid numerical instability
            if self.winic < 1e-4:
                self.winic  *= 2
            if self.winic > 1e16:
                self.winic /= 2 

            #First forward pass to evaluate at f(z)
            T = self.rigid_transform.Transform()
            w = self.A * T * self.image - self.kspace
            fz = xp.sum(xp.real(xp.conj(w) * w))

            #Calculating the first order partial derivated 6 in total
            gradient = xp.ones(6)
            first_ord_partials = (self.A * self.rigid_transform.DTransform(l)* self.image for l in range(6))
            for idx, deriv in enumerate(first_ord_partials):
                gradient[idx] = xp.sum(xp.real(xp.conj(w) * deriv))
            
            #Calculating the second order partials for the Hessian matrix
            #Only the 22 partials of the upper right triangle are calculated
            #then copied to the bottom under the diagonal
            hessian = xp.zeros((6,6))
            for l, m in xp.nditer(xp.triu_indices(6)):
                gl = self.A * self.rigid_transform.DTransform(l) * self.image
                gm = self.A * self.rigid_transform.DTransform(m) * self.image
                he = self.A * self.rigid_transform.DDTransform(l,m) * self.image
                hessian[l,m] = xp.sum(xp.real(xp.conj(gl) * gm + xp.conj(w) * he))
            #Copy the upper triangle to the bottom under the diagonal
            lower_idxs = xp.tril_indices(6, k=-1)
            hessian[lower_idxs]  = hessian.T[lower_idxs]
            
            #Winic adjustment for (27) of paper
            hessian += self.winic * xp.eye(6)

            #Newton's step, lstsq is solving for H^-1 * G
            z_next = self.rigid_transform.parameters - xp.linalg.lstsq(hessian, gradient, rcond=None)[0]

            #Restrict ranges so as to not overshoot newton's method
            #Angles
            z_next[3:][z_next[3:] < -xp.pi] += 2 * xp.pi
            z_next[3:][z_next[3:] >  xp.pi] -= 2 * xp.pi

            #Translations
            min = xp.min(self.rigid_transform.rgrid, axis=(-3,-2,-1))
            max = xp.max(self.rigid_transform.rgrid, axis=(-3,-2,-1))
            max_idx = (z_next[:3] > max).nonzero()
            min_idx = (z_next[:3] < min).nonzero()
            z_next[max_idx] -= xp.array(self.image.shape)[max_idx]
            z_next[min_idx] += xp.array(self.image.shape)[min_idx]

            self.rigid_transform.update_state(z_next)
            T = self.rigid_transform.Transform()
            w = self.A * T * self.image - self.kspace
            fz_next = xp.sum(xp.real(xp.conj(w) * w))

            if fz_next > fz:
                self.winic *= 2
            else:
                self.winic /= 1.2

class TransformEstimationRecon(App):
    def __init__(self, A, states, image, kspace, max_iter):
        self.states = states
        self.image = image
        alg = TransformEstimation(A, self.states, image, kspace, max_iter)
        np.set_printoptions(precision=2)
        super().__init__(alg, show_pbar=False)

    def _output(self):
        print('-'*40)
        print(f'Estimated transform parameters are: {self.states.parameters}')
        print('-'*40)
        print('\n')
        return self.states.parameters

    def _post_update(self):
        print(f'Iteration: {self.alg.iter} -> {self.states.parameters}')

class JointEstimation(Alg):
    def __init__(self, kspace, maps, bins, bsize, cg_iter, nm_iter, joint_iter, tol=1e-6):

        self.img_shape = kspace.shape[1:]

        weights = _estimate_weights(kspace, None, None)
        self.A = Sense(maps, weights=weights, coil_batch_size=None)
        self.kspace = kspace
        self.maps = maps
        self.cg_iter = cg_iter
        self.nm_iter = nm_iter
        self.tol = tol
        self.bins = bins
        self.bsize = bsize

        self.device = backend.get_device(kspace)
        with self.device:
            xp = self.device.xp
            self.tform_parameters = xp.zeros(6, dtype=xp.float64)
            self.recon_image = xp.zeros(self.img_shape, dtype=xp.complex64)
            self.xerr = xp.infty
            p = 1 / (xp.sum(xp.abs(maps)**2, axis=0) + 0.001)
            self.P = Multiply(self.img_shape, p)
        super().__init__(joint_iter)

    def _update(self):
        print(f'Iteration: {self.iter} with tform {self.tform_parameters}')

        tform_states = RigidTransform(self.tform_parameters, self.img_shape, self.device)
        xant = self.recon_image
        self.recon_image = LinearLeastSquares(self.A * tform_states.Transform(), self.kspace, x=self.recon_image, P=self.P, max_iter=self.cg_iter, show_pbar=False).run()

        tform_params_temp = []
        for b in range(self.bins):
            extraction = np.zeros(self.kspace.shape, dtype=np.complex64)
            extraction[:, b*self.bsize:(b+1)*self.bsize] = self.kspace[:, b*self.bsize:(b+1)*self.bsize]
            #pl.ImagePlot(np.squeeze(self.A.H * extraction), title=f'Motion state {b}')
            tform_states = RigidTransform(self.tform_parameters, self.img_shape, self.device)
            start = time.perf_counter()
            t_params = TransformEstimationRecon(self.A, tform_states, self.recon_image, extraction, max_iter=self.nm_iter).run()
            finish = time.perf_counter()
            tform_params_temp.append(t_params)
            print(f'Motion state {b} took {(finish - start):.2f} seconds to complete')

        t_med = np.array(tform_params_temp).mean(0) #Mean params across all the motions states
        self.recon_image = RigidTransform(t_med, self.img_shape, self.device).Transform() * self.recon_image
        self.tform_parameters -= t_med

        xp = self.device.xp
        xant = self.recon_image - xant
        xant = xp.real(xant*xant.conj())
        xant = xp.max(xant)
        self.xerr = xant 
        #self.xerr = xp.linalg.norm(xant - self.recon_image)
        print(f'Error is {self.xerr}')
    
    def _done(self):
        return (self.iter >= self.max_iter) or (self.xerr <= self.tol)



