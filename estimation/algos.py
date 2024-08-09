from sigpy.alg import Alg
from sigpy import backend
from sigpy.mri.linop import Sense
from sigpy.mri.app import _estimate_weights
from sigpy.app import LinearLeastSquares
from estimation.linop import RigidTransform
import time 

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
            print(f'New parameters -> {z_next[:3], z_next[3:] * 180 / xp.pi }')
            if fz_next > fz:
                self.winic *= 2
            else:
                self.winic /= 1.2
            
class JointEstimation(Alg):
    def __init__(self, kspace, maps, cg_iter, nm_iter, joint_iter, tol=1e-6):

        self.img_shape = kspace.shape[1:]

        weights = _estimate_weights(kspace, None, None)
        self.A = Sense(maps, weights=weights)
        self.kspace = kspace
        self.maps = maps
        self.cg_iter = cg_iter
        self.nm_iter = nm_iter
        self.tol = tol

        self.device = backend.get_device(kspace)
        with self.device:
            xp = self.device.xp
            initial_prams = xp.zeros(6, dtype=xp.float64)
            self.tform_states = RigidTransform(initial_prams, self.img_shape, self.device)
            self.recon_image = xp.zeros(self.img_shape, dtype=xp.complex64)
            self.xerr = xp.inf
        super().__init__(joint_iter)

    def _update(self):
        xant = self.recon_image
        self.recon_image = LinearLeastSquares(self.A * self.tform_states.Transform(), self.kspace, max_iter=self.cg_iter).run()

        start = time.perf_counter()
        solver_t = TransformEstimation(self.A, self.tform_states, self.recon_image, self.kspace, max_iter=self.nm_iter)
        while not solver_t.done():
            solver_t.update()
        finish = time.perf_counter()
        print(f'Solve T time elapsed is {finish - start}')

        xp = self.device.xp
        xant = self.recon_image - xant
        xant = xp.real(xant*xant.conj())
        xant = xp.max(xant)
        self.xerr = xant
    
    def _done(self):
        return (self.iter >= self.max_iter) or self.xerr <= self.tol



