import sigpy as sp
from .factors import calc_factors, calc_derivative_factors
from .transforms import RigidTransform, GradRigidTransform

class LevenbergMarquardt(sp.alg.Alg):
    def __init__(self, mps, masks, transforms, image, kspace, kgrid, kkgrid, rgrid, rkgrid, winic, constraint, max_iter, shot_batch_size=1):
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
        self.constraint = constraint
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
                jacobians = []
                for p_idx in range(6):
                    GT = GradRigidTransform(p_idx, self.img_shape, factors_trans, factors_tan, factors_sin, grad_factors)
                    jacobians.append(A * F * S * GT * self.img)
                    gradient[:, p_idx] = xp.sum(xp.real(jacobians[-1] * diff.conj()) , axis=(0,2,3,4))

                #Now we want to estiamte the hessian with our partial transfomrs which is just J.T*J 
                hessian = xp.zeros((self.shot_batch_size, 6, 6))
                for shot in range(len(hessian)):
                    for i in range(6):
                        for j in range(6):
                            if i == j:
                                hessian[shot, i, j] = (1+self.winic[shot])*xp.sum(xp.real(jacobians[i].conj() * jacobians[j]), axis=(0,2,3,4))
                            else:
                                hessian[shot, i, j] = xp.sum(xp.real(jacobians[i].conj() * jacobians[j]), axis=(0,2,3,4))

                z_next = xp.zeros((self.shot_batch_size,6))
                for shot in range(len(z_next)):
                    delta = xp.linalg.lstsq(hessian[shot], gradient[shot], rcond=None)[0]
                    delta *= -1 *(1/self.winic[shot])
                    z_next[shot] = self.transforms[(batch_idx*self.shot_batch_size)+shot] + delta

                z_next[:, 3:][z_next[:, 3:] < -xp.pi] += 2 * xp.pi
                z_next[:, 3:][z_next[:, 3:] >  xp.pi] -= 2 * xp.pi

                factors_trans, factors_tan, factors_sin = calc_factors(z_next, self.kgrid, self.rkgrid)
                T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
                E = A * F * S * T
                diff = (E * self.img) - (self.masks[shot_idx] * self.kspace[:, xp.newaxis])
                energy_next = xp.sum(xp.real(diff * diff.conj()), axis=(0,2,3,4))

                if energy_next < energy_prev:
                    self.winic[shot_idx] = xp.maximum(self.winic[shot_idx]/1.2, 1e-4)
                else:
                    self.winic[shot_idx] = xp.minimum(self.winic[shot_idx]*5, 1e16)

                if (energy_next < energy_prev).any():
                    self.decreasing_err = True
                    z_next = xp.where(energy_next < energy_prev, z_next, self.transforms[shot_idx])
                    sp.copyto(self.transforms[shot_idx], z_next)
                else:
                    self.decreasing_err = False           

        #Restrict new transform by zero mean shifting
        mean_transform = xp.mean(self.transforms, axis=0, keepdims=True)
        factors_trans, factors_tan, factors_sin = calc_factors(mean_transform, self.kgrid, self.rkgrid)
        T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
        sp.copyto(self.img, xp.squeeze(T * self.img)) #output will have one shot as a lingering dim hence the squeeze
        self.img *= self.constraint
        self.transforms -= mean_transform