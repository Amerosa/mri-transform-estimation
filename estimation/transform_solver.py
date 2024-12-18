import sigpy as sp
from .factors import calc_factors, calc_derivative_factors
from .transforms import RigidTransform, GradRigidTransform

class LevenbergMarquardt(sp.alg.Alg):
    def __init__(self, mps, masks, transforms, image, kspace, kgrid, kkgrid, rgrid, rkgrid, winic, constraint, max_iter, shot_batch_size=1, comm=None):
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
        self.comm = comm
        super().__init__(max_iter)
    
    def _update(self):
        xp = self.device.xp
        with self.device:
            for s in range(self.num_shots):
                #keeps the dimension proper so we get [1,6] instead of [6] when indexing transforms
                shot_idx = slice(s, s+1) 

                factors_trans, factors_tan, factors_sin = calc_factors(self.transforms[shot_idx], self.kgrid, self.rkgrid)
                grad_factors, _ = calc_derivative_factors(self.transforms[shot_idx], self.kgrid, self.kkgrid, self.rkgrid, factors_trans, factors_tan, factors_sin)
                T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
                S = sp.linop.Multiply(T.oshape, self.mps[:, xp.newaxis])
                F = sp.linop.FFT(S.oshape, axes=(-3,-2,-1))
                A = sp.linop.Multiply(F.oshape, self.masks[shot_idx])
                
                E = A * F * S * T
                diff = (E * self.img) - (self.masks[shot_idx] * self.kspace[:, xp.newaxis]) 
                energy_prev = xp.sum(xp.real(diff * diff.conj()), axis=(0,2,3,4))

                gradient = []
                partials = []
                for p_idx in range(6):
                    GT = GradRigidTransform(p_idx, self.img_shape, factors_trans, factors_tan, factors_sin, grad_factors)
                    partials.append(A * F * S * GT * self.img) #[ns, nc, kx, ky, kz]
                    gradient.append(xp.real(partials[-1] * diff.conj()).sum())
                gradient = xp.array(gradient)

                #Now we want to estiamte the hessian with our partial transfomrs which is just J.T*J 
                hessian = xp.zeros((6, 6))
                for i in range(6):
                    for j in range(i, 6):
                        if i == j:
                            hessian[i, j] = self.winic[shot_idx,i] +  xp.real(partials[i].conj() * partials[j]).sum()
                        else:
                            val = xp.real(partials[i].conj() * partials[j]).sum()
                            hessian[i, j] = val
                            hessian[j, i] = val
                

                delta = -1 * xp.linalg.lstsq(hessian, gradient, rcond=None)[0]
                z_next = self.transforms[shot_idx] + delta

                z_next[3:][z_next[3:] < -xp.pi] += 2 * xp.pi
                z_next[3:][z_next[3:] >  xp.pi] -= 2 * xp.pi

                factors_trans, factors_tan, factors_sin = calc_factors(z_next, self.kgrid, self.rkgrid)
                T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
                E = A * F * S * T
                diff = (E * self.img) - (self.masks[shot_idx] * self.kspace[:, xp.newaxis])
                energy_next = xp.real(diff * diff.conj()).sum()

                if energy_next < energy_prev:
                    self.winic[shot_idx] = xp.maximum(self.winic[shot_idx]/3, 1e-4)
                    self.transforms[shot_idx] = z_next.copy()

                else:
                    self.winic[shot_idx] = xp.minimum(self.winic[shot_idx]*1.5, 1e16)

        #Restrict new transform by zero mean shifting (if distributed need to gather across all nodes for mean)
        if self.comm is not None:
            gathered_transforms = self.comm.gatherv(self.transforms)
            if self.comm.rank == 0:
                mean_transform = xp.mean(gathered_transforms.reshape(-1, 6), axis=0, keepdims=True)
            else:
                mean_transform = xp.empty((1,6))
            self.comm.bcast(mean_transform)
        else:
            mean_transform = xp.mean(self.transforms, axis=0, keepdims=True)
        
        factors_trans, factors_tan, factors_sin = calc_factors(mean_transform, self.kgrid, self.rkgrid)
        T = RigidTransform(self.img_shape, factors_trans, factors_tan, factors_sin)
        sp.copyto(self.img, xp.squeeze(T * self.img)) #output will have one shot as a lingering dim hence the squeeze
        self.img *= self.constraint
        self.transforms -= mean_transform