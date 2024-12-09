import sigpy as sp
from .factors import generate_transform_grids
from .transform_solver import LevenbergMarquardt
from .image_solver import ImageEstimation
from .metrics import objective_all_shot
import time
import numpy as np


class MotionCorruptedImageRecon(sp.app.App):
    def __init__(
        self, 
        kspace, 
        mps, 
        smask,
        fullres_shape,
        img=None, 
        transforms=None,
        constraint=None,
        P = None,
        max_cg_iter=3,
        max_nm_iter=1,
        max_joint_iter=100,
        tol=1e-6,
        save_objective_values = False,
        device=sp.cpu_device,
        verbose=False
    ):
        self.kspace = kspace
        self.mps = mps
        self.smask = smask
        self.fullres_shape = fullres_shape
        self.img = img
        self.transforms = transforms
        self.constraint = constraint
        self.P = P
        self.max_cg_iter = max_cg_iter
        self.max_nm_iter = max_nm_iter
        self.max_joint_iter = max_joint_iter
        self.tol = tol
        self.save_objective_values = save_objective_values
        self.device = device
        self.verbose = verbose

        xp = self.device.xp
        if self.img is None:
            with self.device:
                self.img = xp.zeros(self.kspace.shape[1:], dtype=self.kspace.dtype)
        else:
            self.img = sp.to_device(self.img, device=device)

        if self.transforms is None:
            with self.device:
                self.transforms = xp.zeros( (len(self.smask), 6), dtype=xp.float16)
        else:
            self.transforms = sp.to_device(self.transforms, device=device)

        self.damp = xp.ones((len(self.smask), 6))
        self.kgrid, self.kkgrid, self.rgrid, self.rkgrid = generate_transform_grids(self.fullres_shape, self.img.shape, self.device)
        self.kspace = sp.to_device(self.kspace, device=device)
        self.mps = sp.to_device(self.mps, device=device)
        self.smask = sp.to_device(self.smask, device=device)
        self.constraint = sp.to_device(self.constraint, device=device)
        
        if self.save_objective_values:
            self.objective_values = [self.objective()]
        
        self.t_iterations = [self.transforms.get()]
        alg = JointMin(self._minX, self._minT, self.img, self.transforms, max_iter=max_joint_iter, tol=tol)
        super().__init__(alg)
    
    def _summarize(self):
        self.t_iterations.append(self.transforms.get())
        if self.save_objective_values:
            self.objective_values.append(self.objective())

        if self.show_pbar:
            if self.save_objective_values:
                self.pbar.set_postfix(
                    obj="{0:.2E}".format(self.objective_values[-1] * 1e6)
                )
            else:
                self.pbar.set_postfix(
                    max_voxel_change="{0:.2E}".format(
                        sp.backend.to_device(self.alg.xerr, sp.backend.cpu_device)
                    )
                )

    def _output(self):
        return self.alg.x, self.alg.t, np.array(self.objective_values), np.stack(self.t_iterations, axis=-1)
    
    def _post_update(self):
        #Tempory implementation for outputting estimatations each iteration
        if self.verbose:
            xp = self.device.xp
            estimate = self.alg.t.copy()
            estimate[: 3:] *= (180 / xp.pi)
            print('\n')
            print(f'Iteration {self.alg.iter} |')
            for shot in range(len(estimate)):
                print(f'MotionState {shot+1}: {xp.array_str(estimate[shot], precision=2)}')
            print(f'Objective: {objective_all_shot(self.img, self.kspace, self.mps, self.smask, self.transforms, self.kgrid, self.rkgrid)}')
            print('-' * 80)
    
    def objective(self):
        return objective_all_shot(self.img, self.kspace, self.mps, self.smask, self.transforms, self.kgrid, self.rkgrid)
    
    def _minX(self):
        ImageEstimation(
            self.kspace,
            self.mps,
            self.smask,
            self.transforms,
            self.kgrid,
            self.rkgrid,
            self.img,
            self.constraint,
            self.P,
            device=self.device,
            max_iter=self.max_cg_iter,
            show_pbar=False
            ).run()
    
    def _minT(self):
        sp.app.App(
            LevenbergMarquardt(
                self.mps,
                self.smask,
                self.transforms,
                self.img,
                self.kspace,
                self.kgrid,
                self.kkgrid,
                self.rgrid,
                self.rkgrid,
                self.damp,
                self.constraint,
                self.max_nm_iter
            ), show_pbar=False).run()

class JointMin(sp.alg.Alg):
    def __init__(self, minX, minT, x, t, max_iter=1000, tol=1e-6):
        self.minX = minX
        self.minT = minT
        self.x = x
        self.t = t
        self.device = sp.get_device(x)
        self.xerr = self.device.xp.inf
        self.tol = tol
        super().__init__(max_iter)

    #TODO Need to come up with a correct convergence right now just runs through max iter
    def _update(self):
        #old_x = self.x.copy()
        self.minX()
        self.minT()
        #with self.device:
            #xp  = self.device.xp
            #diff = self.x - old_x
            #self.xerr = (xp.max(xp.real(diff * xp.conj(diff)))).item() / xp.max(xp.real(self.x * xp.conj(self.x))) ).item()

    def _done(self):
        return (self.iter >= self.max_iter) #or (self.xerr < self.tol)