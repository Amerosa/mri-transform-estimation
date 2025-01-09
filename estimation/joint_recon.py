import sigpy as sp
from .factors import generate_transform_grids
from .transform_solver import LevenbergMarquardt
from .image_solver import ImageEstimation
from .metrics import objective_all_shot
import numpy as np
from .encoding import Encoding


class MotionCorruptedImageRecon(sp.app.App):
    def __init__(
        self, 
        kspace, 
        mps, 
        mask,
        kgrid,
        kkgrid,
        rgrid,
        rkgrid,
        img=None, 
        transforms=None,
        constraint=None,
        P = None,
        max_cg_iter=3,
        max_nm_iter=1,
        max_joint_iter=100,
        tol=1e-6,
        shot_chunk_size=None,
        save_objective_values = True,
        device=sp.cpu_device,
        verbose=False,
        comm=None,
        show_pbar=True
    ):
        self.kspace = kspace
        self.mps = mps
        self.mask = mask
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
        self.comm = comm
        self.shot_chunk_size = shot_chunk_size

        xp = self.device.xp
        with self.device:
            if self.img is None:
                self.img = xp.zeros(self.kspace.shape[1:], dtype=self.kspace.dtype)
            else:
                self.img = sp.to_device(self.img, device=device)

            if self.transforms is None:
                self.transforms = xp.zeros( (len(self.mask), 6), dtype=float)
            else:
                self.transforms = sp.to_device(self.transforms, device=device)

            self.damp = xp.ones((len(self.mask), 6))

            self.kgrid  = kgrid #sp.to_device(kgrid, device=device)
            self.kkgrid = kkgrid #sp.to_device(kkgrid, device=device)
            self.rgrid  = rgrid #sp.to_device(rgrid, device=device)
            self.rkgrid = rkgrid #sp.to_device(rkgrid, device=device)

            self.kspace = sp.to_device(self.kspace, device=device)
            self.mps = sp.to_device(self.mps, device=device)
            self.mask = sp.to_device(self.mask, device=device)
            self.constraint = sp.to_device(self.constraint, device=device)
            
            if self.save_objective_values:
                self.objective_values = [self.objective()]
            
            self.t_iterations = [self.transforms.get()]

            if comm is not None:
                show_pbar = show_pbar and comm.rank == 0

        alg = sp.alg.AltMin(self._minX, self._minT, self.max_joint_iter)
        super().__init__(alg, show_pbar=show_pbar)
    
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
        return self.img, self.transforms, np.array(self.objective_values), np.stack(self.t_iterations, axis=-1)
    
    def _post_update(self):        
        if self.verbose:
            xp = self.device.xp
            estimate = self.transforms.copy()
            estimate[:, 3:] *= (180 / xp.pi)
            print('\n')
            print(f'Iteration {self.alg.iter} |')
            for shot in range(len(estimate)):
                print(f'MotionState {shot+1}: {xp.array_str(estimate[shot], precision=2)}')
            print(f'Objective: {objective_all_shot(self.img, self.kspace, self.mps, self.mask, self.transforms, self.kgrid, self.rkgrid)}')
            print('-' * 80)
    
    def objective(self):
        obj = objective_all_shot(self.img, self.kspace, self.mps, self.mask, self.transforms, self.kgrid, self.rkgrid)
        if self.comm is not None:
            self.comm.allreduce(obj)
        return obj.item()
    
    def _minX(self):
        ImageEstimation(
            self.kspace,
            self.mps,
            self.mask,
            self.transforms,
            self.kgrid,
            self.rkgrid,
            self.img,
            self.P,
            self.constraint,
            device=self.device,
            max_iter=self.max_cg_iter,
            comm=self.comm,
            shot_chunk_size=self.shot_chunk_size,
            show_pbar=False
            ).run()
    
    def _minT(self):
        sp.app.App(
            LevenbergMarquardt(
                self.mps,
                self.mask,
                self.transforms,
                self.img,
                self.kspace,
                self.kgrid,
                self.kkgrid,
                self.rgrid,
                self.rkgrid,
                self.damp,
                self.constraint,
                self.max_nm_iter,
                comm=self.comm
            ), show_pbar=False).run()