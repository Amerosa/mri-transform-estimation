import sigpy as sp
import numpy as np
from .algos import JointEstimation


class MotionCorruptedImageRecon(sp.app.App):
    def __init__(
        self, 
        kspace, 
        mps, 
        smask, 
        img=None, 
        transforms=None,
        max_cg_iter=3,
        max_nm_iter=1,
        max_joint_iter=100,
        tol=1e-6,
        device=sp.cpu_device
    ):
        self.kspace = kspace
        self.mps = mps
        self.smask = smask
        self.img = img
        self.transforms = transforms
        self.max_cg_iter = max_cg_iter
        self.max_nm_iter = max_nm_iter
        self.max_joint_iter = max_joint_iter
        self.tol = tol
        self.device = device

        xp = self.device.xp
        if self.img is None:
            with self.device:
                self.img = xp.zeros(self.kspace.shape[1:], dtype=self.kspace.dtype)

        if self.transforms is None:
            with self.device:
                self.transforms = xp.zeros( (len(self.smask), 6), dtype=xp.float16)

        alg = JointEstimation(
            self.kspace, 
            self.mps, 
            self.smask, 
            self.img,
            self.transforms,
            self.max_cg_iter,
            self.max_nm_iter,
            self.max_joint_iter,
            self.tol
        )
        super().__init__(alg)
    
    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(
                    max_voxel_change="{0:.2E}".format(
                        sp.backend.to_device(self.alg.xerr.item(), sp.backend.cpu_device)
                    ),
                    rot="{0:.2E}".format(
                        sp.backend.to_device(self.alg.transforms[0,4].item() * np.pi/180, sp.backend.cpu_device)
                    ),
                )
    
    def _output(self):
        return self.alg.x
    
    