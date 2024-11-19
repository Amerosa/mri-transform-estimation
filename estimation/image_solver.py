import sigpy as sp
from .factors import calc_factors
from .transforms import RigidTransform

class ImageEstimation(sp.app.App):
    def __init__(        
        self,
        kspace,
        mps,
        mask,
        transforms,
        kgrid,
        rkgrid,
        img=None,
        constraint=None,
        P=None,
        device=sp.cpu_device,
        shot_batch_size=1,
        max_iter=30,
        tol=1e-6,
        show_pbar=True,
        leave_pbar=True,
        ):
        
        self.kspace = kspace
        self.mps = mps
        self.mask = mask
        self.transforms = transforms
        self.kgrid = kgrid
        self.rkgrid = rkgrid
        self.shot_batch_size = shot_batch_size
        self.num_shots = mask.shape[0]
        self.device = device
        self.constraint = constraint
        
        if img is None:
            self.img = device.xp.zeros(kspace.shape[1:], dtype=kspace.dtype)
        self.img = sp.to_device(img, device)
        self.kspace = sp.to_device(self.kspace, device)
        if constraint is not None:
            self.constraint = sp.to_device(self.constraint, device)
        b = self._prep_kspace()
        b = sp.to_device(b, device)
        alg = sp.alg.ConjugateGradient(self._cg_encoding, b, x=self.img, P=P, max_iter=max_iter, tol=tol)
        super().__init__(alg, show_pbar=show_pbar, leave_pbar=leave_pbar)
    
    def _post_update(self):
        if self.constraint is not None:
            self.alg.x *= self.constraint

    def _output(self):
        return self.alg.x

    def _summarize(self):
        if self.show_pbar:
            self.pbar.set_postfix(
                    obj="{0:.2E}".format(self.alg.resid)
                )

    def _cg_encoding(self, x):
        xp = self.device.xp
        recon = xp.zeros(x.shape, dtype=xp.complex64)
        num_chunks = -(self.num_shots//-self.shot_batch_size)
        S = sp.linop.Multiply((self.shot_batch_size, *x.shape), self.mps[:, xp.newaxis])
        F = sp.linop.FFT(S.oshape, axes=(-3,-2,-1))
        for chunk in range(num_chunks):
            shot_idx = slice(chunk * self.shot_batch_size, (chunk+1) * self.shot_batch_size)
            factors_trans, factors_tan, factors_sin = calc_factors(self.transforms[shot_idx], self.kgrid, self.rkgrid)
            T = RigidTransform(x.shape, factors_trans, factors_tan, factors_sin)
            A = sp.linop.Multiply(F.oshape, self.mask[shot_idx])
            E = T.H * S.H * F.H * A * F * S * T
            recon += E * x
        return recon * self.constraint
    
    def _prep_kspace(self):
        xp = self.device.xp
        prepped_input = xp.zeros(self.kspace.shape[1:], dtype=xp.complex64)
        num_chunks = -(self.num_shots//-self.shot_batch_size)
        S = sp.linop.Multiply((self.shot_batch_size, *prepped_input.shape), self.mps[:, xp.newaxis])
        F = sp.linop.FFT(S.oshape, axes=(-3,-2,-1))
        for chunk in range(num_chunks):
            shot_idx = slice(chunk * self.shot_batch_size, (chunk+1) * self.shot_batch_size)
            factors_trans, factors_tan, factors_sin = calc_factors(self.transforms[shot_idx], self.kgrid, self.rkgrid)
            T = RigidTransform(prepped_input.shape, factors_trans, factors_tan, factors_sin)
            A = sp.linop.Multiply(self.kspace[:, xp.newaxis].shape, self.mask[shot_idx])
            E = T.H * S.H * F.H * A
            prepped_input += E * self.kspace[:, xp.newaxis]
        return prepped_input * self.constraint