import sigpy as sp
from .encoding import Encoding

class ImageEstimation(sp.app.App):
    def __init__(        
        self,
        y,
        mps,
        mask,
        transforms,
        kgrid,
        rkgrid,
        x=None,
        P=None,
        constraint=None,
        shot_chunk_size=None,
        max_iter=10,
        tol=0,
        device=sp.cpu_device,
        comm=None,
        show_pbar=True,
        leave_pbar=True,
        ):
        
        y = sp.to_device(y, device)
        
        if constraint is not None:
            self.constraint = sp.to_device(constraint, device)

        E = Encoding(transforms, mask, mps, kgrid, rkgrid, comm=comm, shot_chunk_size=shot_chunk_size)

        if x is None:
            with device:
                x = device.xp.zeros(E.ishape, dtype=y.dtype)
        else:
            x = sp.to_device(x, device)

        alg = sp.alg.ConjugateGradient(E.N, E.H*y, x, P=P, max_iter=max_iter, tol=tol)

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