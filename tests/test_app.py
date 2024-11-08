import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.plot as pl
from sigpy.mri import app, sim
from scipy.ndimage import gaussian_filter

from estimation.app import ImageEstimation, MotionCorruptedImageRecon
from estimation.factors import make_grids, calc_factors
from estimation.transforms import RigidTransform
from estimation.algos import TransformEstimation, LevenbergMarquardt

if __name__ == "__main__":
    unittest.main()

class TestApp(unittest.TestCase):
    def shepp_logan_setup(self):
        img_shape = [64, 64, 64]
        mps_shape = [2, 64, 64, 64]

        img = sp.shepp_logan(img_shape)
        mps = sim.birdcage_maps(mps_shape)

        num_shots = 2
        mask = np.zeros((num_shots, *img_shape), dtype=bool)
        mask[0, :, :, ::2] = 1
        mask[1, :, :, 1::2] = 1

        ksp = np.sum(mask[:, np.newaxis] * sp.fft(mps * img, axes=[-3, -2, -1]), axis=0)
        return img, mps, ksp, mask
    
    def generate_transforms(self, num_shots, rotations, random=False):
        transforms = np.zeros((num_shots, 6), dtype=float)
        for axis, degree in enumerate(rotations):
            if (degree == 0):
                continue
            if random:
                rads = (np.random.rand(num_shots) - 0.5) * degree * np.pi / 180
            else:
                rads = np.linspace(-0.5, 0.5, num_shots, endpoint=True) * degree * np.pi / 180
            transforms[:, axis+3] = rads
        return transforms - np.mean(transforms, axis=0)

    def generate_sampling_mask(self, num_shots, img_shape):
        mask = np.zeros((num_shots, *img_shape), dtype=bool)
        partition_size = img_shape[-1] // num_shots
        for shot in range(num_shots):
            #mask[shot, :,:, shot*partition_size:(shot+1)*partition_size] = 1
            mask[shot, :,:, shot::num_shots] =1 
        return mask
    
    def corrupt_shepp_logan_setup(self, num_shots, rotations, random=False):
        img_shape = [32, 32, 32]
        #
        mps_shape = [8, 32, 32, 32]

        #img_shape = [64, 64, 64]
        #mps_shape = [8, 64, 64, 64]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)
        mask = self.generate_sampling_mask(num_shots, img_shape)

        transforms = self.generate_transforms(num_shots, rotations, random)

        kgrid, _, _, rkgrid = make_grids(img.shape)
        factors_trans, factors_tan, factors_sin = calc_factors(transforms, kgrid, rkgrid)
        T = RigidTransform(img.shape, factors_trans, factors_tan, factors_sin)
        #pl.ImagePlot(T * img, z=0)
        #pl.ImagePlot(mask, z=0)
        ksp = np.sum(mask * sp.fft(mps[:, np.newaxis] * (T * img), axes=[-3,-2,-1]), axis=1)

        return img, mps, ksp, mask, transforms
    
    def test_shepp_logan_ImageEstimation(self):
        img, mps, ksp, mask = self.shepp_logan_setup()

        kgrid, kkgrid, rgrid, rkgrid = make_grids(img.shape)
        transforms = np.array([[0,0,0,0,0,0], [0,0,0,0,0,0]])
        sense_recon = app.SenseRecon(ksp,mps).run()
        img_estimate = ImageEstimation(ksp, mps, mask, transforms, kgrid, rkgrid).run()
        npt.assert_allclose(img, img_estimate, atol=1e-2, rtol=1e-2)
        npt.assert_allclose(sense_recon, img_estimate, atol=1e-2, rtol=1e-2)
        import sigpy.plot as pl
        pl.ImagePlot(np.concatenate((img, img_estimate, sense_recon), axis=-1))

    def test_corrupt_shepp_logan_ImageEstimation(self):
        import sigpy.plot as pl
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(8, [10,5,4], random=True)
        kgrid, _, _, rkgrid = make_grids(img.shape)
        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)
        #M = np.ones(img.shape)
        #norm = np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        #ksp /= norm
        pl.ImagePlot(np.sum(sp.ifft(ksp, axes=[-3,-2,-1]), axis=0))
        precond = 1 / (np.sum(np.abs(mps)**2, axis=0) + 1e-3)
        P = sp.linop.Multiply(img.shape, precond)
        img_estimate = np.zeros(img.shape, dtype=np.complex64)
        img_estimate = ImageEstimation(ksp, mps, mask, transforms, kgrid, rkgrid, img=img_estimate, constraint=M, P=P, tol=1e-6, max_iter=100, device=sp.Device(0)).run()
        pl.ImagePlot(np.concatenate((img, img_estimate.get()), axis=-1))
        npt.assert_allclose(img, img_estimate.get(), atol=1e-2, rtol=0)

    def test_shepp_logan_LevenbergMarquart(self):
        import sigpy.plot as pl
        num_shots = 2
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [10,5,4])
        #ksp /= np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        kgrid, kkgrid, rgrid, rkgrid = make_grids(img.shape)
        estimates = np.zeros((num_shots,6), dtype=float)
        w = np.ones(num_shots)
        #img = sp.to_device(img, sp.Device(0))
        #w  = sp.to_device(w, sp.Device(0))
        #ksp = sp.to_device(ksp, sp.Device(0))
        #mps = sp.to_device(mps, sp.Device(0))
        #mask = sp.to_device(mask, sp.Device(0))
        #init = sp.to_device(transforms, sp.Device(0))
        
        alg = LevenbergMarquardt(mps, mask, estimates, img, ksp, kgrid, kkgrid, rgrid,rkgrid, w, 100, shot_batch_size=1)
        #sp.app.App(alg).run()
        while not alg.done():
            alg.update()
            print(alg.transforms[:, 3:] * 180 / np.pi)
        npt.assert_allclose(transforms, estimates, atol=1e-1, rtol=0)

    def test_shepp_logan_recon(self):
        num_shots = 2
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [10,4,5])
        import sigpy.plot as pl
        import sigpy.mri
        #corr_img = sigpy.mri.app.SenseRecon(ksp, mps, device=sp.Device(0), tol=1e-6).run()
        #corr_img = np.sum(mps * sp.ifft(ksp, axes=[-3,-2,-1]), axis=0)
        #pl.ImagePlot(img)
        #pl.ImagePlot(np.sum(mps * sp.ifft(ksp, axes=[-3,-2,-1]), axis=0))
        norm = np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        #ksp /= norm
        kgrid, kkgrid, rgrid, rkgrid = make_grids(img.shape)
        t = self.generate_transforms(num_shots, [0,0,0])
        w = np.ones(num_shots)
        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)
        #M = np.ones(img.shape, dtype=bool)
        
        #pl.ImagePlot(M)
        recon = MotionCorruptedImageRecon(ksp, mps, mask, transforms=t, constraint=M, max_joint_iter=100, tol=1e-8,device=sp.Device(0)).run()
            #print(i, t[:, 3:] * 180 / np.pi)
        recon = recon.get()
        pl.ImagePlot(np.concatenate([img, recon], axis=-1))
        npt.assert_allclose(img, recon, atol=1e-2)