import unittest

import numpy as np
import numpy.testing as npt

import sigpy as sp
import sigpy.plot as pl
from sigpy.mri import app, sim
import scipy.ndimage
from scipy.ndimage import gaussian_filter

from estimation.joint_recon import MotionCorruptedImageRecon
from estimation.factors import make_grids, calc_factors, generate_transform_grids
from estimation.transforms import RigidTransform
from estimation.transform_solver import LevenbergMarquardt
from estimation.image_solver import ImageEstimation
from estimation.utils import generate_corrupt_kspace, generate_transforms, generate_sampling_mask

if __name__ == "__main__":
    unittest.main()

class TestApp(unittest.TestCase):
    def shepp_logan_setup(self, num_shots=2):
        img_shape = [32, 32, 32]
        mps_shape = [8, 32, 32, 32]
        #img_shape = [64, 64, 64]
        #mps_shape = [8, 64, 64, 64]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)
        mask = self.generate_sampling_mask(num_shots, img_shape)

        ksp = np.sum(mask[:, np.newaxis] * sp.fft(mps * img, axes=[-3, -2, -1]), axis=0)
        return img, mps, ksp, mask
        
    def corrupt_shepp_logan_setup(self, num_shots, translations, rotations, resample_shape=None, random=False):
        img_shape = [32, 32, 32]
        mps_shape = [10, 32, 32, 32]

        #img_shape = [64, 64, 64]
        #mps_shape = [10, 64, 64, 64]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        mask = generate_sampling_mask(num_shots, img_shape)
        transforms = generate_transforms(num_shots, translations, rotations, random)
        kgrid, _, _, rkgrid = generate_transform_grids(img.shape, resample_shape)
        factors_trans, factors_tan, factors_sin = calc_factors(transforms, kgrid, rkgrid)
        T = RigidTransform(img.shape, factors_trans, factors_tan, factors_sin)
        #pl.ImagePlot(T * img, z=0)
        ksp = generate_corrupt_kspace(img, mps, mask, transforms)

        return img, mps, ksp, mask, transforms
    
    def test_shepp_logan_ImageEstimation(self):
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(6, [4,8,10], [20,10,8], random=False)
        kgrid, _, _, rkgrid = generate_transform_grids(img.shape)
        
        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)

        precond = 1 / (np.sum(np.abs(mps)**2, axis=0) + 1e-3)
        P = sp.linop.Multiply(img.shape, precond)
        from sigpy import config
        config.nccl_enabled = False

        comm = sp.Communicator()

        transforms = transforms[comm.rank :: comm.size].copy()
        mask = mask[comm.rank :: comm.size].copy()

        img_estimate = ImageEstimation(ksp, mps, mask, transforms, kgrid, rkgrid, P=P, constraint=M, shot_chunk_size=None, comm=comm, max_iter=20, device=sp.Device(0)).run()
        npt.assert_allclose(img, img_estimate.get(), atol=1e-2, rtol=1e-2)

    def test_shepp_logan_ImageEstimation_subsampled_image(self):
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(6, [4,8,10], [20,10,8], random=False)
        resample_shape = img.shape
        from estimation.utils import resample
        ksp = resample(ksp, 2, axes=[-3,-2], is_kspace=True)
        mps = resample(mps, 2, axes=[-3,-2], is_kspace=False)
        mask = resample(mask, 2, axes=[-3,-2], is_kspace=True)
        img = resample(img, 2, axes=[-3,-2])

        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)
        img *= M
        precond = 1 / (np.sum(np.abs(mps)**2, axis=0) + 1e-3)
        P = sp.linop.Multiply(img.shape, precond)

        kgrid, _, _, rkgrid = generate_transform_grids(resample_shape, img.shape)

        comm = sp.Communicator()

        transforms = transforms[comm.rank :: comm.size].copy()
        mask = mask[comm.rank :: comm.size].copy()

        import sigpy.plot as pl
        pl.ImagePlot(img)
        ftr,ft,fs = calc_factors(transforms, kgrid, rkgrid)
        pl.ImagePlot(RigidTransform(img.shape,ftr,ft,fs) * img, z=0)

        img_estimate = ImageEstimation(ksp, mps, mask, transforms, kgrid, rkgrid, P=P, constraint=M, shot_chunk_size=None, comm=comm, device=sp.Device(0)).run()
        npt.assert_allclose(img, img_estimate.get(), atol=1e-2, rtol=1e-2)


    def test_shepp_logan_LevenbergMarquart(self):
        import sigpy.plot as pl
        num_shots = 8
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [2,2,1], [10,8,4])
        #ksp /= np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        kgrid, kkgrid, rgrid, rkgrid = generate_transform_grids(img.shape, device=sp.Device(0))
        estimates = np.zeros((num_shots,6), dtype=float)
        w = np.ones((num_shots, 6))
        #w[:, :3] = 1e4
        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)
        img = sp.to_device(img, sp.Device(0))
        ksp = sp.to_device(ksp, sp.Device(0))
        mask = sp.to_device(mask, sp.Device(0))
        w = sp.to_device(w, sp.Device(0))
        estimates = sp.to_device(estimates, sp.Device(0))
        M = sp.to_device(M, sp.Device(0))

        xp = sp.get_array_module(estimates)
        comm = sp.Communicator()
        print(comm.rank, comm.size)
        chunk_transforms = xp.array_split(estimates, comm.size)[comm.rank].copy()
        chunk_masks = xp.array_split(mask, comm.size)[comm.rank].copy()
        #[4, 4, 4, 4] sized chunks on the gpu, but only using one each for gpu
    
        #chunk_transforms = sp.to_device(chunk_transforms, sp.Device(0))
        #chunk_masks = sp.to_device(chunk_masks, sp.Device(0))

        alg = LevenbergMarquardt(mps, chunk_masks, chunk_transforms, img, ksp, kgrid, kkgrid, rgrid,rkgrid, w,M, 20, comm=comm)
        while not alg.done():
            alg.update()
        app = sp.app.App(alg)
        app.run()
        estimates = comm.gatherv(app.alg.transforms)
        if comm.rank == 0:
            result = estimates.reshape((-1,6))
            print(result)
            npt.assert_allclose(transforms, result.get(), atol=1e-2, rtol=1e-2)

    def test_shepp_logan_LevenbergMarquartTESst(self):
        from estimation.transform_solver import TransformEsitmation
        num_shots = 8
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [2,2,1], [10,8,4])
        #ksp /= np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        kgrid, kkgrid, rgrid, rkgrid = generate_transform_grids(img.shape, device=sp.Device(0))
        import cupy as cp
        estimates = cp.zeros((num_shots,6), dtype=float)
        damp = cp.ones((num_shots, 6))
        #w[:, :3] = 1e4
        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)
        
        img = sp.to_device(img, sp.Device(0))
        ksp = sp.to_device(ksp, sp.Device(0))
        mask = sp.to_device(mask, sp.Device(0))
        #w = sp.to_device(w, sp.Device(0))
        #estimates = sp.to_device(estimates, sp.Device(0))
        #M = sp.to_device(M, sp.Device(0))

        #xp = sp.get_array_module(estimates)
        comm = sp.Communicator()
        #chunk_transforms = xp.array_split(estimates, comm.size)[comm.rank].copy()
        #chunk_masks = xp.array_split(mask, comm.size)[comm.rank].copy()

        TransformEsitmation(estimates, img, ksp, mps, mask, kgrid, kkgrid, rkgrid, damp, device=sp.Device(0)).run()
        npt.assert_allclose(transforms, estimates, atol=1e-2, rtol=1e-2)

    def test_shepp_logan_resample_TransformEstimation(self):
        from estimation.utils import resample, generate_transforms, generate_sampling_mask
        img_shape = [128, 128, 128]
        mps_shape = [10, 128, 128, 128]
        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        num_shots = 2
        translations = [0,0,2]
        rotations = [10,0,0]

        transforms = generate_transforms(num_shots, translations, rotations)
        mask = generate_sampling_mask(num_shots, img.shape)
        ksp = generate_corrupt_kspace(img, mps, mask, transforms)

        norm = np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        ksp /= norm

        ksp_in = resample(ksp, 2, axes=[-3,-2], is_kspace=True)
        mps_in = resample(mps, 2, axes=[-3,-2], is_kspace=False)
        mask_in = resample(mask, 2, axes=[-3,-2], is_kspace=True)
        
        ground_truth = resample(img, 2, axes=[-3,-2])
        rss = np.sqrt(np.sum(np.abs(ground_truth)**2, axis=0))
        M = rss > (np.max(rss) * 0.1)
        w = np.ones((num_shots, 6))


        comm = sp.Communicator()
        print(comm.rank, comm.size)
        estimates = np.zeros((num_shots,6), dtype=float)
        chunk_transforms = np.array_split(estimates, comm.size)[comm.rank].copy()
        chunk_masks = np.array_split(mask_in, comm.size)[comm.rank].copy()
        #[4, 4, 4, 4] sized chunks on the gpu, but only using one each for gpu
    
        #chunk_transforms = sp.to_device(chunk_transforms, sp.Device(0))
        #chunk_masks = sp.to_device(chunk_masks, sp.Device(0))
        kgrid, kkgrid, rgrid, rkgrid = generate_transform_grids(img.shape, ground_truth.shape, device=sp.Device(-1))
        alg = LevenbergMarquardt(mps_in, chunk_masks, chunk_transforms, ground_truth, ksp_in, kgrid, kkgrid, rgrid,rkgrid, w,M, 20, comm=comm)
        while not alg.done():
            alg.update()
        estimates = comm.gatherv(alg.transforms)
        if comm.rank == 0:
            result = estimates.reshape((-1,6))
            print(result)
            npt.assert_allclose(transforms, result.get(), atol=1e-2, rtol=1e-2)

    def test_shepp_logan_recon_lowres(self):
        from estimation.utils import resample, generate_transforms, generate_sampling_mask
        img_shape = [128, 128, 128]
        mps_shape = [10, 128, 128, 128]
        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        num_shots = 2
        rotations = [10,0,0]

        transforms = generate_transforms(num_shots, rotations)
        mask = generate_sampling_mask(num_shots, img.shape)
        ksp = generate_corrupt_kspace(img, mps, mask, transforms)

        norm = np.max(np.abs(sp.ifft(ksp, axes=[-3,-2,-1])))
        ksp /= norm

        ksp_in = resample(ksp, 2, axes=[-3,-2], is_kspace=True)
        mps_in = resample(mps, 2, axes=[-3,-2], is_kspace=False)
        mask_in = resample(mask, 2, axes=[-3,-2], is_kspace=True)
        
        ground_truth = resample(img, 2, axes=[-3,-2])
        rss = np.sqrt(np.sum(np.abs(ground_truth)**2, axis=0))
        M = rss > (np.max(rss) * 0.1)
        p = 1 / (np.sum(np.real(mps_in * mps_in.conj()), axis=0) + 1e-3)
        P = sp.linop.Multiply(mps_in.shape[1:], p)
        t_init = self.generate_transforms(num_shots, [8,0,0])
        recon, estimates = MotionCorruptedImageRecon(ksp_in, mps_in, mask_in, ksp, mps, mask, ground_truth, img=None, transforms=t_init, constraint=M, P=P, max_joint_iter=400, tol=0,device=sp.Device(0), verbose=True).run()
        recon = recon.get()
        pl.ImagePlot(np.concatenate([ground_truth, recon*norm], axis=-1))
        npt.assert_allclose(transforms, estimates.get(), atol=1e-2, rtol=1e-2)
        npt.assert_allclose(img, recon*norm, atol=1e-2, rtol=1e-2)

    def test_shepp_logan_recon(self):
        num_shots = 4
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [2,2,1], [10,8,4])
        #img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [2,0,1], [0,2,0])
        import sigpy.plot as pl

        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)

        precond = 1 / (np.sum(np.abs(mps)**2, axis=0) + 1e-3)
        P = sp.linop.Multiply(img.shape, precond)
        
        #pl.ImagePlot(M)
        recon, estimates, objs, params = MotionCorruptedImageRecon(ksp, mps, mask, img.shape, 
                                                     img=img, transforms=transforms, 
                                                     constraint=M, P=P,
                                                     device=sp.Device(0), verbose=False,
                                                     max_joint_iter=60,
                                                     save_objective_values=True).run()

        recon = recon.get()
        estimates = estimates.get()
        pl.ImagePlot(np.concatenate([img, recon], axis=-1))
        npt.assert_allclose(transforms, estimates, atol=1e-2, rtol=1e-2)
        npt.assert_allclose(img, recon, atol=1e-2, rtol=1e-2)

    def test_shepp_logan_recon_with_MPICOMM(self):
        num_shots = 4
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [2,2,1], [10,8,4])
        #img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(num_shots, [2,0,1], [0,2,0])
        import sigpy.plot as pl

        M = np.sqrt(np.sum(np.abs(img)**2, axis=0))
        M = M > (np.max(M) * 0.01)

        precond = 1 / (np.sum(np.abs(mps)**2, axis=0) + 1e-3)
        P = sp.linop.Multiply(img.shape, precond)
        from sigpy import config
        config.nccl_enabled = False
        comm = sp.Communicator()

        #mask = mask[comm.rank::comm.size]
        mask = np.array_split(mask, comm.size)[comm.rank]
        #tinit = np.zeros(transforms.shape, dtype=float)[comm.rank::comm.size]
        recon, estimates, objs, params = MotionCorruptedImageRecon(ksp, mps, mask, img.shape, 
                                                     img=None, transforms=None, 
                                                     constraint=M, P=P,
                                                     device=sp.Device(0), verbose=False,
                                                     max_joint_iter=60,
                                                     save_objective_values=True, comm=comm).run()

        estimates = comm.gatherv(estimates, root=0)
        if comm.rank == 0:
            #result = estimates.reshape((-1,6))
            #print(estimates)
            estimates = estimates.reshape((-1,6))
            estimates = sp.to_device(estimates)
            npt.assert_allclose(transforms, estimates, atol=1e-2, rtol=1e-2)
            recon = sp.to_device(recon)
        #recon = recon.get()
        #estimates = estimates.get()
            pl.ImagePlot(np.concatenate([img, recon], axis=-1))
        #npt.assert_allclose(transforms, estimates, atol=1e-2, rtol=1e-2)
            npt.assert_allclose(img, recon, atol=1e-2, rtol=1e-2)