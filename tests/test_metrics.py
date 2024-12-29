import unittest

from scipy.ndimage import gaussian_filter
from sigpy.mri import sim
import sigpy as sp
import numpy as np
import numpy.testing as npt

from estimation.utils import generate_corrupt_kspace, generate_sampling_mask, generate_transforms
from estimation.factors import generate_transform_grids
from estimation.metrics import objective_all_shot
from estimation.joint_recon import MotionCorruptedImageRecon
if __name__ == '__main__':
    unittest.main()

class TestObjectiveFunction(unittest.TestCase):

    def corrupt_shepp_logan_setup(self, num_shots, rotations, random=False):
        #img_shape = [32, 32, 32]
        #mps_shape = [8, 32, 32, 32]

        img_shape = [128, 128, 128]
        mps_shape = [8, 128, 128, 128]

        img = sp.shepp_logan(img_shape)
        img = gaussian_filter(img, 1)
        mps = sim.birdcage_maps(mps_shape)

        mask = generate_sampling_mask(num_shots, img_shape)
        transforms = generate_transforms(num_shots, rotations, random)
        ksp = generate_corrupt_kspace(img, mps, mask, transforms)

        return img, mps, ksp, mask, transforms
    
    def test_zero(self):
        img, mps, ksp, mask, transforms = self.corrupt_shepp_logan_setup(10, [10,4,2])
        wrong_transforms = generate_transforms(10, [10,2,2])
        kgrid,_,_, rkgrid = generate_transform_grids(img.shape)
        resid = objective_all_shot(img, ksp, mps, mask, transforms, kgrid, rkgrid)
        expected = np.zeros(1)
        #MotionCorruptedImageRecon(ksp, mps, mask, img, transforms).run()
        print(resid, expected)
        npt.assert_almost_equal(resid, expected)
        
