import unittest

from scipy.ndimage import gaussian_filter
from sigpy.mri import sim
import sigpy as sp
import numpy as np
import numpy.testing as npt

from estimation.utils import generate_corrupt_kspace, generate_sampling_mask, generate_transforms, resample
from estimation.metrics import objective_all_shot
if __name__ == '__main__':
    unittest.main()

class TestSampling(unittest.TestCase):
    def test_sampling(self):
        x = np.arange(11*100*200*300).reshape((11,100,200,300))
        result_sub_one = x[:, 25:75, 50:150, :]
        result_sub_two = x[:, 38:63, 75:125, :]
        
        img = np.arange(100*100*100).reshape((100,100,100))
        result = img[25:75, 25:75, 25:75]

        npt.assert_array_equal(result_sub_one, resample(x, 1, axes=(1,2)))
        npt.assert_array_equal(result_sub_two, resample(x, 2, axes=(1,2)))
        npt.assert_array_equal(result_sub_two, resample(x, 2, axes=(-3,-2)))
        npt.assert_array_equal(result, resample(img, 1))
    
    def test_shepp_sampling(self):
        img_shape = [128, 128, 128]
        mps_shape = [8, 128, 128, 128]
        img = sp.shepp_logan(img_shape)
        mps = sim.birdcage_maps(mps_shape)

        import sigpy.plot as pl
        pl.ImagePlot(resample(img, 1, axes=(-3,-2,-1), is_kspace=False))
        pl.ImagePlot(resample(mps, 1, axes=(-3,-2,-1), is_kspace=False))
        npt.assert_allclose(img, mps)
    