import unittest

import numpy as np
import numpy.testing as npt
import sigpy as sp
import sigpy.plot as pl
from sigpy.mri.linop import Sense
from sigpy.mri.app import _estimate_weights, SenseRecon
from sigpy.app import LinearLeastSquares
from estimation.algos import TransformEstimation, JointEstimation, TransformEstimationRecon
from estimation.linop import RigidTransform

from scipy.stats import multivariate_normal
import scipy.io as sio

def setup_image(shape):
    radius = 0.85
    sx, sy, sz = shape
    x = np.linspace(-1,1,sx)
    y = np.linspace(-1,1,sy)
    z = np.linspace(-1,1,sz)
    x, y, z = np.meshgrid(x,y,z, indexing='ij')
    covariance = np.diag([0.25,0.25,0.25])
    pos = np.stack((x, y, z), axis=-1)
    circle = np.sqrt(x**2 + y**2 + z**2).astype(np.complex128)
    imag_part = multivariate_normal([0,0,0], covariance).pdf(pos)
    circle += imag_part
    circle[circle > radius] = 0 + 0j
    return circle

def setup_maps(channels, shape):
    assert len(shape) == 3
    result = np.zeros([channels] + shape, dtype=np.complex128)
    x1 = np.linspace(-0.75,.75, channels)
    means = np.repeat(x1, 3).reshape(channels,3)
    for c, mean in zip(range(channels), means):
        sx, sy, sz = shape
        x = np.linspace(-1,1,sx)
        y = np.linspace(-1,1,sy)
        z = np.linspace(-1,1,sz)
        x, y, z = np.meshgrid(x,y,z, indexing='ij')
        pos = np.stack((x, y, z), axis=-1)
        covariance = np.diag([0.25, 0.25, 0.25])
        real_part = multivariate_normal(mean, covariance).pdf(pos)
        imag_part = multivariate_normal(mean, covariance).pdf(pos)
        result[c] = real_part + 1j* imag_part
    return result

def setup_phantom(shape):
    sx, sy, sz = 2 / np.array(shape)
    a,b,c = [0.75,0.65,0.8]
    x0, y0, z0 = [0,0,0]
    x, y, z = np.ogrid[-1:1:sx, -1:1:sy, -1:1:sz]
    ellipse = ((x-x0)/a)**2 + ((y-y0)/b)**2 + ((z-z0)/c)**2 <= 1 
    return ellipse.astype(np.complex64)

ROT_CASES = [
            np.array([0,0,0,5,0,0], dtype=np.float64),
            np.array([0,0,0,0,5,0], dtype=np.float64),
            np.array([0,0,0,0,0,5], dtype=np.float64)]

TRANS_CASES = [
            np.array([0,0,1.2,0,0,0], dtype=np.float64),
            np.array([0,1.2,0,0,0,0], dtype=np.float64),
            np.array([1.2,0,0,0,0,0], dtype=np.float64)]

MIXED_CASES = [    
            np.array([0,0,0,5,10,3], dtype=np.float64),
            np.array([0.7,0.8,1.1,0,0,0], dtype=np.float64),
            np.array([0,1.1,0,5,0,0], dtype=np.float64)]


class TestRigidTransforms(unittest.TestCase):
    def setUp(self):
        img_shape = [16,16,16]
        channels = 10

        self.mps = setup_maps(channels, img_shape)
        self.image = setup_phantom(img_shape)
        print(self.image.shape)
        pl.ImagePlot(self.image, z=1, y=0, x=-1)
        w = _estimate_weights(self.image, None, None)
        self.S = Sense(self.mps, weights=w)
        

    def test_Translations(self):
        
        for t in TRANS_CASES:
            with self.subTest(t=t):
                t[3:] *= np.pi / 180
                states = RigidTransform(t, self.image.shape, sp.cpu_device)
                rot_image = states.Transform() * self.image
                rot_kspace = self.S * rot_image
                
                #pl.ImagePlot(rot_image, title=t)
                
                parameters = np.zeros(6, dtype=np.float64)
                states = RigidTransform(parameters, self.image.shape, sp.cpu_device)
                parameters, _ = TransformEstimationRecon(self.S, states, self.image, rot_kspace, max_iter=6).run()
                
                #pl.ImagePlot(states.Transform() * image, title='Recon')
                npt.assert_allclose(parameters, t, atol=1e-3)


class TestJointEstimation(unittest.TestCase):
    def setUp(self):
        img_shape = [64,64,64]
        channels = 10

        data = sio.loadmat(r"C:\Users\giuse\OneDrive\Documents\mri-physics\mri-transform-estimation\data\xGT.mat")
        image = data['xGT']
        mps = data['S']
        self.mps = np.transpose(mps, [-1, 0, 1 ,2])
        self.image = np.atleast_3d(image)
        print(self.image.shape)

        #self.mps = setup_maps(channels, img_shape)
        #self.image = setup_phantom(img_shape)
        w = _estimate_weights(self.image, None, None)
        self.S = Sense(self.mps, weights=w)

        

        bins = 8
        bin_size = int(self.image.shape[0] / 8)
        self.kspace = np.zeros(self.mps.shape, dtype=np.complex64)
        rots = np.linspace(-2.5, 2.5, bins, endpoint=True) * np.pi / 180
        rotations = np.array([[0,0,0,0,0,r] for r in rots])
        for b, params in enumerate(rotations):
            rt = RigidTransform(params, self.image.shape)
            rot_image = rt.Transform() * self.image
            rot_kspace = self.S * rot_image
            #pl.ImagePlot(np.squeeze(self.image), title='Original Unmoved Image')
            #pl.ImagePlot(np.squeeze(rot_image), title=f'Rotation {params}')
            self.kspace[:, b*bin_size:(b+1)*bin_size] = rot_kspace[:, b*bin_size:(b+1)*bin_size]     

    def test_JointEstimation(self):
        pl.ImagePlot(np.squeeze(self.image), title='Original Unmoved Image')
        pl.ImagePlot(np.squeeze(self.S.H * self.kspace), title='Motion Corrupted image')

        alg_method = JointEstimation(self.kspace, self.mps, 8, 16, 3, 1, 100, tol=1e-12)
        while not alg_method.done():
            #pl.ImagePlot(np.squeeze(alg_method.recon_image), title=f'Recon Image Iteration {alg_method.iter}')
            alg_method.update()

        pl.ImagePlot(np.squeeze(alg_method.recon_image), title='Recon Image')
        pl.ImagePlot(np.squeeze(self.image), title='Ground Truth')  

        #npt.assert_allclose(alg_method.tform_states.parameters, params, atol=1e-3)
        npt.assert_allclose(alg_method.recon_image, self.image, atol=1e-3)




class TestAlgos(unittest.TestCase):
    
    def test_TransformEstimation(self):
        device = sp.Device(-1)
        ROT_CASES = [
            np.array([0,0,0,5,0,0], dtype=np.float64),
            np.array([0,0,0,0,5,0], dtype=np.float64),
            np.array([0,0,0,0,0,5], dtype=np.float64),
            np.array([0,0,1.2,0,0,0], dtype=np.float64),
            np.array([0,1.2,0,0,0,0], dtype=np.float64),
            np.array([1.2,0,0,0,0,0], dtype=np.float64),
            np.array([0,0,0,5,10,3], dtype=np.float64),
            np.array([0.7,0.8,1.1,0,0,0], dtype=np.float64),
            np.array([0,1.1,0,5,0,0], dtype=np.float64)]
        
        init_parameters = np.array([0,0,1.2,0,0,0], dtype=np.float64)
        #init_parameters *= np.pi / 180

        img_shape = [32,32,32]
        channels = 10

        mps = self.setup_maps(channels, img_shape)
        image = self.setup_image(img_shape)
        w = _estimate_weights(image, None, None)
        S = Sense(mps, weights=w)

        #state = RigidTransform(init_parameters, image.shape, device)
        #t_image = state.Transform() * image
        #t_kspace = S * t_image
        #pl.ImagePlot(t_image)
        #pl.ImagePlot(state.Transform().H * t_image)
        
        for t in ROT_CASES:
            with self.subTest(t=t):
                t[3:] *= np.pi / 180
                states = RigidTransform(t, image.shape, device)
                rot_image = states.Transform() * image
                rot_kspace = S * rot_image
                pl.ImagePlot(rot_image, title=t)
                parameters = np.zeros(6, dtype=np.float64)
                states.update_state(parameters)
                alg_method = TransformEstimation(S, states, image, rot_kspace, 10)
        
                while not alg_method.done():
                    alg_method.update()
                pl.ImagePlot(states.Transform() * image, title='Recon')
                npt.assert_allclose(states.parameters, t, atol=1e-3)
        

    def test_JointEstimation(self):
        device = sp.Device(-1)
        img_shape = [32,32,32]
        channels = 10
        init_parameters = np.array([0,0,0,5,0,0], dtype=np.float64)
        init_parameters[3:] *= np.pi / 180
        mps = setup_maps(channels, img_shape)
        image = setup_image(img_shape)
        states = RigidTransform(init_parameters, image.shape, device)
        w = _estimate_weights(image, None, None)
        S = Sense(mps, weights=w)
        rot_image = states.Transform() * image
        rot_kspace = S * rot_image

        alg_method = JointEstimation(rot_kspace, mps, 3, 1, 100, tol=1e-6)
        while not alg_method.done():
            #alg_method.tform_states.parameters = init_parameters
            alg_method.update()

        pl.ImagePlot(np.squeeze(alg_method.recon_image))
        pl.ImagePlot(alg_method.tform_states.Transform() * np.squeeze(alg_method.recon_image))
        pl.ImagePlot(np.squeeze(rot_image))    

        npt.assert_allclose(states.parameters, init_parameters, atol=1e-3)
        npt.assert_allclose(alg_method.recon_image, image, atol=1e-3)
        
    def test_tform(self):
        data = sio.loadmat(r"C:\Users\giuse\OneDrive\Documents\mri-physics\mri-transform-estimation\data\xGT.mat")
        image = data['xGT']
        mps = data['S']
        mps = np.transpose(mps, [-1, 0, 1 ,2])
        #mps = np.repeat(mps, 3, axis=-1)
        #image = np.repeat(image[...,np.newaxis], 1, axis=2)
        image = image[...,np.newaxis]
        w = _estimate_weights(image, None, None)
        S = Sense(mps, weights=w)

        ROT_CASES = [
            np.array([0,0,0,5,0,0], dtype=np.float64),
            np.array([0,0,0,0,5,0], dtype=np.float64),
            np.array([0,0,0,0,0,5], dtype=np.float64),
            np.array([0,0,2,0,0,0], dtype=np.float64),
            np.array([0,2,0,0,0,0], dtype=np.float64),
            np.array([2,0,0,0,0,0], dtype=np.float64),
            np.array([0,0,0,5,10,3], dtype=np.float64),
            np.array([3,10,5,0,0,0], dtype=np.float64),
            np.array([0,5,0,5,0,0], dtype=np.float64)]

        for t in ROT_CASES:
            with self.subTest(t=t):
                t[3:] *= np.pi / 180
                states = RigidTransform(t, image.shape, sp.cpu_device)
                rot_image = states.Transform() * image
                rot_kspace = S * rot_image
                pl.ImagePlot(rot_image, title=t)

                parameters = np.zeros(6, dtype=np.float64)
                states.update_state(parameters)
                alg_method = TransformEstimation(S, states, image, rot_kspace, 6)

                
                while not alg_method.done():
                    alg_method.update()
                pl.ImagePlot(states.Transform() * image, title='Recon')
                npt.assert_allclose(states.parameters, t, atol=1e-3)

        """     def test_FixedTformRecon(self):
        device = sp.Device(-1)
        img_shape = [64,64,64]
        channels = 5
        init_parameters = np.array([0,0,0,5,0,0], dtype=np.float64)
        init_parameters *= np.pi / 180
        mps = self.setup_maps(channels, img_shape)
        image = self.setup_image(img_shape)
        states = RigidTransform(init_parameters, image.shape, device)
        w = _estimate_weights(image, None, None)
        S = Sense(mps, weights=w)
        rot_image = states.Transform() * image
        rot_kspace = S * rot_image




        #new_parameters = np.zeros(6, dtype=np.float64)
        #states.update_state(new_parameters)
        recon_image = LinearLeastSquares(S * states.Transform(), rot_kspace, max_iter=100, tol=1e-8).run()
        pl.ImagePlot(recon_image)
        pl.ImagePlot(rot_image)
        npt.assert_allclose(recon_image, rot_image, atol=1e-3) """

    def test_realishJoint(self):
        data = sio.loadmat(r"C:\Users\giuse\OneDrive\Documents\mri-physics\mri-transform-estimation\data\xGT.mat")
        image = data['xGT']
        mps = data['S']
        mps = np.transpose(mps, [-1, 0, 1 ,2])
        image = np.atleast_3d(image)
        init_parameters = np.array([0,0,0.7,0,0,10], dtype=np.float64)
        init_parameters[3:] *= np.pi / 180
        states = RigidTransform(init_parameters, image.shape, sp.backend.Device(-1))
        w = _estimate_weights(image, None, None)
        S = Sense(mps, weights=w)
        rot_image = states.Transform() * image
        rot_kspace = S * rot_image

        alg_method = JointEstimation(rot_kspace, mps, 3, 1, 100, tol=1e-6)
        while not alg_method.done():
            alg_method.update()

        pl.ImagePlot(np.squeeze(alg_method.tform_states.Transform().H * alg_method.recon_image), title='recon image')
        pl.ImagePlot(np.squeeze(rot_image), title='og rotated image')    

        npt.assert_allclose(states.parameters, init_parameters, atol=1e-3)
        npt.assert_allclose(alg_method.recon_image, image, atol=1e-3)


if __name__ == "__main__":
    unittest.main()