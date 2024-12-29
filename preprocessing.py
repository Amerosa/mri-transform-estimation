import numpy as np
import sigpy as sp
import sigpy.mri
import sigpy.plot as pl
import time
from numba import njit
import twixtools

#@njit(fastmath=True, parallel=True, cache=True)
def coil_compression(mps, ksp=None, perc=0.95, reg=1e-6):
    
    #Compute P value
    #mps are [nc, kx, ky, kz]
    num_coils = mps.shape[0]
    img_shape = mps.shape[1:]

    mps_conj = mps.conj()
    norm = 1 / (np.sum(mps * mps.conj(), axis=0) + reg)

    mps = mps.reshape((num_coils, -1))
    mps_conj = mps.conj().reshape((num_coils, -1))
    P = (mps * norm.flatten()) @ mps_conj.T
    
    #Threshold the singular values and vectors
    U, S, _ = np.linalg.svd(P)
    print(S)
    singular_total = np.sum(S)
    num_reduced_coils = num_coils - ((np.cumsum(S) / singular_total) >= perc).sum()
    print(f'Reducing from {num_coils} -> {num_reduced_coils} coils')

    #Make the compression matrix
    A = U.conj().T[:num_reduced_coils]
    compressed_mps = (A @ mps).reshape(num_reduced_coils, *img_shape)

    if ksp is not None:
        compressed_ksp = (A @ ksp.reshape(num_coils, -1)).reshape(num_reduced_coils, *img_shape) 

    return compressed_mps, compressed_ksp

@njit(fastmath=True, parallel=True, cache=True)
def coil_compression_fast(mps):

    num_shots = mps.shape[0]
    mps = mps.reshape((num_shots, -1))
    P = np.reciprocal(mps.conj() @ mps.T)
    P = P @ mps.conj()
    P = mps @ P
    print(P.shape)

def get_kspaces(filename):
    multi_twix = twixtools.read_twix(filename, parse_pmu=False)
    
    # map the twix data to twix_array objects
    mapped = twixtools.map_twix(multi_twix)
    mapped_img_data = mapped[-1]['image']
    mapped_refscan_data = mapped[-1]['refscan']
    
    # make sure that we later squeeze the right dimensions:
    print(f'Img Data non singleton dims : {mapped_img_data.non_singleton_dims}')
    print(f'RefScan Data non singleton dims : {mapped_refscan_data.non_singleton_dims}')
    
    # remove 2x oversampling and zero pad to ensure same shape
    mapped_img_data.flags['remove_os'] = True
    mapped_img_data.flags['zf_missing_lines'] = True
    mapped_refscan_data.flags['remove_os'] = True
    mapped_refscan_data.flags['zf_missing_lines'] = True
    
    
    image_ksp = mapped_img_data[:].squeeze()
    refscan_ksp = mapped_refscan_data[:].squeeze()
    print(f'Dimensions of image k-space is {image_ksp.shape}')
    print(f'Dimensions of refscan k-space is {refscan_ksp.shape}')
    
    #Rearrange so that shape follows format of [nc, par, line, col]
    image_ksp = np.transpose(image_ksp, (2, 0, 1 ,3))
    refscan_ksp = np.transpose(refscan_ksp, (2, 0, 1, 3))
    print(f'New dimenions are [nc, par, line, col] -> {refscan_ksp.shape}')

    return image_ksp, refscan_ksp


if __name__ == '__main__':
    #mps = np.load(r"C:\Users\giuse\OneDrive\Documents\mri-physics\mri-transform-estimation\data\test_sample\uncomp_maps.npy")
    #ksp = np.load(r"C:\Users\giuse\OneDrive\Documents\mri-physics\mri-transform-estimation\data\test_sample\uncomp_kspace.npy")
    filepath = r"C:\Users\giuse\OneDrive\Documents\mri-physics\mri-transform-estimation\data\MR160-105-0813-02_T1_twix\meas_MID00097_FID16376_T1_Sag_MPRAGE_p2_TR_1870_0_8mm.dat"

    #start = time.perf_counter() 
    #coil_compression(mps)
    #end = time.perf_counter()
    #pl.ImagePlot(mps, z=0)
    #start = time.perf_counter() 
    #mps, ksp = coil_compression(mps, perc=0.95, ksp=ksp)

    #end = time.perf_counter()
    #print(end - start)
    
    ksp, ref_ksp = get_kspaces(filepath)

    print(ksp.dtype, ref_ksp.dtype)
    ref_img = sp.ifft(ref_ksp, axes=(1,2,3))
    magnitude = np.sum(np.abs(ref_img)**2, axis=(0,1,2))
    threshold = np.percentile(magnitude, 10)
    desired_indicies = np.argwhere(magnitude>threshold).flatten()

    refscan_partial_space = sp.fft(ref_img, axes=[1,2])
    device = sp.Device(0)
    print('estimating mps...')
    from tqdm import tqdm
    pbar = tqdm(total=len(desired_indicies), desc='Estimating mps', leave=True)
    with device:
        xp = device.xp 
        mps = xp.zeros(refscan_partial_space.shape, dtype=np.complex64)
        for e, i in enumerate(desired_indicies):
            pbar.update()
            mps[:,:,:,i] =  sigpy.mri.app.EspiritCalib(refscan_partial_space[:,:,:,i], device=device, show_pbar=False).run()
    pbar.close()
    mps = sp.to_device(mps)
    img_combined = np.sqrt(np.sum(ref_img**2, axis=0))
    t = 0.1 * np.max(np.abs(img_combined))
    mask = np.abs(img_combined) > t

    kernel = np.ones((16,16,16))/16**2
    #smoothed = np.empty(normalized.shape, dtype=normalized.dtype)
    from scipy.signal import convolve
    #for c in range(normalized.shape[0]):
    #    smoothed[c] = convolve(normalized[c], kernel, mode='same')
    mask = convolve(mask, kernel, mode='same')

    #pl.ImagePlot(mask)
    mps *= mask
    #pl.ImagePlot(mps, z=0, x=2, y=3)
    #pl.ImagePlot(mps, z=0, x=1, y=2)

    mps, ksp = coil_compression(mps, ksp)

    #pl.ImagePlot(mps, z=0, x=2, y=3)
    #pl.ImagePlot(mps, z=0, x=1, y=2)
    #pl.ImagePlot(sp.ifft(ksp, axes=[-3,-2,-1]))
    norm = np.linalg.norm(sp.ifft(ksp, axes=[-3,-2,-1]))
    recon = sigpy.mri.app.SenseRecon(ksp / norm, mps, device=device).run()
    recon *= norm
    pl.ImagePlot(recon)
