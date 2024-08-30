import argparse
import numpy as np
from sigpy.mri.linop import Sense
from sigpy.mri.app import _estimate_weights
import sigpy.plot as pl
from estimation.linop import RigidTransform

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Raw dat input file")
    parser.add_argument("--output", help="Output file for the reconstructed image")
    parser.add_argument("--maps", help="File for sensitivity maps if already generated")
    parser.add_argument("--bins", type=int, help="Number of linescans we transform estimate at once")
    parser.add_argument("--bin_size", type=int, help="Size of bins")
    
    args = parser.parse_args()

    sense_recon = np.load(args.input)
    mps = np.load(args.maps)
    bins = args.bins
    bin_size = args.bin_size

    kspace = np.zeros(mps.shape, dtype=np.complex64)
    w = _estimate_weights(sense_recon, None, None)
    S = Sense(mps, weights=w)

    #Generate rotations and extract kspace section to stich into new motion corrupted kspace
    rots = np.linspace(-2.5, 2.5, bins, endpoint=True, dtype=np.float64) * np.pi / 180.
    rotations = np.array([[0,0,0,r,0,0] for r in rots])
    for b, params in enumerate(rotations):
        T = RigidTransform(params, sense_recon.shape).Transform()
        rot_image = T * sense_recon
        rot_kspace = S * rot_image
        kspace[:, b*bin_size:(b+1)*bin_size, :, :] = rot_kspace[:, b*bin_size:(b+1)*bin_size, :, :]

    np.save(args.output, kspace)

if __name__ == '__main__':
    main()