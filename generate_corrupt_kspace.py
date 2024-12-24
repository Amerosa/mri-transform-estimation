import argparse
import sigpy as sp
import numpy as np
from estimation.encoding import Encoding
from estimation.factors import generate_transform_grids
from estimation.utils import jitter_transform, generate_sampling_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simulate jittery motion on an image producing corrupted kspace"
    )
    parser.add_argument('img', help="File for image of brain to be corrupted")
    parser.add_argument('mps', help="File for coil sensitivity maps")
    parser.add_argument('dst', help="File to save kspace output")
    args = parser.parse_args()

    img = np.load(args.img)
    mps = np.load(args.mps)

    kgrid, _, _, rkgrid = generate_transform_grids(img.shape)
    transforms = jitter_transform(64)
    transforms = sp.to_device(transforms)
    mask = generate_sampling_mask(64, img.shape)
    E = Encoding(transforms, mask, mps, kgrid, rkgrid, shot_chunk_size=1)
    ksp = E * img
    np.save(args.dst, ksp)
