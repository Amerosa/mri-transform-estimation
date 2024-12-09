import argparse
import numpy as np
from estimation.utils import generate_transforms

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Simulate motion from parameters, creates m rows of motion shots interpolating between -n/2, n/2 for each param'
    )
    parser.add_argument("out", help="Output file")
    parser.add_argument('-s', '--shots', type=int, default=2, help='Number of motions states or binning of read out lines')
    parser.add_argument('-t', '--translations', nargs='+', type=int, help="Translation parameters, defaults to zero for each axis") 
    parser.add_argument('-r', '--rotations', nargs='+', type=int, help="Rotation parameters, defaults to zero for each axis") 
    args = parser.parse_args()

    #TODO Proper bounds checking at some bound for general usage
    if args.translations is None:
        translations = [0,0,0]
    else:
        translations = args.translations

    if args.rotations is None:
        rotations = [0,0,0]
    else:
        rotations = args.rotations

    parameters = generate_transforms(args.shots, translations, rotations)
    np.save(args.out, parameters) 