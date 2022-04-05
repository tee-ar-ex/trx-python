#!/usr/bin/env python

""" Simple comparison of tractogram by subtracting the coordinates' data.
Does not account for shuffling of streamlines. Simple A-B operations.

Differences below 1e^3 are expected for affine with large rotation/scaling.
Difference below 1e^6 are expected for isotropic data with small rotation.
"""

import argparse
import os

import numpy as np

from trx_file_memmap import load
from tractography_file_format.utils import load_tractogram_with_reference


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs=2,
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx')
    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                   'support (.nii or .nii.gz).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_ext = os.path.splitext(args.in_tractograms[0])[1]
    if in_ext != '.trx':
        sft_1 = load_tractogram_with_reference(parser, args,
                                               args.in_tractograms[0],
                                               bbox_check=False)
    else:
        trx = load(args.in_tractograms[0])
        sft_1 = trx.to_sft()

    in_ext = os.path.splitext(args.in_tractograms[1])[1]
    if in_ext != '.trx':
        sft_2 = load_tractogram_with_reference(parser, args,
                                               args.in_tractograms[1],
                                               bbox_check=False)
    else:
        trx = load(args.in_tractograms[1])
        sft_2 = trx.to_sft()

    if np.allclose(sft_1.streamlines._data, sft_2.streamlines._data,
                   atol=0.001):
        print('Matching tractograms in rasmm!')
    else:
        print('Average difference in rasmm of {}'.format(np.average(
            sft_1.streamlines._data - sft_2.streamlines._data, axis=0)))

    sft_1.to_voxmm()
    sft_2.to_voxmm()
    if np.allclose(sft_1.streamlines._data, sft_2.streamlines._data,
                   atol=0.001):
        print('Matching tractograms in voxmm!')
    else:
        print('Average difference in voxmm of {}'.format(np.average(
            sft_1.streamlines._data - sft_2.streamlines._data, axis=0)))

    sft_1.to_vox()
    sft_2.to_vox()
    if np.allclose(sft_1.streamlines._data, sft_2.streamlines._data,
                   atol=0.001):
        print('Matching tractograms in vox!')
    else:
        print('Average difference in vox of {}'.format(np.average(
            sft_1.streamlines._data - sft_2.streamlines._data, axis=0)))


if __name__ == "__main__":
    main()
