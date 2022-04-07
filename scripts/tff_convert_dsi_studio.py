#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script is made to fix DSI-Studio TRK file (unknown space/convention) to
make it compatible with TrackVis, MI-Brain, Dipy Horizon (Stateful Tractogram).

The script either make it match with an anatomy from DSI-Studio.

This script was tested on various datasets and worked on all of them. However,
always verify the results and if a specific case does not work. Open an issue
on the Scilpy GitHub repository.

WARNING: This script is still experimental, DSI-Studio evolves quickly and
results may vary depending on the data itself as well as DSI-studio version.
"""

import argparse
import os
import gzip
import shutil

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram, load_tractogram

from tractography_file_format.utils import get_axis_shift_vector, flip_sft


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_dsi_tractogram',
                   help='Path of the input tractogram file from DSI studio '
                        '(.trk).')
    p.add_argument('in_dsi_fa',
                   help='Path of the input FA from DSI Studio (.nii.gz).')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    invalid = p.add_mutually_exclusive_group()
    invalid.add_argument('--remove_invalid', action='store_true',
                         help='Remove the streamlines landing out of the '
                              'bounding box.')
    invalid.add_argument('--keep_invalid', action='store_true',
                         help='Keep the streamlines landing out of the '
                              'bounding box.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError(
            '{} already exists, use -f to overwrite.'.format(args.out_tractogram))

    with gzip.open(args.in_dsi_tractogram, 'rb') as f_in:
        with open('tmp.trk', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            sft = load_tractogram('tmp.trk', 'same', bbox_valid_check=False)
            os.remove('tmp.trk')

    # LPS -> RAS convention in voxel space
    sft.to_vox()
    flip_axis = ['x', 'y']
    sft_fix = StatefulTractogram(sft.streamlines, args.in_dsi_fa, Space.VOXMM)
    sft_fix.to_vox()
    sft_fix.streamlines._data -= get_axis_shift_vector(flip_axis)
    sft_flip = flip_sft(sft_fix, flip_axis)

    sft_flip.to_rasmm()
    sft_flip.streamlines._data -= [0.5, 0.5, -0.5]

    if args.remove_invalid:
        sft_flip.remove_invalid_streamlines()
    save_tractogram(sft_flip, args.out_tractogram,
                    bbox_valid_check=not args.keep_invalid)


if __name__ == "__main__":
    main()
