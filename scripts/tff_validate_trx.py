#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Removal of streamlines that are out of the volume bounding box. In voxel space
no negative coordinate and no above volume dimension coordinate are possible.
Any streamline that do not respect these two conditions are removed.

The --cut_invalid option will cut streamlines so that their longest segment are
within the bounding box
"""

import argparse
import logging

import numpy as np

from trx.io import save_wrapper, load_wrapper
from trx.trx_file_memmap import TrxFile
from trx.streamlines_ops import perform_streamlines_operation, intersection

try:
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram, load_tractogram
    from dipy.tracking.streamline import set_number_of_points
    from dipy.tracking.utils import density_map
    dipy_available = True
except ImportError:
    dipy_available = False

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('--out_tractogram',
                   help='Save tractogram after removing streamlines with '
                        'invalid coordinates or streamlines with single or no'
                        ' point.')
    p.add_argument('--remove_identical_streamlines', action='store_true',
                   help='Remove identical streamlines from the set.')

    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (.nii or .nii.gz).')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    tractogram_obj = load_wrapper(args.in_tractogram, args.reference)
    if not isinstance(tractogram_obj, StatefulTractogram):
        sft = tractogram_obj.to_sft()
    else:
        sft = tractogram_obj

    ori_len = len(sft)
    _, invalid_coord_ind = sft.remove_invalid_streamlines()

    indices = [i for i in range(len(sft)) if len(sft.streamlines[i]) <= 1]

    for i in np.setdiff1d(range(len(sft)), indices):
        norm = np.linalg.norm(np.gradient(sft.streamlines[i],
                                            axis=0), axis=1)
        if (norm < 0.001).any():
            indices.append(i)

    indices_val = np.setdiff1d(range(len(sft)), indices).astype(np.uint32)
    logging.warning('Removed {} invalid streamlines.'.format(
        ori_len - len(indices_val)))

    if args.remove_identical_streamlines:
        ori_len = len(indices_val)
        _, indices_uniq = perform_streamlines_operation(intersection,
                                                   [sft.streamlines])
        logging.warning('Removed {} overlapping streamlines.'.format(
            ori_len - len(indices_uniq)))

        indices_final = np.intersect1d(indices_val, indices_uniq)
        print(len(indices_val), len(indices_uniq), ori_len)
    else:
        indices_final = indices_val

    if args.out_tractogram:
        save_wrapper(sft[indices_final], args.out_tractogram)


if __name__ == "__main__":
    main()
