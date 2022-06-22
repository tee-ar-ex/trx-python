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
import numpy as np

from trx.io import load_wrapper, save_wrapper
from trx.trx_file_memmap import TrxFile, concatenate


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractograms', nargs='+',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')
    p.add_argument('out_tractogram',
                   help='Save tractogram after removing streamlines with '
                        'invalid coordinates\nor streamlines with single or no'
                        ' point.')

    p.add_argument('--delete_dpv', action='store_true',
                   help='Delete the dpv if it exists. '
                        'Required if not all input has the same metadata.')
    p.add_argument('--delete_dps', action='store_true',
                   help='Delete the dps if it exists. '
                        'Required if not all input has the same metadata.')
    p.add_argument('--delete_groups', action='store_true',
                   help='Delete the groups if it exists. '
                        'Required if not all input has the same metadata.')
    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (.nii or .nii.gz).')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    trx_list = []
    has_group = False
    for filename in args.in_tractograms:
        tractogram_obj = load_wrapper(filename, args.reference)

        if not isinstance(tractogram_obj, TrxFile):
            tractogram_obj = TrxFile.from_sft(tractogram_obj)
        elif len(tractogram_obj.groups):
            has_group = True
        trx_list.append(tractogram_obj)

    trx = concatenate(trx_list, delete_dpv=args.delete_dpv,
                      delete_dps=args.delete_dps,
                      delete_groups=args.delete_groups,
                      check_space_attributes=True,
                      preallocation=not has_group)
    save_wrapper(trx, args.out_tractogram)


if __name__ == "__main__":
    main()
