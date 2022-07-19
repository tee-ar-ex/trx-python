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

from trx.workflows import validate_tractogram


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

    validate_tractogram(args.in_tractogram, reference=args.reference,
                        out_tractogram=args.out_tractogram,
                        remove_identical_streamlines=args.remove_identical_streamlines)


if __name__ == "__main__":
    main()
