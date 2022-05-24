#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import os

from trx.workflows import generate_trx_from_scratch


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                   'support (.nii or .nii.gz).')
    p.add_argument('out_tractogram', metavar='OUT_TRACTOGRAM',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx.')

    p1 = p.add_argument_group(title='Positions options')
    p1.add_argument('--positions', metavar='POSITIONS',
                    help='.')
    p1.add_argument('--offsets', metavar='OFFSETS',
                    help='.')
    p1.add_argument('--positions_csv', metavar='POSITIONS',
                    help='x1,y1, z1,.')
    p1.add_argument('--space', choices=['RASMM', 'VOXMM', 'VOX'],
                    default='RASMM',
                    help='')
    p1.add_argument('--origin', choices=['NIFTI', 'TRACKVIS'],
                    default='NIFTI',
                    help='')
    p2 = p.add_argument_group(title='Data type options')
    p2.add_argument('--positions_dtype', default='float32',
                    choices=['float16', 'float32', 'float64'],
                    help='Specify the datatype for positions for trx. [%(default)s]')
    p2.add_argument('--offsets_dtype', default='uint64',
                    choices=['uint32', 'uint64'],
                    help='Specify the datatype for offsets for trx. [%(default)s]')

    p3 = p.add_argument_group(title='Streamlines metadata options')
    p3.add_argument('--dpv', metavar=('FILE', 'DTYPE'), nargs=2,
                    action='append',
                    help='.')
    p3.add_argument('--dps', metavar=('FILE', 'DTYPE'), nargs=2,
                    action='append',
                    help='.')
    p3.add_argument('--groups', metavar=('FILE', 'DTYPE'), nargs=2,
                    action='append',
                    help='.')
    p3.add_argument('--dpg', metavar=('GROUP', 'FILE', 'DTYPE'), nargs=3,
                    action='append',
                    help='.')

    p.add_argument('--verify_invalid', action='store_true',
                   help='.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(
            args.out_tractogram))

    if args.positions_csv and args.positions:
        raise IOError('Cannot use both positions options.')
    if args.positions and args.offsets is None:
        raise IOError('--offsets must be provided if --positions is used.')
    if args.offsets and args.positions is None:
        raise parser.error(
            '--positions must be provided if --offsets is used.')

    generate_trx_from_scratch(args.reference, args.out_tractogram,
                              positions_csv=args.positions_csv,
                              positions=args.positions, offsets=args.offsets,
                              positions_dtype=args.positions_dtype,
                              offsets_dtype=args.offsets_dtype,
                              space_str=args.space, origin_str=args.origin,
                              verify_invalid=args.verify_invalid,
                              dpv=args.dpv, dps=args.dps,
                              groups=args.groups, dpg=args.dpg)


if __name__ == "__main__":
    main()
