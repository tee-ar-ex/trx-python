#!/usr/bin/env python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversion of '.tck', '.trk', '.fib', '.vtk' and 'dpy' files using updated file
format standard. TRK file always needs a reference file, a NIFTI, for
conversion. The FIB file format is in fact a VTK, MITK Diffusion supports it.
"""

import argparse
import os

from dipy.io.streamline import save_tractogram

from trx_file_memmap import TrxFile
from trx_file_memmap import load, save
from tractography_file_format.utils import load_tractogram_with_reference

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('out_name', metavar='OUTPUT_NAME',
                   help='Output filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('--reference',
                        help='Reference anatomy for tck/vtk/fib/dpy file\n'
                             'support (.nii or .nii.gz).')
    p.add_argument(
        '-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_name) and not args.overwrite:
        raise IOError('{} already exists, use -f to overwrite.'.format(args.out_name))

    in_ext = os.path.splitext(args.in_tractogram)[1]
    out_ext = os.path.splitext(args.out_name)[1]

    if in_ext == out_ext:
        parser.error('Input and output cannot be of the same file format')

    if in_ext != '.trx':
        sft = load_tractogram_with_reference(parser, args, args.in_tractogram,
                                         bbox_check=False)
    else:
        trx = load(args.in_tractogram)
        sft = trx.to_sft()

    if out_ext != '.trx':
        save_tractogram(sft, args.out_name, bbox_valid_check=False)
    else:
        trx = TrxFile.from_sft(sft)
        save(trx, args.out_name)



if __name__ == "__main__":
    main()
