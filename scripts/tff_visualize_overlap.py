#!/usr/bin/env python

import argparse
import os

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram, set_sft_logger_level
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import (create_nifti_header, get_reference_info,
                           is_header_compatible)
from dipy.tracking.streamline import select_random_set_of_streamlines, set_number_of_points
from dipy.tracking.utils import density_map
import vtk

from dipy.io.streamline import save_tractogram

from trx_file_memmap import TrxFile
from trx_file_memmap import load, save
from tractography_file_format.utils import load_tractogram_with_reference, get_reference_info_wrapper
from dipy.viz import window, actor, colormap


def display(volume, volume_affine=None, streamlines=None, title='FURY'):
    volume = volume.astype(float)
    scene = window.Scene()
    scene.background((1.,0.5,0.))
    slicer_actor_1 = actor.slicer(volume, affine=volume_affine,
                                  value_range=(volume.min(), volume.max()),
                                  interpolation='nearest')
    slicer_actor_2 = actor.slicer(volume, affine=volume_affine,
                                  value_range=(volume.min(), volume.max()),
                                  interpolation='nearest')
    slicer_actor_3 = actor.slicer(volume, affine=volume_affine,
                                value_range=(volume.min(), volume.max()),
                                  interpolation='nearest')
    slicer_actor_1.display_extent(0, volume.shape[0],
                                volume.shape[1] // 2, volume.shape[1] // 2,
                                0, volume.shape[2])
    slicer_actor_2.display_extent(volume.shape[0] // 2, volume.shape[0] // 2,
                                0, volume.shape[1],
                                0, volume.shape[2])
    slicer_actor_3.display_extent(0, volume.shape[0],
                                  0, volume.shape[1],
                                  volume.shape[2] // 2, volume.shape[2] // 2)
    scene.add(slicer_actor_1)
    scene.add(slicer_actor_2)
    scene.add(slicer_actor_3)
    if streamlines is not None:
        streamlines_actor = actor.line(streamlines, colormap.line_colors(streamlines),
                                      opacity=0.25)
        scene.add(streamlines_actor)
    window.show(scene, title=title, size=(800, 800))


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy')

    p.add_argument('--reference',
                        help='Reference anatomy for tck/vtk/fib/dpy file\n'
                             'support (.nii or .nii.gz).')

    p.add_argument('--remove_invalid', action='store_true',
        help='.')

    p.add_argument('-f', dest='overwrite', action='store_true',
        help='Force overwriting of the output files.')

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_ext = os.path.splitext(args.in_tractogram)[1]

    if in_ext != '.trx':
        sft = load_tractogram_with_reference(parser, args, args.in_tractogram,
                                         bbox_check=False)
    else:
        trx = load(args.in_tractogram)
        sft = trx.to_sft()

    if args.remove_invalid:
        sft.remove_invalid_streamlines()

    sft.data_per_point = None
    streamlines = set_number_of_points(sft.streamlines, 100)

    # Approach (1)
    density_1 = density_map(sft.streamlines, sft.affine, sft.dimensions).astype(float)
    display(density_1, volume_affine=sft.affine, streamlines=sft.streamlines,  title='RASMM')

    # Approach (2)
    sft.to_vox()
    density_2 = density_map(sft.streamlines, np.eye(4), sft.dimensions).astype(float)

    # Normalization 0-1
    density_1 = density_1 / density_1.max()
    density_2 = density_2 / density_2.max()

    # Small difference due to the interpretation of the affine as float32 or float64
    print(np.allclose(density_1, density_2, atol=0.01))

    display(density_2, streamlines=sft.streamlines, title='VOX')

if __name__ == "__main__":
    main()
