#!/usr/bin/env python

"""
Display a tractogram and its density map (computed from Dipy) in rasmm,
voxmm and vox space with its bounding box.
"""

import argparse
import itertools
import os

from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import density_map
from dipy.viz import window, actor, colormap
from fury.utils import get_bounds
import fury.utils as ut_vtk
import nibabel as nib
import numpy as np
import vtk

from trx_file_memmap import load
from tractography_file_format.utils import load_tractogram_with_reference


def display(volume, volume_affine=None, streamlines=None, title='FURY',
            display_bounds=True):

    volume = volume.astype(float)
    scene = window.Scene()
    scene.background((1., 0.5, 0.))

    # Show the X/Y/Z plane intersecting, mid-slices
    slicer_actor_1 = actor.slicer(volume, affine=volume_affine,
                                  value_range=(volume.min(), volume.max()),
                                  interpolation='nearest', opacity=0.8)
    slicer_actor_2 = actor.slicer(volume, affine=volume_affine,
                                  value_range=(volume.min(), volume.max()),
                                  interpolation='nearest', opacity=0.8)
    slicer_actor_3 = actor.slicer(volume, affine=volume_affine,
                                  value_range=(volume.min(), volume.max()),
                                  interpolation='nearest', opacity=0.8)
    slicer_actor_1.display(y=volume.shape[1] // 2)
    slicer_actor_2.display(x=volume.shape[0] // 2)
    slicer_actor_3.display(z=volume.shape[2] // 2)

    scene.add(slicer_actor_1)
    scene.add(slicer_actor_2)
    scene.add(slicer_actor_3)

    # Bounding box to facilitate error detections
    if display_bounds:
        src = vtk.vtkCubeSource()
        bounds = np.round(get_bounds(slicer_actor_1), 6)
        src.SetBounds(bounds)
        src.Update()
        cube_actor = ut_vtk.get_actor_from_polydata(src.GetOutput())
        cube_actor.GetProperty().SetRepresentationToWireframe()
        scene.add(cube_actor)

        # Show each corner's coordinates
        corners = itertools.product(bounds[0:2], bounds[2:4], bounds[4:6])
        for corner in corners:
            text_actor = actor.text_3d('{}, {}, {}'.format(
                *corner), corner, font_size=6, justification='center')
            scene.add(text_actor)

        # Show the X/Y/Z dimensions
        text_actor_x = actor.text_3d('{}'.format(np.abs(bounds[0]-bounds[1])),
                                     ((bounds[0]+bounds[1])/2,
                                      bounds[2],
                                      bounds[4]),
                                     font_size=10, justification='center')
        text_actor_y = actor.text_3d('{}'.format(np.abs(bounds[2]-bounds[3])),
                                     (bounds[0],
                                      (bounds[2]+bounds[3])/2,
                                      bounds[4]),
                                     font_size=10, justification='center')
        text_actor_z = actor.text_3d('{}'.format(np.abs(bounds[4]-bounds[5])),
                                     (bounds[0],
                                      bounds[2],
                                      (bounds[4]+bounds[5])/2),
                                     font_size=10, justification='center')
        scene.add(text_actor_x)
        scene.add(text_actor_y)
        scene.add(text_actor_z)

    if streamlines is not None:
        streamlines_actor = actor.line(streamlines,
                                       colormap.line_colors(streamlines),
                                       opacity=0.25)
        scene.add(streamlines_actor)
    window.show(scene, title=title, size=(800, 800))


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_tractogram', metavar='IN_TRACTOGRAM',
                   help='Tractogram filename. Format must be one of \n'
                        'trk, tck, vtk, fib, dpy, trx')
    p.add_argument('--reference',
                   help='Reference anatomy for tck/vtk/fib/dpy file\n'
                        'support (nii or nii.gz).')
    p.add_argument('--remove_invalid', action='store_true',
                   help='Removes invalid streamlines to avoid the density_map'
                        'function to crash.')

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
        sft.streamlines._data = sft.streamlines._data.astype(float)

    sft.data_per_point = None
    sft.streamlines = set_number_of_points(sft.streamlines, 200)

    if args.remove_invalid:
        sft.remove_invalid_streamlines()

    # Approach (1)
    density_1 = density_map(sft.streamlines, sft.affine, sft.dimensions)
    if args.reference is not None:
        img = nib.load(args.reference)
        display(img.get_fdata(), volume_affine=img.affine,
                streamlines=sft.streamlines,  title='RASMM')
    else:
        display(density_1, volume_affine=sft.affine,
                streamlines=sft.streamlines,  title='RASMM')

    # Approach (2)
    sft.to_vox()
    density_2 = density_map(sft.streamlines, np.eye(4), sft.dimensions)

    # Small difference due to casting of the affine as float32 or float64
    diff = density_1 - density_2
    print('Total difference of {} voxels with total value of {}'.format(
        np.count_nonzero(diff), np.sum(np.abs(diff))))

    if args.reference is not None:
        display(img.get_fdata(), streamlines=sft.streamlines, title='VOX')
    else:
        display(density_2, streamlines=sft.streamlines, title='VOX')

    # Try VOXMM
    sft.to_voxmm()
    affine = np.eye(4)
    affine[0:3, 0:3] *= sft.voxel_sizes

    if args.reference is not None:
        display(img.get_fdata(), volume_affine=affine,
                streamlines=sft.streamlines, title='VOX')
    else:
        display(density_1, volume_affine=affine,
                streamlines=sft.streamlines,  title='VOXMM')


if __name__ == "__main__":
    main()
