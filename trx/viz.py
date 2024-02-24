# -*- coding: utf-8 -*-

import itertools
import logging

import numpy as np
try:
    from dipy.viz import window, actor, colormap
    from fury.utils import get_bounds
    import fury.utils as ut_vtk
    import vtk
    fury_available = True
except ImportError:
    fury_available = False


def display(volume, volume_affine=None, streamlines=None, title='FURY',
            display_bounds=True):
    if not fury_available:
        logging.error('Fury library is missing, visualization functions '
                      'are not available.')
        return None
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
