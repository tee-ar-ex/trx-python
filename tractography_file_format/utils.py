#!/usr/bin/env python

import os
import logging

import dipy
from dipy.io.streamline import load_tractogram
from dipy.io.utils import is_reference_info_valid
import nibabel as nib
import numpy as np

import trx_file_memmap


def split_name_with_nii(filename):
    """
    Returns the clean basename and extension of a file.
    Means that this correctly manages the ".nii.gz" extensions.

    Parameters
    ----------
    filename: str
        The filename to clean

    Returns
    -------
        base, ext : tuple(str, str)
        Clean basename and the full extension
    """
    base, ext = os.path.splitext(filename)

    if ext == ".gz":
        # Test if we have a .nii additional extension
        temp_base, add_ext = os.path.splitext(base)

        if add_ext == ".nii":
            ext = add_ext + ext
            base = temp_base

    return base, ext


def get_reference_info_wrapper(reference):
    """ Will compare the spatial attribute of 2 references
    Parameters
    ----------
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), TrxFile or trx.header (dict)
        Reference that provides the spatial attribute.
    Returns
    -------
    output : tuple
        - affine ndarray (4,4), np.float32, tranformation of VOX to RASMM
        - dimensions ndarray (3,), int16, volume shape for each axis
        - voxel_sizes  ndarray (3,), float32, size of voxel for each axis
        - voxel_order, string, Typically 'RAS' or 'LPS'
    """
    is_nifti = False
    is_trk = False
    is_sft = False
    is_trx = False
    if isinstance(reference, str):
        _, ext = split_name_with_nii(reference)
        if ext in ['.nii', '.nii.gz']:
            header = nib.load(reference).header
            is_nifti = True
        elif ext == '.trk':
            header = nib.streamlines.load(reference, lazy_load=True).header
            is_trk = True
        elif ext == '.trx':
            header = trx_file_memmap.load(reference).header
            is_trx = True
    elif isinstance(reference, trx_file_memmap.trx_file_memmap.TrxFile):
        header = reference.header
        is_trx = True
    elif isinstance(reference, nib.nifti1.Nifti1Image):
        header = reference.header
        is_nifti = True
    elif isinstance(reference, nib.streamlines.trk.TrkFile):
        header = reference.header
        is_trk = True
    elif isinstance(reference, nib.nifti1.Nifti1Header):
        header = reference
        is_nifti = True
    elif isinstance(reference, dict) and 'magic_number' in reference:
        header = reference
        is_trk = True
    elif isinstance(reference, dict) and 'NB_VERTICES' in reference:
        header = reference
        is_trx = True
    elif isinstance(reference, dipy.io.stateful_tractogram.StatefulTractogram):
        is_sft = True

    if is_nifti:
        affine = header.get_best_affine()
        dimensions = header['dim'][1:4]
        voxel_sizes = header['pixdim'][1:4]

        if not affine[0:3, 0:3].any():
            raise ValueError(
                'Invalid affine, contains only zeros.'
                'Cannot determine voxel order from transformation')
        voxel_order = ''.join(nib.aff2axcodes(affine))
    elif is_trk:
        affine = header['voxel_to_rasmm']
        dimensions = header['dimensions']
        voxel_sizes = header['voxel_sizes']
        voxel_order = header['voxel_order']
    elif is_sft:
        affine, dimensions, voxel_sizes, voxel_order =\
            reference.space_attributes
    elif is_trx:
        affine = header['VOXEL_TO_RASMM']
        dimensions = header['DIMENSIONS']
        voxel_sizes = nib.affines.voxel_sizes(affine)
        voxel_order = ''.join(nib.aff2axcodes(affine))
    else:
        raise TypeError('Input reference is not one of the supported format')

    if isinstance(voxel_order, np.bytes_):
        voxel_order = voxel_order.decode('utf-8')

    # Run this function to logging the warning from it
    is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order)

    return affine, dimensions, voxel_sizes, voxel_order


def is_header_compatible(reference_1, reference_2):
    """ Will compare the spatial attribute of 2 references
    Parameters
    ----------
    reference_1 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.
    reference_2 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.
    Returns
    -------
    output : bool
        Does all the spatial attribute match
    """

    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = \
        get_reference_info_wrapper(reference_1)
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = \
        get_reference_info_wrapper(reference_2)

    identical_header = True
    if not np.allclose(affine_1, affine_2, rtol=1e-03, atol=1e-03):
        logging.error('Affine not equal')
        identical_header = False

    if not np.array_equal(dimensions_1, dimensions_2):
        logging.error('Dimensions not equal')
        identical_header = False

    if not np.allclose(voxel_sizes_1, voxel_sizes_2, rtol=1e-03, atol=1e-03):
        logging.error('Voxel_size not equal')
        identical_header = False

    if voxel_order_1 != voxel_order_2:
        logging.error('Voxel_order not equal')
        identical_header = False

    return identical_header


def is_argument_set(args, arg_name):
    # Check that attribute is not None
    return not getattr(args, arg_name, None) is None


def load_tractogram_with_reference(parser, args, filepath,
                                   bbox_check=True):
    # Force the usage of --reference for all file formats without an header
    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        if is_argument_set(args, 'reference'):
            logging.warning('Reference is discarded for this file format '
                            '{}.'.format(filepath))
        sft = load_tractogram(filepath, 'same',
                              bbox_valid_check=bbox_check)
    elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
        if (not is_argument_set(args, 'reference')) \
                or (args.reference is None) or (args.reference == 'same'):
            parser.error('--reference is required for this file format '
                         '{}.'.format(filepath))
        else:
            sft = load_tractogram(filepath, args.reference,
                                  bbox_valid_check=bbox_check)

    else:
        parser.error('{} is an unsupported file format'.format(filepath))

    return sft
