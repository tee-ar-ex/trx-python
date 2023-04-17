#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from nibabel.streamlines.tractogram import TractogramItem
from nibabel.streamlines.tractogram import Tractogram
from nibabel.streamlines.array_sequence import ArraySequence
import os
import logging

import nibabel as nib
import numpy as np

try:
    import dipy
    dipy_available = True
except ImportError:
    dipy_available = False


def split_name_with_gz(filename):
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

        if add_ext == ".nii" or add_ext == ".trk":
            ext = add_ext + ext
            base = temp_base

    return base, ext


def get_reference_info_wrapper(reference):
    """ Will compare the spatial attribute of 2 references.

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
    from trx import trx_file_memmap
    is_nifti = False
    is_trk = False
    is_sft = False
    is_trx = False
    if isinstance(reference, str):
        _, ext = split_name_with_gz(reference)
        if ext in ['.nii', '.nii.gz']:
            header = nib.load(reference).header
            is_nifti = True
        elif ext == '.trk':
            header = nib.streamlines.load(reference, lazy_load=True).header
            is_trk = True
        elif ext == '.trx':
            header = trx_file_memmap.load(reference).header
            is_trx = True
    elif isinstance(reference, trx_file_memmap.TrxFile):
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
    elif dipy_available and \
            isinstance(reference, dipy.io.stateful_tractogram.StatefulTractogram):
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

    if dipy_available:
        from dipy.io.utils import is_reference_info_valid
        is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order)

    return affine, dimensions, voxel_sizes, voxel_order


def is_header_compatible(reference_1, reference_2):
    """ Will compare the spatial attribute of 2 references.

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


def get_axis_shift_vector(flip_axes):
    """
    Parameters
    ----------
    flip_axes : list of str
        String containing the axis to flip.
        Possible values are 'x', 'y', 'z'
    Returns
    -------
    flip_vector : np.ndarray (3,)
        Vector containing the axis to flip.
        Possible values are -1, 1
    """
    shift_vector = np.zeros(3)
    if 'x' in flip_axes:
        shift_vector[0] = -1.0
    if 'y' in flip_axes:
        shift_vector[1] = -1.0
    if 'z' in flip_axes:
        shift_vector[2] = -1.0

    return shift_vector


def get_axis_flip_vector(flip_axes):
    """
    Parameters
    ----------
    flip_axes : list of str
        String containing the axis to flip.
        Possible values are 'x', 'y', 'z'
    Returns
    -------
    flip_vector : np.ndarray (3,)
        Vector containing the axis to flip.
        Possible values are -1, 1
    """
    flip_vector = np.ones(3)
    if 'x' in flip_axes:
        flip_vector[0] = -1.0
    if 'y' in flip_axes:
        flip_vector[1] = -1.0
    if 'z' in flip_axes:
        flip_vector[2] = -1.0

    return flip_vector


def get_shift_vector(sft):
    """
    When flipping a tractogram the shift vector is used to change the origin
    of the grid from the corner to the center of the grid.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram object
    Returns
    -------
    shift_vector : ndarray
        Shift vector to apply to the streamlines
    """
    dims = sft.space_attributes[1]
    shift_vector = -1.0 * (np.array(dims) / 2.0)

    return shift_vector


def flip_sft(sft, flip_axes):
    """ Flip the streamlines in the StatefulTractogram according to the
    flip_axes. Uses the spatial information to flip according to the center
    of the grid.

    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram to flip
    flip_axes : list of str
        Axes to flip.
        Possible values are 'x', 'y', 'z'
    Returns
    -------
    sft : StatefulTractogram
        StatefulTractogram with flipped axes
    """
    if not dipy_available:
        logging.error('Dipy library is missing, cannot use functions related '
                      'to the StatefulTractogram.')
        return None

    flip_vector = get_axis_flip_vector(flip_axes)
    shift_vector = get_shift_vector(sft)

    flipped_streamlines = []
    for streamline in sft.streamlines:
        mod_streamline = streamline + shift_vector
        mod_streamline *= flip_vector
        mod_streamline -= shift_vector
        flipped_streamlines.append(mod_streamline)

    from dipy.io.stateful_tractogram import StatefulTractogram
    new_sft = StatefulTractogram.from_sft(flipped_streamlines, sft,
                                          data_per_point=sft.data_per_point,
                                          data_per_streamline=sft.data_per_streamline)
    return new_sft


def load_matrix_in_any_format(filepath):
    """ Load a matrix from a txt file OR a npy file.

    Parameters
    ----------
    filepath : str
        Path to the matrix file.
    Returns
    -------
    matrix : numpy.ndarray
        The matrix.
    """
    _, ext = os.path.splitext(filepath)
    if ext == '.txt':
        data = np.loadtxt(filepath)
    elif ext == '.npy':
        data = np.load(filepath)
    else:
        raise ValueError('Extension {} is not supported'.format(ext))

    return data


def get_reverse_enum(space_str, origin_str):
    """ Convert string representation to enums for the StatefulTractogram.

    Parameters
    ----------
    space_str : str
        String representing the space.
    origin_str : str
        String representing the origin.
    Returns
    -------
    output : str
        Space and Origin as Enums.
    """
    if not dipy_available:
        logging.error('Dipy library is missing, cannot use functions related '
                      'to the StatefulTractogram.')
        return None
    from dipy.io.stateful_tractogram import Space, Origin
    origin = Origin.NIFTI if origin_str.lower() == 'nifti' else Origin.TRACKVIS
    if space_str.lower() == 'rasmm':
        space = Space.RASMM
    elif space_str.lower() == 'voxmm':
        space = Space.VOXMM
    else:
        space = Space.VOX

    return space, origin


def convert_data_dict_to_tractogram(data):
    """ Convert a data from a lazy tractogram to a tractogram

    Keyword arguments:
        data -- The data dictionary to convert into a nibabel tractogram

    Returns:
        A Tractogram object
    """
    streamlines = ArraySequence(data['strs'])
    streamlines._data = streamlines._data

    for key in data['dps']:
        shape = (len(streamlines), len(data['dps'][key]) // len(streamlines))
        data['dps'][key] = np.array(data['dps'][key]).reshape(shape)

    for key in data['dpv']:
        shape = (len(streamlines._data), len(
            data['dpv'][key]) // len(streamlines._data))
        data['dpv'][key] = np.array(data['dpv'][key]).reshape(shape)

        tmp_arr = ArraySequence()
        tmp_arr._data = data['dpv'][key]
        tmp_arr._offsets = streamlines._offsets
        tmp_arr._lengths = streamlines._lengths
        data['dpv'][key] = tmp_arr

    obj = Tractogram(streamlines, data_per_point=data['dpv'],
                     data_per_streamline=data['dps'])

    return obj


def append_generator_to_dict(gen, data):
    if isinstance(gen, TractogramItem):
        data['strs'].append(gen.streamline.tolist())
        for key in gen.data_for_points:
            if key not in data['dpv']:
                data['dpv'][key] = np.array([])
            data['dpv'][key] = np.append(
                data['dpv'][key], gen.data_for_points[key])
        for key in gen.data_for_streamline:
            if key not in data['dps']:
                data['dps'][key] = np.array([])
            data['dps'][key] = np.append(
                data['dps'][key], gen.data_for_streamline[key])
    else:
        data['strs'].append(gen.tolist())


def verify_trx_dtype(trx, dict_dtype):
    """ Verify if the dtype of the data in the trx is the same as the one in
    the dict.

    Parameters
    ----------
    trx : Tractogram
        Tractogram to verify.
    dict_dtype : dict
        Dictionary containing the dtype to verify.
    Returns
    -------
    output : bool
        True if the dtype is the same, False otherwise.
    """
    identical = True
    for key in dict_dtype:
        if key == 'positions':
            if trx.streamlines._data.dtype != dict_dtype[key]:
                logging.warning('Positions dtype is different')
                identical = False
        elif key == 'offsets':
            if trx.streamlines._offsets.dtype != dict_dtype[key]:
                logging.warning('Offsets dtype is different')
                identical = False
        elif key == 'dpv':
            for key_dpv in dict_dtype[key]:
                if trx.data_per_vertex[key_dpv]._data.dtype != dict_dtype[key][key_dpv]:
                    logging.warning(
                        'Data per vertex ({}) dtype is different'.format(key_dpv))
                    identical = False
        elif key == 'dps':
            for key_dps in dict_dtype[key]:
                if trx.data_per_streamline[key_dps].dtype != dict_dtype[key][key_dps]:
                    logging.warning(
                        'Data per streamline ({}) dtype is different'.format(key_dps))
                    identical = False
        elif key == 'dpg':
            for key_group in dict_dtype[key]:
                for key_dpg in dict_dtype[key][key_group]:
                    if trx.data_per_point[key_group][key_dpg].dtype != dict_dtype[key][key_group][key_dpg]:
                        logging.warning(
                            'Data per group ({}) dtype is different'.format(key_dpg))
                        identical = False
        elif key == 'groups':
            for key_group in dict_dtype[key]:
                if trx.data_per_point[key_group]._data.dtype != dict_dtype[key][key_group]:
                    logging.warning(
                        'Data per group ({}) dtype is different'.format(key_group))
                    identical = False

    return identical
