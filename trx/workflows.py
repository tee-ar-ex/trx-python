#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import logging
import os
import shutil

import nibabel as nib
import numpy as np
try:
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram, load_tractogram
    from dipy.tracking.streamline import set_number_of_points
    from dipy.tracking.utils import density_map
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.trx_file_memmap import load, save, TrxFile
from trx.viz import display
from trx.utils import (flip_sft, is_header_compatible,
                                            get_axis_shift_vector,
                                            load_tractogram_with_reference,
                                            split_name_with_gz)


def convert_dsi_studio(in_dsi_tractogram, in_dsi_fa, out_tractogram,
                       remove_invalid=True, keep_invalid=False):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return None
    in_ext = split_name_with_gz(in_dsi_tractogram)[1]
    out_ext = split_name_with_gz(out_tractogram)[1]

    if in_ext == '.trk.gz':
        with gzip.open(in_dsi_tractogram, 'rb') as f_in:
            with open('tmp.trk', 'wb') as f_out:
                f_out.writelines(f_in)
                sft = load_tractogram('tmp.trk', 'same',
                                      bbox_valid_check=False)
                os.remove('tmp.trk')
    elif in_ext == '.trk':
        sft = load_tractogram(in_dsi_tractogram, 'same',
                              bbox_valid_check=False)
    else:
        raise IOError('{} is not currently supported.'.format(in_ext))

    sft.to_vox()
    sft_fix = StatefulTractogram(sft.streamlines, in_dsi_fa, Space.VOXMM,
                                 data_per_point=sft.data_per_point,
                                 data_per_streamline=sft.data_per_streamline)
    sft_fix.to_vox()
    flip_axis = ['x', 'y']
    sft_fix.streamlines._data -= get_axis_shift_vector(flip_axis)
    sft_flip = flip_sft(sft_fix, flip_axis)

    sft_flip.to_rasmm()
    sft_flip.streamlines._data -= [0.5, 0.5, -0.5]

    if remove_invalid:
        sft_flip.remove_invalid_streamlines()

    if out_ext != '.trx':
        save_tractogram(sft_flip, out_tractogram,
                        bbox_valid_check=not keep_invalid)
    else:
        trx = TrxFile.from_sft(sft_flip)
        save(trx, out_tractogram)


def convert_tractogram(in_tractogram, out_tractogram, reference,
                       pos_dtype='float32', offsets_dtype='uint32'):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return None
    in_ext = split_name_with_gz(in_tractogram)[1]
    out_ext = split_name_with_gz(out_tractogram)[1]

    if in_ext == out_ext:
        raise IOError('Input and output cannot be of the same file format')

    if in_ext != '.trx':
        sft = load_tractogram_with_reference(in_tractogram, reference,
                                             bbox_check=False)
    else:
        trx = load(in_tractogram)
        sft = trx.to_sft()

    if out_ext != '.trx':
        if out_ext == '.vtk':
            if sft.streamlines._data.dtype.name != pos_dtype:
                sft.streamlines._data = sft.streamlines._data.astype(pos_dtype)
            if offsets_dtype == 'uint64' or offsets_dtype == 'uint32':
                offsets_dtype = offsets_dtype[1:]
            if sft.streamlines._offsets.dtype.name != offsets_dtype:
                sft.streamlines._offsets = sft.streamlines._offsets.astype(offsets_dtype)
        save_tractogram(sft, out_tractogram, bbox_valid_check=False)
    else:
        trx = TrxFile.from_sft(sft)
        if trx.streamlines._data.dtype.name != pos_dtype:
            trx.streamlines._data = trx.streamlines._data.astype(pos_dtype)
        if trx.streamlines._offsets.dtype.name != offsets_dtype:
            trx.streamlines._offsets = trx.streamlines._offsets.astype(offsets_dtype)
        save(trx, out_tractogram)


def tractogram_simple_compare(in_tractograms, reference):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return
    in_ext = os.path.splitext(in_tractograms[0])[1]
    if in_ext != '.trx':
        sft_1 = load_tractogram_with_reference(in_tractograms[0], reference,
                                               bbox_check=False)
    else:
        trx = load(in_tractograms[0])
        sft_1 = trx.to_sft()

    in_ext = os.path.splitext(in_tractograms[1])[1]
    if in_ext != '.trx':
        sft_2 = load_tractogram_with_reference(in_tractograms[1], reference,
                                               bbox_check=False)
    else:
        trx = load(in_tractograms[1])
        sft_2 = trx.to_sft()

    if np.allclose(sft_1.streamlines._data, sft_2.streamlines._data,
                   atol=0.001):
        print('Matching tractograms in rasmm!')
    else:
        print('Average difference in rasmm of {}'.format(np.average(
            sft_1.streamlines._data - sft_2.streamlines._data, axis=0)))

    sft_1.to_voxmm()
    sft_2.to_voxmm()
    if np.allclose(sft_1.streamlines._data, sft_2.streamlines._data,
                   atol=0.001):
        print('Matching tractograms in voxmm!')
    else:
        print('Average difference in voxmm of {}'.format(np.average(
            sft_1.streamlines._data - sft_2.streamlines._data, axis=0)))

    sft_1.to_vox()
    sft_2.to_vox()
    if np.allclose(sft_1.streamlines._data, sft_2.streamlines._data,
                   atol=0.001):
        print('Matching tractograms in vox!')
    else:
        print('Average difference in vox of {}'.format(np.average(
            sft_1.streamlines._data - sft_2.streamlines._data, axis=0)))


def verify_header_compatibility(in_files):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return
    all_valid = True
    for filepath in in_files:
        if not os.path.isfile(filepath):
            print('{} does not exist'.format(filepath))
        _, in_extension = split_name_with_gz(filepath)
        if in_extension not in ['.trk', '.nii', '.nii.gz', '.trx']:
            raise IOError('{} does not have a supported extension'.format(
                filepath))
        if not is_header_compatible(in_files[0], filepath):
            print('{} and {} do not have compatible header.'.format(
                in_files[0], filepath))
            all_valid = False
    if all_valid:
        print('All input files have compatible headers.')


def tractogram_visualize_overlap(in_tractogram, reference, remove_invalid=True):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return None
    in_ext = os.path.splitext(in_tractogram)[1]

    if in_ext != '.trx':
        sft = load_tractogram_with_reference(in_tractogram, reference,
                                             bbox_check=False)
    else:
        trx = load(in_tractogram)
        sft = trx.to_sft()
        sft.streamlines._data = sft.streamlines._data.astype(float)

    sft.data_per_point = None
    sft.streamlines = set_number_of_points(sft.streamlines, 200)

    if remove_invalid:
        sft.remove_invalid_streamlines()

    # Approach (1)
    density_1 = density_map(sft.streamlines, sft.affine, sft.dimensions)
    img = nib.load(reference)
    display(img.get_fdata(), volume_affine=img.affine,
            streamlines=sft.streamlines,  title='RASMM')

    # Approach (2)
    sft.to_vox()
    density_2 = density_map(sft.streamlines, np.eye(4), sft.dimensions)

    # Small difference due to casting of the affine as float32 or float64
    diff = density_1 - density_2
    print('Total difference of {} voxels with total value of {}'.format(
        np.count_nonzero(diff), np.sum(np.abs(diff))))

    display(img.get_fdata(), streamlines=sft.streamlines, title='VOX')

    # Try VOXMM
    sft.to_voxmm()
    affine = np.eye(4)
    affine[0:3, 0:3] *= sft.voxel_sizes

    display(img.get_fdata(), volume_affine=affine,
            streamlines=sft.streamlines,  title='VOXMM')
