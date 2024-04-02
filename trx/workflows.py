# -*- coding: utf-8 -*-

from copy import deepcopy
import csv
import gzip
import json
import logging
import os
import tempfile

import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
try:
    import dipy
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.io import get_trx_tmp_dir, load, load_sft_with_reference, save
from trx.streamlines_ops import perform_streamlines_operation, intersection
import trx.trx_file_memmap as tmm
from trx.viz import display
from trx.utils import (flip_sft, is_header_compatible,
                       get_axis_shift_vector,
                       get_reference_info_wrapper,
                       get_reverse_enum,
                       load_matrix_in_any_format,
                       split_name_with_gz)


def convert_dsi_studio(in_dsi_tractogram, in_dsi_fa, out_tractogram,
                       remove_invalid=True, keep_invalid=False):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return None
    from dipy.io.stateful_tractogram import StatefulTractogram, Space
    from dipy.io.streamline import save_tractogram, load_tractogram

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
        trx = tmm.TrxFile.from_sft(sft_flip)
        tmm.save(trx, out_tractogram)


def convert_tractogram(in_tractogram, out_tractogram, reference,
                       pos_dtype='float32', offsets_dtype='uint32'):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return None
    from dipy.io.streamline import save_tractogram

    in_ext = split_name_with_gz(in_tractogram)[1]
    out_ext = split_name_with_gz(out_tractogram)[1]

    if in_ext == out_ext:
        raise IOError('Input and output cannot be of the same file format.')

    if in_ext != '.trx':
        sft = load_sft_with_reference(in_tractogram, reference,
                                      bbox_check=False)
    else:
        trx = tmm.load(in_tractogram)
        sft = trx.to_sft()
        trx.close()

    if out_ext != '.trx':
        if out_ext == '.vtk':
            if sft.streamlines._data.dtype.name != pos_dtype:
                sft.streamlines._data = sft.streamlines._data.astype(pos_dtype)
            if offsets_dtype == 'uint64' or offsets_dtype == 'uint32':
                offsets_dtype = offsets_dtype[1:]
            if sft.streamlines._offsets.dtype.name != offsets_dtype:
                sft.streamlines._offsets = sft.streamlines._offsets.astype(
                    offsets_dtype)
        save_tractogram(sft, out_tractogram, bbox_valid_check=False)
    else:
        trx = tmm.TrxFile.from_sft(sft)
        if trx.streamlines._data.dtype.name != pos_dtype:
            trx.streamlines._data = trx.streamlines._data.astype(pos_dtype)
        if trx.streamlines._offsets.dtype.name != offsets_dtype:
            trx.streamlines._offsets = trx.streamlines._offsets.astype(
                offsets_dtype)
        tmm.save(trx, out_tractogram)
        trx.close()


def tractogram_simple_compare(in_tractograms, reference):
    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return
    from dipy.io.stateful_tractogram import StatefulTractogram

    tractogram_obj = load(in_tractograms[0], reference)
    if not isinstance(tractogram_obj, StatefulTractogram):
        sft_1 = tractogram_obj.to_sft()
        tractogram_obj.close()
    else:
        sft_1 = tractogram_obj

    tractogram_obj = load(in_tractograms[1], reference)
    if not isinstance(tractogram_obj, StatefulTractogram):
        sft_2 = tractogram_obj.to_sft()
        tractogram_obj.close()
    else:
        sft_2 = tractogram_obj

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
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.tracking.streamline import set_number_of_points
    from dipy.tracking.utils import density_map

    tractogram_obj = load(in_tractogram, reference)
    if not isinstance(tractogram_obj, StatefulTractogram):
        sft = tractogram_obj.to_sft()
        tractogram_obj.close()
    else:
        sft = tractogram_obj
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


def validate_tractogram(in_tractogram, reference, out_tractogram,
                        remove_identical_streamlines=True, precision=1):

    if not dipy_available:
        logging.error('Dipy library is missing, scripts are not available.')
        return None
    from dipy.io.stateful_tractogram import StatefulTractogram

    tractogram_obj = load(in_tractogram, reference)

    if not isinstance(tractogram_obj, StatefulTractogram):
        sft = tractogram_obj.to_sft()
        # tractogram_obj.close()
    else:
        sft = tractogram_obj

    ori_dtype = sft.dtype_dict
    ori_len = len(sft)
    tot_remove = 0

    invalid_coord_ind, _ = sft.remove_invalid_streamlines()
    tot_remove += len(invalid_coord_ind)
    logging.warning('Removed {} streamlines with invalid coordinates.'.format(
        len(invalid_coord_ind)))

    indices = [i for i in range(len(sft)) if len(sft.streamlines[i]) <= 1]
    tot_remove = + len(indices)
    logging.warning('Removed {} invalid streamlines (1 or 0 points).'.format(
        len(indices)))

    for i in np.setdiff1d(range(len(sft)), indices):
        norm = np.linalg.norm(np.diff(sft.streamlines[i],
                                      axis=0), axis=1)

        if (norm < 0.001).any():
            indices.append(i)

    indices_val = np.setdiff1d(range(len(sft)), indices).astype(np.uint32)
    logging.warning('Removed {} invalid streamlines (overlapping points).'.format(
        ori_len - len(indices_val)))
    tot_remove += ori_len - len(indices_val)

    if remove_identical_streamlines:
        _, indices_uniq = perform_streamlines_operation(intersection,
                                                        [sft.streamlines],
                                                        precision=precision)
        indices_final = np.intersect1d(
            indices_val, indices_uniq).astype(np.uint32)
        logging.warning('Removed {} overlapping streamlines.'.format(
            ori_len - len(indices_final) - tot_remove))

        indices_final = np.intersect1d(indices_val, indices_uniq)
    else:
        indices_final = indices_val

    if out_tractogram:
        streamlines = sft.streamlines[indices_final].copy()
        dpp = {}
        for key in sft.data_per_point.keys():
            dpp[key] = sft.data_per_point[key][indices_final].copy()

        dps = {}
        for key in sft.data_per_streamline.keys():
            dps[key] = sft.data_per_streamline[key][indices_final]
        new_sft = StatefulTractogram.from_sft(streamlines, sft,
                                              data_per_point=dpp,
                                              data_per_streamline=dps)
        new_sft.dtype_dict = ori_dtype
        save(new_sft, out_tractogram)


def generate_trx_from_scratch(reference, out_tractogram, positions_csv=False,
                              positions=False, offsets=False,
                              positions_dtype='float32', offsets_dtype='uint64',
                              space_str='rasmm', origin_str='nifti',
                              verify_invalid=True, dpv=[], dps=[],
                              groups=[], dpg=[]):
    with get_trx_tmp_dir() as tmp_dir_name:
        if positions_csv:
            with open(positions_csv, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)
                data = [np.reshape(i, (len(i) // 3, 3)).astype(float)
                        for i in data]
                streamlines = ArraySequence(data)
        else:
            positions = load_matrix_in_any_format(positions)
            offsets = load_matrix_in_any_format(offsets)
            lengths = tmm._compute_lengths(offsets)
            streamlines = ArraySequence()
            streamlines._data = positions
            streamlines._offsets = deepcopy(offsets)
            streamlines._lengths = lengths

        if space_str.lower() != 'rasmm' or origin_str.lower() != 'nifti' or \
                verify_invalid:
            if not dipy_available:
                logging.error('Dipy library is missing, advanced options '
                              'related to spatial transforms and invalid '
                              'streamlines are not available.')
                return
            from dipy.io.stateful_tractogram import StatefulTractogram

            space, origin = get_reverse_enum(space_str, origin_str)
            sft = StatefulTractogram(streamlines, reference, space, origin)
            if verify_invalid:
                rem, _ = sft.remove_invalid_streamlines()
                print('{} streamlines were removed becaused they were '
                      'invalid.'.format(len(rem)))
            sft.to_rasmm()
            sft.to_center()
            streamlines = sft.streamlines
            streamlines._offsets = offsets

        affine, dimensions, _, _ = get_reference_info_wrapper(reference)
        header = {
            "DIMENSIONS": dimensions.tolist(),
            "VOXEL_TO_RASMM": affine.tolist(),
            "NB_VERTICES": len(streamlines._data),
            "NB_STREAMLINES": len(streamlines)-1,
        }

        if header['NB_STREAMLINES'] <= 1:
            raise IOError('To use this script, you need at least 2'
                          'streamlines.')

        with open(os.path.join(tmp_dir_name, "header.json"), "w") as out_json:
            json.dump(header, out_json)

        curr_filename = os.path.join(tmp_dir_name, 'positions.3.{}'.format(
            positions_dtype))
        streamlines._data.astype(positions_dtype).tofile(
            curr_filename)
        curr_filename = os.path.join(tmp_dir_name, 'offsets.{}'.format(
            offsets_dtype))
        streamlines._offsets.astype(offsets_dtype).tofile(
            curr_filename)

        if dpv:
            os.mkdir(os.path.join(tmp_dir_name, 'dpv'))
            for arg in dpv:
                curr_arr = np.squeeze(load_matrix_in_any_format(arg[0]).astype(
                    arg[1]))
                if arg[1] == 'bool':
                    arg[1] = 'bit'
                if curr_arr.ndim > 2:
                    raise IOError('Maximum of 2 dimensions for dpv/dps/dpg.')
                dim = '' if curr_arr.ndim == 1 else '{}.'.format(
                    curr_arr.shape[-1])
                curr_filename = os.path.join(tmp_dir_name, 'dpv', '{}.{}{}'.format(
                    os.path.basename(os.path.splitext(arg[0])[0]), dim, arg[1]))
                curr_arr.tofile(curr_filename)

        if dps:
            os.mkdir(os.path.join(tmp_dir_name, 'dps'))
            for arg in dps:
                curr_arr = np.squeeze(load_matrix_in_any_format(arg[0]).astype(
                    arg[1]))
                if arg[1] == 'bool':
                    arg[1] = 'bit'
                if curr_arr.ndim > 2:
                    raise IOError('Maximum of 2 dimensions for dpv/dps/dpg.')
                dim = '' if curr_arr.ndim == 1 else '{}.'.format(
                    curr_arr.shape[-1])
                curr_filename = os.path.join(tmp_dir_name, 'dps', '{}.{}{}'.format(
                    os.path.basename(os.path.splitext(arg[0])[0]), dim, arg[1]))
                curr_arr.tofile(curr_filename)
        if groups:
            os.mkdir(os.path.join(tmp_dir_name, 'groups'))
            for arg in groups:
                curr_arr = load_matrix_in_any_format(arg[0]).astype(arg[1])
                if arg[1] == 'bool':
                    arg[1] = 'bit'
                if curr_arr.ndim > 2:
                    raise IOError('Maximum of 2 dimensions for dpv/dps/dpg.')
                dim = '' if curr_arr.ndim == 1 else '{}.'.format(
                    curr_arr.shape[-1])
                curr_filename = os.path.join(tmp_dir_name, 'groups', '{}.{}{}'.format(
                    os.path.basename(os.path.splitext(arg[0])[0]), dim, arg[1]))
                curr_arr.tofile(curr_filename)

        if dpg:
            os.mkdir(os.path.join(tmp_dir_name, 'dpg'))
            for arg in dpg:
                if not os.path.isdir(os.path.join(tmp_dir_name, 'dpg', arg[0])):
                    os.mkdir(os.path.join(tmp_dir_name, 'dpg', arg[0]))
                curr_arr = load_matrix_in_any_format(arg[1]).astype(arg[2])
                if arg[1] == 'bool':
                    arg[1] = 'bit'
                if curr_arr.ndim > 2:
                    raise IOError('Maximum of 2 dimensions for dpv/dps/dpg.')
                if curr_arr.shape == (1, 1):
                    curr_arr = curr_arr.reshape((1,))
                dim = '' if curr_arr.ndim == 1 else '{}.'.format(
                    curr_arr.shape[-1])
                curr_filename = os.path.join(tmp_dir_name, 'dpg', arg[0], '{}.{}{}'.format(
                    os.path.basename(os.path.splitext(arg[1])[0]), dim, arg[2]))
                curr_arr.tofile(curr_filename)

        trx = tmm.load(tmp_dir_name)
        tmm.save(trx, out_tractogram)
        trx.close()


def manipulate_trx_datatype(in_filename, out_filename, dict_dtype):
    trx = tmm.load(in_filename)

    # For each key in dict_dtype, we create a new memmap with the new dtype
    # and we copy the data from the old memmap to the new one.
    for key in dict_dtype:
        if key == 'positions':
            tmp_mm = np.memmap(tempfile.NamedTemporaryFile(),
                               dtype=dict_dtype[key],
                               mode='w+',
                               shape=trx.streamlines._data.shape)
            tmp_mm[:] = trx.streamlines._data[:]
            trx.streamlines._data = tmp_mm
        elif key == 'offsets':
            tmp_mm = np.memmap(tempfile.NamedTemporaryFile(),
                               dtype=dict_dtype[key],
                               mode='w+',
                               shape=trx.streamlines._offsets.shape)
            tmp_mm[:] = trx.streamlines._offsets[:]
            trx.streamlines._offsets = tmp_mm
        elif key == 'dpv':
            for key_dpv in dict_dtype[key]:
                tmp_mm = np.memmap(tempfile.NamedTemporaryFile(),
                                   dtype=dict_dtype[key][key_dpv],
                                   mode='w+',
                                   shape=trx.data_per_vertex[key_dpv]._data.shape)
                tmp_mm[:] = trx.data_per_vertex[key_dpv]._data[:]
                trx.data_per_vertex[key_dpv]._data = tmp_mm
        elif key == 'dps':
            for key_dps in dict_dtype[key]:
                tmp_mm = np.memmap(tempfile.NamedTemporaryFile(),
                                   dtype=dict_dtype[key][key_dps],
                                   mode='w+',
                                   shape=trx.data_per_streamline[key_dps].shape)
                tmp_mm[:] = trx.data_per_streamline[key_dps][:]
                trx.data_per_streamline[key_dps] = tmp_mm
        elif key == 'dpg':
            for key_group in dict_dtype[key]:
                for key_dpg in dict_dtype[key][key_group]:
                    tmp_mm = np.memmap(tempfile.NamedTemporaryFile(),
                                       dtype=dict_dtype[key][key_group][key_dpg],
                                       mode='w+',
                                       shape=trx.data_per_group[key_group][key_dpg].shape)
                    tmp_mm[:] = trx.data_per_group[key_group][key_dpg][:]
                    trx.data_per_group[key_group][key_dpg] = tmp_mm
        elif key == 'groups':
            for key_group in dict_dtype[key]:
                tmp_mm = np.memmap(tempfile.NamedTemporaryFile(),
                                   dtype=dict_dtype[key][key_group],
                                   mode='w+',
                                   shape=trx.groups[key_group].shape)
                tmp_mm[:] = trx.groups[key_group][:]
                trx.groups[key_group] = tmp_mm

    tmm.save(trx, out_filename)
    trx.close()
