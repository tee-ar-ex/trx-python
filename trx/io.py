#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

try:
    import dipy
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.io.streamline import load_tractogram, save_tractogram
    from dipy.io.utils import is_header_compatible
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.utils import split_name_with_gz
from trx.trx_file_memmap import load, save, TrxFile


def load_sft_with_reference(filepath, reference=None,
                            bbox_check=True):
    if not dipy_available:
        logging.error('Dipy library is missing, cannot use functions related '
                      'to the StatefulTractogram.')
        return None
    # Force the usage of --reference for all file formats without an header
    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        if reference is not None and reference != 'same':
            logging.warning('Reference is discarded for this file format '
                            '{}.'.format(filepath))
        sft = load_tractogram(filepath, 'same',
                              bbox_valid_check=bbox_check)
    elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
        if reference is None or reference == 'same':
            raise IOError('--reference is required for this file format '
                          '{}.'.format(filepath))
        else:
            sft = load_tractogram(filepath, reference,
                                  bbox_valid_check=bbox_check)

    else:
        raise IOError('{} is an unsupported file format'.format(filepath))

    return sft


def load_wrapper(tractogram_filename, reference):
    in_ext = split_name_with_gz(tractogram_filename)[1]
    if in_ext != '.trx' and not os.path.isdir(tractogram_filename):
        tractogram_obj = load_sft_with_reference(tractogram_filename,
                                                 reference,
                                                 bbox_check=False)
    else:
        tractogram_obj = load(tractogram_filename)

    return tractogram_obj


def save_wrapper(tractogram_obj, tractogram_filename):
    out_ext = split_name_with_gz(tractogram_filename)[1]

    if out_ext != '.trx':
        if not isinstance(tractogram_obj, StatefulTractogram):
            tractogram_obj = tractogram_obj.to_sft()
        save_tractogram(tractogram_obj, tractogram_filename,
                        bbox_valid_check=False)
    else:
        if not isinstance(tractogram_obj, TrxFile):
            tractogram_obj = TrxFile.from_sft(tractogram_obj)
        save(tractogram_obj, tractogram_filename)


def concatenate_trx(trx_list, erase_metadata=False, metadata_fake_init=False):
    """ Concatenate a list of TrxFile together """

    if not is_header_compatible(trx, trx_list[0]):
        raise ValueError('Incompatible TRX, check space attributes.')

    if erase_metadata:
        trx_list[0].data_per_point = {}
        trx_list[0].data_per_streamline = {}

    for trx in trx_list[1:]:
        if erase_metadata:
            trx.data_per_point = {}
            trx.data_per_streamline = {}
        elif metadata_fake_init:
            for dps_key in list(trx.data_per_streamline.keys()):
                if dps_key not in trx_list[0].data_per_streamline.keys():
                    del trx.data_per_streamline[dps_key]
            for dpp_key in list(trx.data_per_point.keys()):
                if dpp_key not in trx_list[0].data_per_point.keys():
                    del trx.data_per_point[dpp_key]

            for dps_key in trx_list[0].data_per_streamline.keys():
                if dps_key not in trx.data_per_streamline:
                    arr_shape =\
                        list(trx_list[0].data_per_streamline[dps_key].shape)
                    arr_shape[0] = len(trx)
                    trx.data_per_streamline[dps_key] = np.zeros(arr_shape)
            for dpp_key in trx_list[0].data_per_point.keys():
                if dpp_key not in trx.data_per_point:
                    arr_seq = ArraySequence()
                    arr_seq_shape = list(
                        trx_list[0].data_per_point[dpp_key]._data.shape)
                    arr_seq_shape[0] = len(trx.streamlines._data)
                    arr_seq._data = np.zeros(arr_seq_shape)
                    arr_seq._offsets = trx.streamlines._offsets
                    arr_seq._lengths = trx.streamlines._lengths
                    trx.data_per_point[dpp_key] = arr_seq

    total_streamlines = 0
    total_points = 0
    lengths = []
    for trx in trx_list:
        total_streamlines += len(trx.streamlines._offsets)
        total_points += len(trx.streamlines._data)
        lengths.extend(trx.streamlines._lengths)
    lengths = np.array(lengths, dtype=np.uint32)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.uint64)

    dpp = {}
    for dpp_key in trx_list[0].data_per_point.keys():
        arr_seq_shape = list(trx_list[0].data_per_point[dpp_key]._data.shape)
        arr_seq_shape[0] = total_points
        dpp[dpp_key] = ArraySequence()
        dpp[dpp_key]._data = np.zeros(arr_seq_shape)
        dpp[dpp_key]._lengths = lengths
        dpp[dpp_key]._offsets = offsets

    dps = {}
    for dps_key in trx_list[0].data_per_streamline.keys():
        arr_seq_shape = list(trx_list[0].data_per_streamline[dps_key].shape)
        arr_seq_shape[0] = total_streamlines
        dps[dps_key] = np.zeros(arr_seq_shape)

    streamlines = ArraySequence()
    streamlines._data = np.zeros((total_points, 3))
    streamlines._lengths = lengths
    streamlines._offsets = offsets

    pts_counter = 0
    strs_counter = 0
    for trx in trx_list:
        pts_curr_len = len(trx.streamlines._data)
        strs_curr_len = len(trx.streamlines._offsets)

        if strs_curr_len == 0 or pts_curr_len == 0:
            continue

        streamlines._data[pts_counter:pts_counter+pts_curr_len] = \
            trx.streamlines._data

        for dpp_key in trx_list[0].data_per_point.keys():
            dpp[dpp_key]._data[pts_counter:pts_counter+pts_curr_len] = \
                trx.data_per_point[dpp_key]._data
        for dps_key in trx_list[0].data_per_streamline.keys():
            dps[dps_key][strs_counter:strs_counter+strs_curr_len] = \
                trx.data_per_streamline[dps_key]
        pts_counter += pts_curr_len
        strs_counter += strs_curr_len

    # fused_trx = StatefulTractogram.from_trx(streamlines, trx_list[0],
    #                                         data_per_point=dpp,
    #                                         data_per_streamline=dps)
    # return fused_sft
