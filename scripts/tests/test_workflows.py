#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import tempfile

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
try:
    from dipy.io.streamline import load_tractogram
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.trx_file_memmap import load
from trx.fetcher import (get_testing_files_dict,
                                              fetch_data, get_home)
from trx.workflows import (convert_dsi_studio,
                                                convert_tractogram)


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['DSI.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option_convert_dsi(script_runner):
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    ret = script_runner.run('tff_convert_dsi_studio.py', '--help')
    assert ret.success


def test_help_option_convert(script_runner):
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    ret = script_runner.run('tff_convert_tractogram.py', '--help')
    assert ret.success


def test_execution_convert_dsi():
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_trk = os.path.join(get_home(), 'DSI',
                          'CC.trk.gz')
    in_nii = os.path.join(get_home(), 'DSI',
                          'CC.nii.gz')
    exp_data = os.path.join(get_home(), 'DSI',
                            'CC_fix_data.npy')
    exp_offsets = os.path.join(get_home(), 'DSI',
                               'CC_fix_offsets.npy')
    convert_dsi_studio(in_trk, in_nii, 'fixed.trk',
                       remove_invalid=False,
                       keep_invalid=True)

    data_fix = np.load(exp_data)
    offsets_fix = np.load(exp_offsets)

    sft = load_tractogram('fixed.trk', 'same')
    assert_equal(sft.streamlines._data, data_fix)
    assert_equal(sft.streamlines._offsets, offsets_fix)


def test_execution_convert_to_trx():
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_trk = os.path.join(get_home(), 'DSI',
                          'CC_fix.trk')
    exp_data = os.path.join(get_home(), 'DSI',
                            'CC_fix_data.npy')
    exp_offsets = os.path.join(get_home(), 'DSI',
                               'CC_fix_offsets.npy')
    convert_tractogram(in_trk, 'CC_fix.trx', None)

    data_fix = np.load(exp_data)
    offsets_fix = np.load(exp_offsets)

    trx = load('CC_fix.trx')
    assert_equal(trx.streamlines._data.dtype, np.float32)
    assert_equal(trx.streamlines._offsets.dtype, np.uint32)
    assert_array_equal(trx.streamlines._data, data_fix)
    assert_array_equal(trx.streamlines._offsets, offsets_fix)


def test_execution_convert_from_trx():
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_trk = os.path.join(get_home(), 'DSI',
                          'CC_fix.trk')
    in_nii = os.path.join(get_home(), 'DSI',
                          'CC.nii.gz')
    exp_data = os.path.join(get_home(), 'DSI',
                            'CC_fix_data.npy')
    exp_offsets = os.path.join(get_home(), 'DSI',
                               'CC_fix_offsets.npy')

    # Sequential conversions
    convert_tractogram(in_trk, 'CC_fix.trx', None)
    convert_tractogram('CC_fix.trx', 'CC_converted.tck', None)
    convert_tractogram('CC_fix.trx', 'CC_converted.trk', None)

    data_fix = np.load(exp_data)
    offsets_fix = np.load(exp_offsets)

    sft = load_tractogram('CC_converted.trk', 'same')
    assert_equal(sft.streamlines._data, data_fix)
    assert_equal(sft.streamlines._offsets, offsets_fix)

    sft = load_tractogram('CC_converted.tck', in_nii)
    assert_equal(sft.streamlines._data, data_fix)
    assert_equal(sft.streamlines._offsets, offsets_fix)


def test_execution_convert_dtype_p16_o64():
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_trk = os.path.join(get_home(), 'DSI',
                          'CC_fix.trk')
    convert_tractogram(in_trk, 'CC_fix_p16_o64.trx', None,
                       pos_dtype='float16', offsets_dtype='uint64')

    trx = load('CC_fix_p16_o64.trx')
    assert_equal(trx.streamlines._data.dtype, np.float16)
    assert_equal(trx.streamlines._offsets.dtype, np.uint64)


def test_execution_convert_dtype_p64_o32():
    if not dipy_available:
        pytest.skip('Dipy library is missing, cannot test scripts involving '
                      'tck/trk/vtk.')
    os.chdir(os.path.expanduser(tmp_dir.name))
    in_trk = os.path.join(get_home(), 'DSI',
                          'CC_fix.trk')
    convert_tractogram(in_trk, 'CC_fix_p64_o32.trx', None,
                       pos_dtype='float64', offsets_dtype='uint32')

    trx = load('CC_fix_p64_o32.trx')
    assert_equal(trx.streamlines._data.dtype, np.float64)
    assert_equal(trx.streamlines._offsets.dtype, np.uint32)
