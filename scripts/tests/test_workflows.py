#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
try:
    from dipy.io.streamline import load_tractogram
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.trx_file_memmap import load
from trx.fetcher import (get_testing_files_dict,
                         fetch_data, get_home)
from trx.workflows import (convert_dsi_studio,
                           convert_tractogram,
                           generate_trx_from_scratch)


# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['DSI.zip', 'trx_from_scratch.zip'])
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


def test_help_option_generate_trx_from_scratch(script_runner):
    ret = script_runner.run('tff_generate_trx_from_scratch.py', '--help')
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


def test_execution_generate_trx_from_scratch():
    os.chdir(os.path.expanduser(tmp_dir.name))
    reference_fa = os.path.join(get_home(), 'trx_from_scratch',
                                'fa.nii.gz')
    raw_arr_dir = os.path.join(get_home(), 'trx_from_scratch',
                               'test_npy')
    expected_trx = os.path.join(get_home(), 'trx_from_scratch',
                                'expected.trx')

    dpv = [(os.path.join(raw_arr_dir, 'dpv_cx.npy'), 'uint8'),
           (os.path.join(raw_arr_dir, 'dpv_cy.npy'), 'uint8'),
           (os.path.join(raw_arr_dir, 'dpv_cz.npy'), 'uint8')]
    dps = [(os.path.join(raw_arr_dir, 'dps_algo.npy'), 'uint8'),
           (os.path.join(raw_arr_dir, 'dps_cw.npy'), 'float64')]
    dpg = [('g_AF_L', os.path.join(raw_arr_dir, 'dpg_AF_L_mean_fa.npy'), 'float32'),
           ('g_AF_R', os.path.join(raw_arr_dir, 'dpg_AF_R_mean_fa.npy'), 'float32'),
           ('g_AF_L', os.path.join(raw_arr_dir, 'dpg_AF_L_volume.npy'), 'float32')]
    groups = [(os.path.join(raw_arr_dir, 'g_AF_L.npy'), 'int32'),
              (os.path.join(raw_arr_dir, 'g_AF_R.npy'), 'int32'),
              (os.path.join(raw_arr_dir, 'g_CST_L.npy'), 'int32')]
    print(os.path.join(raw_arr_dir,
                                                   'offsets.npy'))
    generate_trx_from_scratch(reference_fa, 'generated.trx',
                              positions=os.path.join(raw_arr_dir,
                                                     'positions.npy'),
                              offsets=os.path.join(raw_arr_dir,
                                                   'offsets.npy'),
                              positions_dtype='float16',
                              offsets_dtype='uint64',
                              space_str='rasmm', origin_str='nifti',
                              verify_invalid=True, dpv=dpv, dps=dps,
                              groups=groups, dpg=dpg)

    exp_trx = load(expected_trx)
    gen_trx = load('generated.trx')

    assert_allclose(exp_trx.streamlines._data, gen_trx.streamlines._data,
                    atol=0.1, rtol=0.1)
    assert_equal(exp_trx.streamlines._offsets, gen_trx.streamlines._offsets)

    for key in exp_trx.data_per_vertex.keys():
        assert_equal(exp_trx.data_per_vertex[key]._data,
                     gen_trx.data_per_vertex[key]._data)
        assert_equal(exp_trx.data_per_vertex[key]._offsets,
                     gen_trx.data_per_vertex[key]._offsets)
    for key in exp_trx.data_per_streamline.keys():
        assert_equal(exp_trx.data_per_streamline[key],
                     gen_trx.data_per_streamline[key])
    for key in exp_trx.groups.keys():
        assert_equal(exp_trx.groups[key], gen_trx.groups[key])

    for group in exp_trx.groups.keys():
        if group in exp_trx.data_per_group:
            for key in exp_trx.data_per_group[group].keys():
                assert_equal(exp_trx.data_per_group[group][key],
                             gen_trx.data_per_group[group][key])
