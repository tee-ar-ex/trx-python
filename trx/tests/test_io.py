#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import os
import psutil

import pytest
import numpy as np
from numpy.testing import assert_allclose

try:
    import dipy
    from dipy.io.streamline import save_tractogram
    dipy_available = True
except ImportError:
    dipy_available = False

import trx.trx_file_memmap as tmm
from trx.trx_file_memmap import TrxFile
from trx.io import load, save, get_trx_tmp_dir
from trx.fetcher import (get_testing_files_dict,
                         fetch_data, get_home)


fetch_data(get_testing_files_dict(), keys=['gold_standard.zip'])
tmp_gs_dir = get_trx_tmp_dir()


@pytest.mark.parametrize("path", [("gs.trk"), ("gs.tck"),
                                  ("gs.vtk")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_load_vox(path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(gs_dir, path)

    obj = load(path, os.path.join(gs_dir, 'gs.nii'))
    sft = obj.to_sft()
    save_tractogram(sft, path)
    obj.close()
    save_tractogram(sft, path)


@pytest.mark.parametrize("path", [("gs.trx"), ("gs.trk"), ("gs.tck"),
                                  ("gs.vtk")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_load_vox(path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(gs_dir, path)
    coord = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                    'gs_vox_space.txt'))
    obj = load(path, os.path.join(gs_dir, 'gs.nii'))

    sft = obj.to_sft() if isinstance(obj, TrxFile) else obj
    sft.to_vox()

    assert_allclose(sft.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", [("gs.trx"), ("gs.trk"), ("gs.tck"),
                                  ("gs.vtk")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_load_voxmm(path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(gs_dir, path)
    coord = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                    'gs_voxmm_space.txt'))
    obj = load(path, os.path.join(gs_dir, 'gs.nii'))

    sft = obj.to_sft() if isinstance(obj, TrxFile) else obj
    sft.to_voxmm()

    assert_allclose(sft.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", [("gs.trk"), ("gs.trx"), ("gs_fldr.trx")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_multi_load_save_rasmm(path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    basename, ext = os.path.splitext(path)
    out_path = os.path.join(tmp_gs_dir.name, '{}_tmp{}'.format(basename, ext))
    path = os.path.join(gs_dir, path)
    coord = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                    'gs_rasmm_space.txt'))

    obj = load(path, os.path.join(gs_dir, 'gs.nii'))
    for _ in range(100):
        save(obj, out_path)
        if isinstance(obj, TrxFile):
            obj.close()
        obj = load(out_path, os.path.join(gs_dir, 'gs.nii'))

    assert_allclose(obj.streamlines._data, coord, rtol=1e-04, atol=1e-06)


@pytest.mark.parametrize("path", [("gs.trx"), ("gs_fldr.trx")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_delete_tmp_gs_dir(path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(gs_dir, path)

    trx1 = tmm.load(path)
    if os.path.isfile(path):
        tmp_gs_dir = deepcopy(trx1._uncompressed_folder_handle.name)
        assert os.path.isdir(tmp_gs_dir)
    sft = trx1.to_sft()
    trx1.close()

    coord_rasmm = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                          'gs_rasmm_space.txt'))
    coord_vox = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                        'gs_vox_space.txt'))

    # The folder trx representation does not need tmp files
    if os.path.isfile(path):
        assert not os.path.isdir(tmp_gs_dir)

    assert_allclose(sft.streamlines._data, coord_rasmm, rtol=1e-04, atol=1e-06)

    # Reloading the TRX and checking its data, then closing
    trx2 = tmm.load(path)
    assert_allclose(trx2.streamlines._data,
                    sft.streamlines._data, rtol=1e-04, atol=1e-06)
    trx2.close()

    sft.to_vox()
    assert_allclose(sft.streamlines._data, coord_vox, rtol=1e-04, atol=1e-06)

    trx3 = tmm.load(path)
    assert_allclose(trx3.streamlines._data,
                    coord_rasmm, rtol=1e-04, atol=1e-06)
    trx3.close()


@pytest.mark.parametrize("path", [("gs.trx")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_close_tmp_files(path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(gs_dir, path)

    trx = tmm.load(path)
    process = psutil.Process(os.getpid())
    open_files = process.open_files()

    count = 0
    for open_file in open_files:
        if 'trx' in open_file.path:
            count += 1

    assert count == 6
    trx.close()

    open_files = process.open_files()
    count = 0
    for open_file in open_files:
        if 'trx' in open_file.path:
            count += 1
    assert not count


@pytest.mark.parametrize("tmp_path", [("~"), ("use_working_dir")])
def test_change_tmp_dir(tmp_path):
    gs_dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(gs_dir, 'gs.trx')

    if tmp_path == 'use_working_dir':
        os.environ['TRX_TMPDIR'] = 'use_working_dir'
    else:
        os.environ['TRX_TMPDIR'] = os.path.expanduser(tmp_path)

    trx = tmm.load(path)
    tmp_gs_dir = deepcopy(trx._uncompressed_folder_handle.name)

    if tmp_path == 'use_working_dir':
        assert os.path.dirname(tmp_gs_dir) == os.getcwd()
    else:
        assert os.path.dirname(tmp_gs_dir) == os.path.expanduser(tmp_path)

    trx.close()
    assert not os.path.isdir(tmp_gs_dir)
