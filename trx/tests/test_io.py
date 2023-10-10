#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

try:
    import dipy
    dipy_available = True
except ImportError:
    dipy_available = False

import trx.trx_file_memmap as tmm
from trx.trx_file_memmap import TrxFile
from trx.io import load, save, get_trx_tmpdir
from trx.fetcher import (get_testing_files_dict,
                         fetch_data, get_home)


fetch_data(get_testing_files_dict(), keys=['gold_standard.zip'])
tmp_dir = get_trx_tmpdir()


@pytest.mark.parametrize("path", [("gs.trx"), ("gs.trk"), ("gs.tck"),
                                  ("gs.vtk")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_load_vox(path):
    dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(dir, path)
    coord = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                    'gs_vox_space.txt'))
    obj = load(path, os.path.join(dir, 'gs.nii'))

    sft = obj.to_sft() if isinstance(obj, TrxFile) else obj
    sft.to_vox()

    assert_allclose(sft.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", [("gs.trx"), ("gs.trk"), ("gs.tck"),
                                  ("gs.vtk")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_load_voxmm(path):
    dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(dir, path)
    coord = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                    'gs_voxmm_space.txt'))
    obj = load(path, os.path.join(dir, 'gs.nii'))

    sft = obj.to_sft() if isinstance(obj, TrxFile) else obj
    sft.to_voxmm()

    assert_allclose(sft.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", [("gs.trk"), ("gs.trx"), ("gs_fldr.trx")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_multi_load_save_rasmm(path):
    dir = os.path.join(get_home(), 'gold_standard')
    basename, ext = os.path.splitext(path)
    out_path = os.path.join(tmp_dir.name, '{}_tmp{}'.format(basename, ext))
    path = os.path.join(dir, path)
    coord = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                    'gs_rasmm_space.txt'))

    obj = load(path, os.path.join(dir, 'gs.nii'))
    for _ in range(100):
        save(obj, out_path)
        if isinstance(obj, TrxFile):
            obj.close()
        obj = load(out_path, os.path.join(dir, 'gs.nii'))

    assert_allclose(obj.streamlines._data, coord, rtol=1e-04, atol=1e-06)


@pytest.mark.parametrize("path", [("gs.trx"), ("gs_fldr.trx")])
@pytest.mark.skipif(not dipy_available, reason='Dipy is not installed.')
def test_close_tmp_file(path):
    dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(dir, path)

    trx = tmm.load(path)
    tmp_dir = deepcopy(trx._uncompressed_folder_handle)
    if os.path.isfile(path):
        assert os.path.isdir(tmp_dir.name)
    sft = trx.to_sft()
    trx.close()

    coord_rasmm = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                          'gs_rasmm_space.txt'))
    coord_vox = np.loadtxt(os.path.join(get_home(), 'gold_standard',
                                        'gs_vox_space.txt'))

    # The folder trx representation does not need tmp files
    if os.path.isfile(path):
        assert not os.path.isdir(tmp_dir.name)
    assert_allclose(sft.streamlines._data, coord_rasmm, rtol=1e-04, atol=1e-06)
    sft.to_vox()
    assert_allclose(sft.streamlines._data, coord_vox, rtol=1e-04, atol=1e-06)


@pytest.mark.parametrize("tmp_path", [("~"), ("use_working_dir")])
def test_close_tmp_file(tmp_path):
    dir = os.path.join(get_home(), 'gold_standard')
    path = os.path.join(dir, 'gs.trx')

    if tmp_path == 'use_working_dir':
        os.environ['TRX_TMPDIR'] = 'use_working_dir'
    else:
        os.environ['TRX_TMPDIR'] = os.path.expanduser(tmp_path)

    trx = tmm.load(path)
    tmp_dir = deepcopy(trx._uncompressed_folder_handle)

    if tmp_path == 'use_working_dir':
        assert os.path.dirname(tmp_dir.name) == os.getcwd()
    else:
        assert os.path.dirname(tmp_dir.name) == os.path.expanduser(tmp_path)

    trx.close()
    assert not os.path.isdir(tmp_dir.name)
