#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

try:
    import dipy
    dipy_available = True
except ImportError:
    dipy_available = False

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
        obj = load(out_path, os.path.join(dir, 'gs.nii'))

    assert_allclose(obj.streamlines._data, coord, rtol=1e-04, atol=1e-06)
