#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from nibabel.streamlines.tests.test_tractogram import make_dummy_streamline
import tempfile

import pytest
from nibabel.streamlines import LazyTractogram, Tractogram
import numpy as np
import trx.trx_file_memmap as tmm
from trx.fetcher import (get_testing_files_dict,
                         fetch_data, get_home)
from tempfile import mkdtemp

import pathlib
import os

fetch_data(get_testing_files_dict(), keys=['memmap_test_data.zip'])
tmp_dir = tempfile.TemporaryDirectory()


# @pytest.mark.parametrize(
#     "arr,expected,value_error",
#     [
#         (np.ones((5, 5, 5), dtype=np.int16), None, True),
#         (np.ones((5, 4), dtype=np.int16), "mean_fa.4.int16", False),
#         (np.ones((5, 4), dtype=np.float64), "mean_fa.4.float64", False),
#         (np.ones((5, 1), dtype=np.float64), "mean_fa.float64", False),
#         (np.ones((1), dtype=np.float64), "mean_fa.float64", False),
#     ],
# )
# def test__generate_filename_from_data(
#     arr, expected, value_error, filename="mean_fa.bit"
# ):

#     if value_error:
#         with pytest.raises(ValueError):
#             new_fn = tmm._generate_filename_from_data(arr=arr,
#                                                       filename=filename)
#             assert new_fn is None
#     else:
#         new_fn = tmm._generate_filename_from_data(arr=arr, filename=filename)
#         assert new_fn == expected


# @pytest.mark.parametrize(
#     "filename,expected,value_error",
#     [
#         ("mean_fa.float64", ("mean_fa", 1, ".float64"), False),
#         ("mean_fa.5.int32", ("mean_fa", 5, ".int32"), False),
#         ("mean_fa", None, True),
#         ("mean_fa.5.4.int32", None, True),
#         pytest.param(
#             "mean_fa.fa", None, True, marks=pytest.mark.xfail,
#             id="invalid extension"
#         ),
#     ],
# )
# def test__split_ext_with_dimensionality(filename, expected, value_error):
#     if value_error:
#         with pytest.raises(ValueError):
#             assert tmm._split_ext_with_dimensionality(filename) == expected
#     else:
#         assert tmm._split_ext_with_dimensionality(filename) == expected


# @pytest.mark.parametrize(
#     "offsets,nb_vertices,expected",
#     [
#         (np.array(range(5), dtype=np.int16), 4, np.array([1, 1, 1, 1, 0])),
#         (np.array([0, 1, 1, 3, 4], dtype=np.int32),
#          4, np.array([1, 0, 2, 1, 0])),
#         (np.array(range(4), dtype=np.uint64), 4, np.array([1, 1, 1, 1])),
#         pytest.param(np.array([0, 1, 0, 3, 4], dtype=np.int16), 4,
#                      np.array([1, 3, 0, 1, 0]), marks=pytest.mark.xfail,
#                      id="offsets not sorted"),
#     ],
# )
# def test__compute_lengths(offsets, nb_vertices, expected):

#     offsets = tmm._append_last_offsets(offsets, nb_vertices)
#     lengths = tmm._compute_lengths(offsets=offsets)
#     assert np.array_equal(lengths, expected)


# @pytest.mark.parametrize(
#     "ext,expected",
#     [
#         (".bit", True),
#         (".int16", True),
#         (".float32", True),
#         (".ushort", True),
#         (".txt", False),
#     ],
# )
# def test__is_dtype_valid(ext, expected):
#     assert tmm._is_dtype_valid(ext) == expected


# @pytest.mark.parametrize(
#     "arr,l_bound,r_bound,expected",
#     [
#         (np.array(range(5), dtype=np.int16), None, None, 4),
#         (np.array([0, 1, 0, 3, 4], dtype=np.int16), None, None, 1),
#         (np.array([0, 1, 2, 0, 4], dtype=np.int16), None, None, 2),
#         (np.array(range(5), dtype=np.int16), 1, 2, 2),
#         (np.array(range(5), dtype=np.int16), 3, 3, 3),
#         (np.zeros((5), dtype=np.int16), 3, 3, -1),
#     ],
# )
# def test__dichotomic_search(arr, l_bound, r_bound, expected):
#     end_idx = tmm._dichotomic_search(arr, l_bound=l_bound, r_bound=r_bound)
#     assert end_idx == expected


# @pytest.mark.parametrize(
#     "basename, create, expected",
#     [
#         ("offsets.int16", True, np.array(range(12), dtype=np.int16).reshape((
#             3, 4))),
#         ("offsets.float32", False, None),
#     ],
# )
# def test__create_memmap(basename, create, expected):
#     if create:
#         # Need to create array before evaluating
#         filename = os.path.join(mkdtemp(), basename)
#         fp = np.memmap(filename, dtype=np.int16, mode="w+", shape=(3, 4))
#         fp[:] = expected[:]
#         mmarr = tmm._create_memmap(filename=filename, shape=(3, 4),
#                                    dtype=np.int16)
#         assert np.array_equal(mmarr, expected)

#     else:
#         filename = os.path.join(mkdtemp(), basename)
#         mmarr = tmm._create_memmap(filename=filename, shape=(0,),
#                                    dtype=np.int16)
#         assert os.path.isfile(filename)
#         assert np.array_equal(mmarr, np.zeros(shape=(0,), dtype=np.float32))


# # need dpg test with missing keys
# @pytest.mark.parametrize(
#     "path,check_dpg,value_error",
#     [
#         ("small_compressed.trx", False, False),
#         ("small.trx", True, False),
#         ("small_fldr.trx", False, False),
#         ("dontexist.trx", False, True),
#     ],
# )
# def test_load(path, check_dpg, value_error):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     # Need to perhaps improve test
#     if value_error:
#         with pytest.raises(ValueError):
#             assert not isinstance(
#                 tmm.load(input_obj=path, check_dpg=check_dpg), tmm.TrxFile
#             )
#     else:
#         assert isinstance(tmm.load(input_obj=path, check_dpg=check_dpg),
#                           tmm.TrxFile)


# @pytest.mark.parametrize("path", [("small.trx")])
# def test_load_zip(path):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     assert isinstance(tmm.load_from_zip(path), tmm.TrxFile)


# @pytest.mark.parametrize("path", [("small_fldr.trx")])
# def test_load_directory(path):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     assert isinstance(tmm.load_from_directory(path), tmm.TrxFile)


# @pytest.mark.parametrize("path", [("small.trx")])
# def test_concatenate(path):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     trx1 = tmm.load(path)
#     trx2 = tmm.load(path)
#     concat = tmm.concatenate([trx1, trx2])
#     assert len(concat) == 2 * len(trx2)


# @pytest.mark.parametrize("path", [("small.trx")])
# def test_resize(path):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     trx1 = tmm.load(path)
#     concat = tmm.TrxFile(nb_vertices=1000000, nb_streamlines=10000,
#                          init_as=trx1)

#     tmm.concatenate([concat, trx1], preallocation=True, delete_groups=True)
#     concat.resize()
#     assert len(concat) == len(trx1)


# @pytest.mark.parametrize(
#     "path, buffer",
#     [
#         ("small.trx", 10000),
#         ("small.trx", 0)
#     ]
# )
# def test_append(path, buffer):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     trx1 = tmm.load(path)
#     concat = tmm.TrxFile(nb_vertices=1, nb_streamlines=1,
#                          init_as=trx1)

#     concat.append(trx1, extra_buffer=buffer)
#     if buffer > 0:
#         concat.resize()
#     assert len(concat) == len(trx1)


# @pytest.mark.parametrize("path, buffer", [("small.trx", 10000)])
# def test_append_StatefulTractogram(path, buffer):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     trx = tmm.load(path)
#     obj = trx.to_sft()
#     concat = tmm.TrxFile(nb_vertices=1, nb_streamlines=1, init_as=trx)

#     concat.append(obj, extra_buffer=buffer)
#     if buffer > 0:
#         concat.resize()
#     assert len(concat) == len(obj)

# @pytest.mark.parametrize("path, buffer", [("small.trx", 10000)])
# def test_append_Tractogram(path, buffer):
#     path = os.path.join(get_home(), 'memmap_test_data', path)
#     trx = tmm.load(path)
#     obj = trx.to_tractogram()
#     concat = tmm.TrxFile(nb_vertices=1, nb_streamlines=1, init_as=trx)

#     concat.append(obj, extra_buffer=buffer)
#     if buffer > 0:
#         concat.resize()
#     assert len(concat) == len(obj)


@pytest.mark.parametrize("path, buffer", [("small.trx", 10000)])
def test_append_LazyTractogram(path, buffer):
    DATA = {}
    DATA['rng'] = np.random.RandomState(1776)
    DATA['streamlines'] = []
    DATA['fa'] = []
    DATA['commit_weights'] = []
    DATA['clusters_QB'] = []
    for i in [1, 2, 5, 2, 1]:
        data = make_dummy_streamline(i)
        streamline, data_per_point, data_for_streamline = data
        DATA['streamlines'].append(streamline)
        DATA['fa'].append(data_per_point['fa'].astype(np.float16))
        DATA['commit_weights'].append(
            data_for_streamline['mean_curvature'].astype(np.float32))
        DATA['clusters_QB'].append(
            data_for_streamline['mean_torsion'].astype(np.uint16))

    DATA['data_per_point'] = {'fa': [e for e in DATA['fa']]}
    DATA['data_per_streamline'] = {
        'commit_weights': [e for e in DATA['commit_weights']],
        'clusters_QB': [e for e in DATA['clusters_QB']]}

    DATA['streamlines_func'] = lambda: (e for e in DATA['streamlines'])
    DATA['data_per_point_func'] = {'fa': lambda: (e for e in DATA['fa'])}
    DATA['data_per_streamline_func'] = {
        'commit_weights': lambda: (e for e in DATA['commit_weights']),
        'clusters_QB': lambda: (e for e in DATA['clusters_QB'])}

    obj = LazyTractogram(DATA['streamlines_func'],
                         DATA['data_per_streamline_func'],
                         DATA['data_per_point_func'],
                         affine_to_rasmm=np.eye(4))
    # print(DATA)
    path = os.path.join(get_home(), 'memmap_test_data', path)
    trx = tmm.TrxFile.from_tractogram(Tractogram(DATA['streamlines'],
                                         DATA['data_per_streamline'],
                                         DATA['data_per_point'],
                                         affine_to_rasmm=np.eye(4)),
                                         reference=path)
    concat = tmm.TrxFile(nb_vertices=0, nb_streamlines=0, init_as=trx)

    # for key in ['color_x', 'color_y', 'color_z']:
    #     del concat.data_per_vertex[key]
    # for key in ['commit_colors', 'algo']:
    #     del concat.data_per_streamline[key]

    concat.append(obj, extra_buffer=buffer)
    print(concat)
    assert False
    # if buffer > 0:
    #     concat.resize()

    # assert len(concat) == len(obj)


# def test_zip_from_folder():
#     pass


# def test_trxfile_init():
#     pass


# def test_trxfile_print():
#     pass


# def test_trxfile_len():
#     fake = tmm.TrxFile(nb_vertices=100, nb_streamlines=10)
#     assert len(fake) == 10


# def test_trxfile_getitem():
#     pass


# def test_trxfile_deepcopy():
#     pass


# def test_get_real_len():
#     fake = tmm.TrxFile(nb_vertices=100, nb_streamlines=10)
#     assert fake._get_real_len() == (0, 0)


# def test_copy_fixed_arrays_from():
#     pass


# def test_initialize_empty_trx():
#     pass


# def test_create_trx_from_pointer():
#     pass


# def test_trxfile_getgroup():
#     pass


# def test_trxfile_select():
#     pass


# def test_trxfile_to_memory():
#     pass


# def test_trxfile_close():
#     pass
