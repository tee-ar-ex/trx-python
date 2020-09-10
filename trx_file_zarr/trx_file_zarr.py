from copy import deepcopy
import yaml
import logging
import os
import shutil
import tempfile
import zipfile

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.utils import get_reference_info
from nibabel.affines import voxel_sizes
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import PerArraySequenceDict, PerArrayDict
import numpy as np
import zarr
from zarr.util import TreeViewer


def compute_lengths(offsets, nb_points):
    """ Compute lengths from offsets and header information """
    if len(offsets) > 1:
        lengths = np.ediff1d(offsets, to_end=nb_points-offsets[-1])
    elif len(offsets) == 1:
        lengths = np.array([nb_points])
    else:
        lengths = np.array([0])

    return lengths.astype(np.uint32)


def load(input_obj, directory_func=zarr.DirectoryStore):
    if os.path.isdir(input_obj):
        store = directory_func(input_obj)
    elif os.path.isfile(input_obj):
        store = zarr.ZipStore(input_obj)

    trx = TrxFile()
    trx._zcontainer = zarr.group(store=store, overwrite=False)

    return trx


def _check_same_keys(key_1, key_2):
    key_1 = list(key_1)
    key_2 = list(key_2)
    key_1.sort()
    key_2.sort()
    return key_1 == key_2


class TrxFile():
    """ Core class of the TrxFile """

    def __init__(self, init_as=None, reference=None):
        """ Initialize an empty TrxFile, support preallocation """
        if init_as is not None:
            affine = init_as._zcontainer.attrs['VOXEL_TO_RASMM']
            dimensions = init_as._zcontainer.attrs['DIMENSIONS']
        elif reference is not None:
            affine, dimensions, _, _ = get_reference_info(reference)
        else:
            logging.debug('No reference provided, using blank space '
                          'attributes, please update them later.')
            affine = np.eye(4).astype(np.float32)
            dimensions = [1, 1, 1]

        store_TMP = zarr.storage.TempStore()
        self._zcontainer = zarr.group(store=store_TMP, overwrite=True)
        self.voxel_to_rasmm = affine
        self.dimensions = dimensions
        self.nb_points = 0
        self.nb_streamlines = 0

        if init_as:
            positions_dtype = init_as._zcontainer['positions'].dtype
        else:
            positions_dtype = np.float16
        self._zcontainer.create_dataset('positions', shape=(0, 3),
                                        chunks=(100000, None),
                                        dtype=positions_dtype)

        self._zcontainer.create_dataset('offsets', shape=(0,),
                                        chunks=(10000,), dtype=np.uint64)

        self._zcontainer.create_group('data_per_point')
        self._zcontainer.create_group('data_per_streamline')
        self._zcontainer.create_group('data_per_group')
        self._zcontainer.create_group('groups')

        if init_as is None:
            return

        # if len(init_as._zdpp):
        # self._zcontainer.create_group('data_per_point')
        for dpp_key in init_as._zdpp.array_keys():
            empty_shape = list(init_as._zdpp[dpp_key].shape)
            empty_shape[0] = 0
            dtype = init_as._zdpp[dpp_key].dtype
            chunks = [100000]
            for _ in range(len(empty_shape)-1):
                chunks.append(None)

            self._zdpp.create_dataset(dpp_key, shape=empty_shape,
                                      chunks=chunks, dtype=dtype)

        # if len(init_as._zdps):
        # self._zcontainer.create_group('data_per_streamline')
        for dps_key in init_as._zdps.array_keys():
            empty_shape = list(init_as._zdps[dps_key].shape)
            empty_shape[0] = 0
            dtype = init_as._zdps[dps_key].dtype
            chunks = [10000]
            for _ in range(len(empty_shape)-1):
                chunks.append(None)

            self._zdps.create_dataset(dps_key, shape=empty_shape,
                                      chunks=chunks, dtype=dtype)

        # if len(init_as._zgrp):
        # self._zcontainer.create_group('groups')
        for grp_key in init_as._zgrp.array_keys():
            empty_shape = list(init_as._zgrp[grp_key].shape)
            empty_shape[0] = 0
            dtype = init_as._zgrp[grp_key].dtype
            self._zgrp.create_dataset(grp_key, shape=empty_shape,
                                      chunks=(1000,), dtype=dtype)

        # if len(init_as._zdpg):
        # self._zcontainer.create_group('data_per_group')
        for grp_key in init_as._zdpg.group_keys():
            if len(init_as._zdpg[grp_key]):
                self._zdpg.create_group(grp_key)
            for dpg_key in init_as._zdpg[grp_key].array_keys():
                empty_shape = list(init_as._zdpg[grp_key][dpg_key].shape)
                empty_shape[0] = 0
                dtype = init_as._zdpg[grp_key][dpg_key].dtype
                self._zdpg[grp_key].create_dataset(dpg_key, shape=empty_shape,
                                                   chunks=None, dtype=dtype)

    def append(self, app_trx, delete_dpg=False, keep_first_dpg=True):
        """ """
        if not np.allclose(self.voxel_to_rasmm,
                           app_trx.voxel_to_rasmm) \
                or not np.array_equal(self.dimensions,
                                      app_trx.dimensions):
            raise ValueError('Wrong space attributes.')

        if delete_dpg and keep_first_dpg:
            print('1')
        if not _check_same_keys(self._zdpp.array_keys(),
                                app_trx._zdpp.array_keys()):
            print('A')
        if not _check_same_keys(self._zdps.array_keys(),
                                app_trx._zdps.array_keys()):
            print('B')
        if not (delete_dpg or keep_first_dpg) and \
                (len(self._zdpg) or len(self._zdpg)):
            print('C')

        self._zcontainer['positions'].append(app_trx._zcontainer['positions'])
        self._zcontainer['offsets'].append(app_trx._zcontainer['offsets'])

        for dpp_key in self._zdpp.array_keys():
            self._zdpp[dpp_key].append(app_trx._zdpp[dpp_key])
        for dps_key in self._zdps.array_keys():
            self._zdps[dps_key].append(app_trx._zdps[dps_key])
        for grp_key in self._zgrp.array_keys():
            self._zgrp[grp_key].append(app_trx._zgrp[grp_key] +
                                       self.nb_streamlines)

        if keep_first_dpg:
            for grp_key in self._zdpg.group_keys():
                for dpg_key in self._zdpg[grp_key].array_keys():
                    self._zdpg[grp_key][dpg_key].append(
                        app_trx._zdpg[grp_key][dpg_key])

        self.nb_points += app_trx.nb_points
        self.nb_streamlines += app_trx.nb_streamlines

    def tree(self):
        print(self._zcontainer.tree())

    @staticmethod
    def from_sft(sft, cast_position=np.float16):
        """ Generate a valid TrxFile from a StatefulTractogram """
        if not np.issubdtype(cast_position, np.floating):
            logging.warning('Casting as {}, considering using a floating point '
                            'dtype.'.format(cast_position))

        trx = TrxFile()
        trx.voxel_to_rasmm = sft.affine.tolist()
        trx.dimensions = sft.dimensions
        trx.nb_streamlines = len(sft.streamlines._lengths)
        trx.nb_points = len(sft.streamlines._data)

        old_space = deepcopy(sft.space)
        old_origin = deepcopy(sft.origin)
        sft.to_rasmm()
        sft.to_center()

        del trx._zcontainer['positions'], trx._zcontainer['offsets']
        trx._zcontainer.create_dataset('positions',
                                       data=sft.streamlines._data,
                                       chunks=(100000, None), dtype=np.float16)
        trx._zcontainer.create_dataset('offsets',
                                       data=sft.streamlines._offsets,
                                       chunks=(1000,), dtype=np.uint64)

        for dpp_key in sft.data_per_point.keys():
            trx._zdpp.create_dataset(dpp_key,
                                     data=sft.data_per_point[dpp_key]._data,
                                     chunks=(100000, None), dtype=np.float32)
        for dps_key in sft.data_per_streamline.keys():
            trx._zdps.create_dataset(dps_key,
                                     data=sft.data_per_streamline[dps_key],
                                     chunks=(10000, None), dtype=np.float32)
        sft.to_space(old_space)
        sft.to_origin(old_origin)

        return trx

    def to_sft(self):
        """ Convert a TrxFile to a valid StatefulTractogram """
        affine = self.voxel_to_rasmm
        dimensions = self.dimensions
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = ''.join(aff2axcodes(affine))
        space_attributes = (affine, dimensions, vox_sizes, vox_order)

        sft = StatefulTractogram(
            self.streamlines, space_attributes, Space.RASMM)
        #  data_per_point=self.data_per_point,
        #  data_per_streamline=self.data_per_streamline)

        return sft

    def _select(self, indices, keep_group=True, keep_dpg=False):
        """ Get a subset of items, always points to the same memmaps """
        indices = np.array(indices, np.uint32)
        if len(indices) and np.max(indices) > self.nb_streamlines - 1 or \
                np.min(indices) < 0:
            raise ValueError('Invalid indices.')

        if keep_dpg and not keep_group:
            raise ValueError('Cannot keep dpg if not keeping groups.')

        new_trx = TrxFile(init_as=self)
        if len(indices) == 0:
            return new_trx

        tmp_streamlines = self.streamlines
        arr_seq = ArraySequence()
        arr_seq._data = np.array(tmp_streamlines._data)
        arr_seq._offsets = np.array(tmp_streamlines._offsets)
        arr_seq._lengths = np.array(tmp_streamlines._lengths)

        chunk_offsets = list(range(0, len(arr_seq._offsets)+100000, 100000))
        for i in range(len(chunk_offsets)-1):
            chunk_arr_seq = arr_seq[chunk_offsets[i]:chunk_offsets[i+1]]

            tmp_indices = indices[indices >= chunk_offsets[i]]
            tmp_indices = tmp_indices[tmp_indices < chunk_offsets[i+1]]
            tmp_indices -= chunk_offsets[i]
            if len(tmp_indices) == 0:
                continue

            new_trx._zcontainer['offsets'].append(
                [len(new_trx._zcontainer['positions'])])
            new_trx._zcontainer['positions'].append(
                chunk_arr_seq[tmp_indices].get_data())
            new_trx._zcontainer['offsets'].append(
                np.cumsum(chunk_arr_seq[tmp_indices]._lengths[:-1]) +
                new_trx._zcontainer['offsets'][-1])

        new_trx.nb_streamlines = len(new_trx._zcontainer['offsets'])
        new_trx.nb_points = len(new_trx._zcontainer['positions'])

        # print(new_trx.streamlines)
        # for dpp_key in self._zdpp.array_keys():
        #     arr_seq = ArraySequence()
        #     arr_seq._data = np.array(self.data_per_point[dpp_key]._data)
        #     for i in range(len(chunk_offsets)-1):
        #         chunk_arr_seq = arr_seq[chunk_offsets[i]:chunk_offsets[i+1]]
        #         new_trx._zdpp[dpp_key] = chunk_arr_seq._data

        # for dps_key in self._zdps.array_keys():
        #     for i in range(len(chunk_offsets)-1):
        #         new_trx._zdps[dps_key] = chunk_arr_seq._data

        # arr_seq._offsets = np.array(tmp_streamlines._offsets)
        # arr_seq._lengths = np.array(tmp_streamlines._lengths)

        if keep_group:
            for grp_key in self._zgrp.array_keys():
                # print(indices, np.array(self._zgrp[grp_key]))
                new_group = np.intersect1d(indices, self._zgrp[grp_key],
                                           assume_unique=True)
                if len(new_group):
                    new_trx._zgrp[grp_key].append(new_group)
                else:
                    del new_trx._zgrp[grp_key]

        if keep_dpg:
            for grp_key in self._zdpg.group_keys():
                if grp_key in new_trx._zgrp:
                    for dpg_key in self._zdpg[grp_key].array_keys():
                        new_trx._zdpg[grp_key][dpg_key].append(
                            self._zdpg[grp_key][dpg_key])
        else:
            del new_trx._zcontainer['data_per_group']
            new_trx._zcontainer.create_group('data_per_group')

        new_trx.nb_streamlines = len(new_trx._zcontainer['offsets'])
        new_trx.nb_points = len(new_trx._zcontainer['positions'])
        new_trx.prune_grp_and_dpg()

        return new_trx

    def prune_grp_and_dpg(self):
        for grp_key in self._zgrp.array_keys():
            if self._zcontainer['groups'][grp_key].shape[0] == 0:
                del self._zgrp[grp_key]

        for grp_key in self._zdpg.group_keys():
            if grp_key not in self._zgrp:
                del self._zcontainer['data_per_group'][grp_key]
                continue
            for dpg_key in self._zdpg[grp_key].array_keys():
                if self._zdpg[grp_key][dpg_key].shape[0] == 0:
                    del self._zcontainer['data_per_group'][grp_key][dpg_key]

    @ property
    def data_per_streamline(self):
        dps_arr_dict = PerArrayDict()
        for dps_key in self._zdps.array_keys():
            dps_arr_dict[dps_key] = self._zdps[dps_key]

        return dps_arr_dict

    @ property
    def data_per_point(self):
        dpp_arr_seq_dict = PerArraySequenceDict()
        for dpp_key in self._zdpp.array_keys():
            arr_seq = ArraySequence()
            arr_seq._data = self._zdpp[dpp_key]
            arr_seq._offsets = self._zcontainer['offsets']
            arr_seq._lengths = compute_lengths(arr_seq._offsets,
                                               self.nb_points)
            dpp_arr_seq_dict[dpp_key] = arr_seq

        return dpp_arr_seq_dict

    @ property
    def streamlines(self):
        """ """
        streamlines = ArraySequence()
        streamlines._data = self._zcontainer['positions']
        streamlines._offsets = self._zcontainer['offsets']
        streamlines._lengths = compute_lengths(streamlines._offsets,
                                               self.nb_points)
        return streamlines

    @ property
    def voxel_to_rasmm(self):
        """ """
        return np.array(self._zcontainer.attrs['VOXEL_TO_RASMM'],
                        dtype=np.float32)

    @ voxel_to_rasmm.setter
    def voxel_to_rasmm(self, val):
        if isinstance(val, np.ndarray):
            val = val.astype(np.float32).tolist()
        self._zcontainer.attrs['VOXEL_TO_RASMM'] = val

    @ property
    def dimensions(self):
        """ """
        return np.array(self._zcontainer.attrs['DIMENSIONS'], dtype=np.uint16)

    @ dimensions.setter
    def dimensions(self, val):
        if isinstance(val, np.ndarray):
            val = val.astype(np.uint16).tolist()
        self._zcontainer.attrs['DIMENSIONS'] = val

    @ property
    def nb_streamlines(self):
        """ """
        return self._zcontainer.attrs['NB_STREAMLINES']

    @ nb_streamlines.setter
    def nb_streamlines(self, val):
        self._zcontainer.attrs['NB_STREAMLINES'] = int(val)

    @ property
    def nb_points(self):
        """ """
        return self._zcontainer.attrs['NB_POINTS']

    @ nb_points.setter
    def nb_points(self, val):
        self._zcontainer.attrs['NB_POINTS'] = int(val)

    @ property
    def _zdpp(self):
        """ """
        return self._zcontainer['data_per_point']

    @ property
    def _zdps(self):
        """ """
        return self._zcontainer['data_per_streamline']

    @ property
    def _zdpg(self):
        """ """
        return self._zcontainer['data_per_group']

    @ property
    def _zgrp(self):
        """ """
        return self._zcontainer['groups']

    def __str__(self):
        """ Generate the string for printing """
        affine = np.array(self.voxel_to_rasmm, dtype=np.float32)
        dimensions = np.array(self.dimensions, dtype=np.uint16)
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = ''.join(aff2axcodes(affine))

        text = 'VOXEL_TO_RASMM: \n{}'.format(
            np.array2string(affine,
                            formatter={'float_kind': lambda x: "%.6f" % x}))
        text += '\nDIMENSIONS: {}'.format(
            np.array2string(dimensions))
        text += '\nVOX_SIZES: {}'.format(
            np.array2string(vox_sizes,
                            formatter={'float_kind': lambda x: "%.2f" % x}))
        text += '\nVOX_ORDER: {}'.format(vox_order)

        text += '\nNB_STREAMLINES: {}'.format(self.nb_streamlines)
        text += '\nNB_POINTS: {}'.format(self.nb_points)

        text += '\n'+TreeViewer(self._zcontainer).__unicode__()

        return text
