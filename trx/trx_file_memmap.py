#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import json
import logging
import os
import shutil
import tempfile
from typing import Any, List, Tuple, Type, Union, Optional
import zipfile

import nibabel as nib
from nibabel.affines import voxel_sizes
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.trk import TrkFile
import numpy as np
from numpy.typing import ArrayLike, NDArray

from trx.utils import get_reference_info_wrapper


def _append_last_offsets(nib_offsets: NDArray, nb_vertices: int) -> NDArray:
    """Appends the last element of offsets from header information

    Keyword arguments:
        nib_offsets -- NDArray
            Array of offsets with the last element being the start of the last
            streamline (nibabel convention)
        nb_vertices -- int
            Total number of vertices in the streamlines
    Returns:
        Offsets -- NDArray (VTK convention)
    """
    def is_sorted(a): return np.all(a[:-1] <= a[1:])
    if not is_sorted(nib_offsets):
        raise ValueError('Offsets must be sorted values.')
    return np.append(nib_offsets, nb_vertices).astype(nib_offsets.dtype)


def _generate_filename_from_data(arr: ArrayLike, filename: str) -> str:
    """Determines the data type from array data and generates the appropriate
    filename

    Keyword arguments:
        arr -- a NumPy array (1-2D, otherwise ValueError raised)
        filename -- the original filename

    Returns:
        An updated filename
    """
    base, ext = os.path.splitext(filename)
    if ext:
        logging.warning("Will overwrite provided extension if needed.")

    dtype = arr.dtype
    dtype = "bit" if dtype == bool else dtype.name

    if arr.ndim == 1:
        new_filename = "{}.{}".format(base, dtype)
    elif arr.ndim == 2:
        dim = arr.shape[-1]
        if dim == 1:
            new_filename = "{}.{}".format(base, dtype)
        else:
            new_filename = "{}.{}.{}".format(base, arr.shape[-1], dtype)
    else:
        raise ValueError("Invalid dimensionality.")

    return new_filename


def _split_ext_with_dimensionality(filename: str) -> Tuple[str, int, str]:
    """Takes a filename and splits it into its components

    Keyword arguments:
        filename -- Input filename

    Returns:
        tuple of strings (basename, dimension, extension)

    """
    basename = os.path.basename(filename)
    split = basename.split(".")

    if len(split) != 2 and len(split) != 3:
        raise ValueError("Invalid filename.")
    basename = split[0]
    ext = ".{}".format(split[-1])
    dim = 1 if len(split) == 2 else split[1]

    _is_dtype_valid(ext)

    return basename, int(dim), ext


def _compute_lengths(offsets: NDArray) -> NDArray:
    """Compute lengths from offsets

    Keyword arguments:
        offsets -- An NDArray of offsets

    Returns:
        lengths -- An NDArray of lengths
    """
    if len(offsets) > 0:
        last_elem_pos = _dichotomic_search(offsets)
        lengths = np.ediff1d(offsets)
        if len(lengths) > last_elem_pos:
            lengths[last_elem_pos] = 0
    else:
        lengths = np.array([0])

    return lengths.astype(np.uint32)


def _is_dtype_valid(ext: str) -> bool:
    """Verifies that filename extension is a valid datatype

    Keyword arguments:
        ext -- filename extension

    Returns:
        boolean representing if provided datatype is valid
    """
    if ext.replace(".", "") == "bit":
        return True
    try:
        isinstance(np.dtype(ext.replace(".", "")), np.dtype)
        return True
    except TypeError:
        return False


def _dichotomic_search(
    x: NDArray, l_bound: Optional[int] = None, r_bound: Optional[int] = None
) -> int:
    """Find where data of a contiguous array is actually ending

    Keyword arguments:
        x -- NDArray of values
        l_bound -- lower bound index for search
        r_bound -- upper bound index for search
    Returns:
        index at which array value is 0 (if possible), otherwise returns -1"""
    if l_bound is None and r_bound is None:
        l_bound = 0
        r_bound = len(x) - 1

    if l_bound == r_bound:
        val = l_bound if x[l_bound] != 0 else -1
        return val

    mid_bound = (l_bound + r_bound + 1) // 2

    if x[mid_bound] == 0:
        return _dichotomic_search(x, l_bound, mid_bound - 1)
    else:
        return _dichotomic_search(x, mid_bound, r_bound)


def _create_memmap(
    filename: str,
    mode: str = "r",
    shape: Tuple = (1,),
    dtype: np.dtype = np.float32,
    offset: int = 0,
    order: str = "C",
) -> NDArray:
    """Wrapper to support empty array as memmaps

    Keyword arguments:
        filename -- filename of the file where the empty memmap should be created
        mode -- file open mode (see: np.memmap for options)
        shape -- shape of memmapped NDArray
        dtype -- datatype of memmapped NDArray
        offset -- offset of the data within the file
        order -- data representation on disk (C or Fortran)

    Returns:
        mmapped NDArray or a zero-filled Numpy array if array has a shape of 0 in the first dimension
    """
    if np.dtype(dtype) == bool:
        filename = filename.replace(".bool", ".bit")

    if shape[0]:
        return np.memmap(
            filename, mode=mode, offset=offset, shape=shape, dtype=dtype, order=order
        )
    else:
        if not os.path.isfile(filename):
            f = open(filename, "wb")
            f.close()
        return np.zeros(shape, dtype=dtype)


def load(input_obj: str, check_dpg: bool = True) -> Type["TrxFile"]:
    """Load a TrxFile (compressed or not)

    Keyword arguments:
    input_obj -- A directory name or filepath to the trx data
    check_dpg -- Boolean denoting if group metadata should be checked

    Returns:
        TrxFile object representing the read data
    """
    # TODO Check if 0 streamlines, if yes then 0 vertices is expected (vice-versa)
    # TODO 4x4 affine matrices should contains values (no all-zeros)
    # TODO 3x1 dimensions array should contains values at each position (int)
    if os.path.isfile(input_obj):
        was_compressed = False
        with zipfile.ZipFile(input_obj, "r") as zf:
            for info in zf.infolist():
                if info.compress_type != 0:
                    was_compressed = True
                    break
        if was_compressed:
            with zipfile.ZipFile(input_obj, "r") as zf:
                tmpdir = tempfile.TemporaryDirectory()
                zf.extractall(tmpdir.name)
                trx = load_from_directory(tmpdir.name)
                trx._uncompressed_folder_handle = tmpdir
                logging.info(
                    "File was compressed, call the close() " "function before exiting."
                )
        else:
            trx = load_from_zip(input_obj)
    elif os.path.isdir(input_obj):
        trx = load_from_directory(input_obj)
    else:
        raise ValueError("File/Folder does not exist")

    # Example of robust check for metadata
    if check_dpg:
        for dpg in trx.data_per_group.keys():
            if dpg not in trx.groups.keys():
                raise ValueError(
                    "An undeclared group ({}) has " "data_per_group.".format(
                        dpg)
                )

    return trx


def load_from_zip(filename: str) -> Type["TrxFile"]:
    """Load a TrxFile from a single zipfile. Note: does not work with compressed zipfiles

    Keyword arguments:
    filename -- path of the zipped TrxFile

    Returns:
        TrxFile representing the read data
    """
    with zipfile.ZipFile(filename, mode="r") as zf:
        with zf.open("header.json") as zf_header:
            header = json.load(zf_header)
            header["VOXEL_TO_RASMM"] = np.reshape(
                header["VOXEL_TO_RASMM"], (4, 4)
            ).astype(np.float32)
            header["DIMENSIONS"] = np.array(
                header["DIMENSIONS"], dtype=np.uint16)

        files_pointer_size = {}
        for zip_info in zf.filelist:
            elem_filename = zip_info.filename
            _, ext = os.path.splitext(elem_filename)
            if ext == ".json" or zip_info.is_dir():
                continue

            if not _is_dtype_valid(ext):
                continue
                raise ValueError(
                    "The dtype {} is not supported".format(elem_filename))

            if ext == ".bit":
                ext = ".bool"

            mem_adress = zip_info.header_offset + len(zip_info.FileHeader())
            dtype_size = np.dtype(ext[1:]).itemsize
            size = zip_info.file_size / dtype_size

            if len(zip_info.extra):
                mem_adress += 4

            if size.is_integer():
                files_pointer_size[elem_filename] = mem_adress, int(size)
            else:
                raise ValueError("Wrong size or datatype")

    return TrxFile._create_trx_from_pointer(
        header, files_pointer_size, root_zip=filename
    )


def load_from_directory(directory: str) -> Type["TrxFile"]:
    """Load a TrxFile from a folder containing memmaps

    Keyword arguments:
    filename -- path of the zipped TrxFile

    Returns:
        TrxFile representing the read data
    """

    directory = os.path.abspath(directory)
    with open(os.path.join(directory, "header.json")) as header:
        header = json.load(header)
        header["VOXEL_TO_RASMM"] = np.reshape(header["VOXEL_TO_RASMM"], (4, 4)).astype(
            np.float32
        )
        header["DIMENSIONS"] = np.array(header["DIMENSIONS"], dtype=np.uint16)
    files_pointer_size = {}
    for root, dirs, files in os.walk(directory):
        for name in files:
            elem_filename = os.path.join(root, name)
            _, ext = os.path.splitext(elem_filename)
            if ext == ".json":
                continue

            if not _is_dtype_valid(ext):
                raise ValueError(
                    "The dtype of {} is not supported".format(elem_filename)
                )

            if ext == ".bit":
                ext = ".bool"

            dtype_size = np.dtype(ext[1:]).itemsize
            size = os.path.getsize(elem_filename) / dtype_size

            if size.is_integer():
                files_pointer_size[elem_filename] = 0, int(size)
            elif os.path.getsize(elem_filename) == 1:
                files_pointer_size[elem_filename] = 0, 0
            else:
                raise ValueError("Wrong size or datatype")

    return TrxFile._create_trx_from_pointer(header, files_pointer_size, root=directory)


def concatenate(
    trx_list: List["TrxFile"],
    delete_dpv: bool = False,
    delete_dps: bool = False,
    delete_groups: bool = False,
    check_space_attributes: bool = True,
    preallocation: bool = False,
) -> "TrxFile":
    """Concatenate multiple TrxFile together, support preallocation

    Keyword arguments:
        trx_list -- A list containing TrxFiles to concatenate
        delete_dpv -- Delete dpv keys that do not exist in all the provided TrxFiles
        delete_dps -- Delete dps keys that do not exist in all the provided TrxFile
        delete_groups -- Delete all the groups that currently exist in the TrxFiles
        check_space_attributes -- Verify that dimensions and size of data are similar between all the TrxFiles
        preallocation -- Preallocated TrxFile has already been generated and is the first element in trx_list
                         (Note: delete_groups must be set to True as well)

    Returns:
        TrxFile representing the concatenated data

    """
    trx_list = [
        curr_trx for curr_trx in trx_list if curr_trx.header["NB_STREAMLINES"] > 0
    ]
    if len(trx_list) == 0:
        logging.warning("Inputs of concatenation were empty.")
        return TrxFile()

    ref_trx = trx_list[0]
    all_dps = []
    all_dpv = []
    for curr_trx in trx_list:
        all_dps.extend(list(curr_trx.data_per_streamline.keys()))
        all_dpv.extend(list(curr_trx.data_per_vertex.keys()))
    all_dps, all_dpv = set(all_dps), set(all_dpv)

    if check_space_attributes:
        for curr_trx in trx_list[1:]:
            if not np.allclose(
                ref_trx.header["VOXEL_TO_RASMM"], curr_trx.header["VOXEL_TO_RASMM"]
            ) or not np.array_equal(
                ref_trx.header["DIMENSIONS"], curr_trx.header["DIMENSIONS"]
            ):
                raise ValueError("Wrong space attributes.")

    if preallocation and not delete_groups:
        raise ValueError(
            "Groups are variables, cannot be handled with " "preallocation"
        )

    # Verifying the validity of fixed-size arrays, coherence between inputs
    for curr_trx in trx_list:
        for key in all_dpv:
            if key not in ref_trx.data_per_vertex.keys() or key not in curr_trx.data_per_vertex.keys():
                if not delete_dpv:
                    logging.debug(
                        "{} dpv key does not exist in all TrxFile.".format(key)
                    )
                    raise ValueError(
                        "TrxFile must be sharing identical dpv " "keys.")
            elif (
                ref_trx.data_per_vertex[key]._data.dtype
                != curr_trx.data_per_vertex[key]._data.dtype
            ):
                logging.debug(
                    "{} dpv key is not declared with the same dtype "
                    "in all TrxFile.".format(key)
                )
                raise ValueError("Shared dpv key, has different dtype.")

    for curr_trx in trx_list:
        for key in all_dps:
            if key not in ref_trx.data_per_streamline.keys() or key not in curr_trx.data_per_streamline.keys():
                if not delete_dps:
                    logging.debug(
                        "{} dps key does not exist in all " "TrxFile.".format(
                            key)
                    )
                    raise ValueError(
                        "TrxFile must be sharing identical dps " "keys.")
            elif (
                ref_trx.data_per_streamline[key].dtype
                != curr_trx.data_per_streamline[key].dtype
            ):
                logging.debug(
                    "{} dps key is not declared with the same dtype "
                    "in all TrxFile.".format(key)
                )
                raise ValueError("Shared dps key, has different dtype.")

    all_groups_len = {}
    all_groups_dtype = {}
    # Variable-size arrays do not have to exist in all TrxFile
    if not delete_groups:
        for trx_1 in trx_list:
            for group_key in trx_1.groups.keys():
                # Concatenating groups together
                if group_key in all_groups_len:
                    all_groups_len[group_key] += len(trx_1.groups[group_key])
                else:
                    all_groups_len[group_key] = len(trx_1.groups[group_key])
                if (
                    group_key in all_groups_dtype
                    and trx_1.groups[group_key].dtype != all_groups_dtype[group_key]
                ):
                    raise ValueError("Shared group key, has different dtype.")
                else:
                    all_groups_dtype[group_key] = trx_1.groups[group_key].dtype

    # Once the checks are done, actually concatenate
    to_concat_list = trx_list[1:] if preallocation else trx_list
    if not preallocation:
        nb_vertices = 0
        nb_streamlines = 0
        for curr_trx in to_concat_list:
            curr_strs_len, curr_pts_len = curr_trx._get_real_len()
            nb_streamlines += curr_strs_len
            nb_vertices += curr_pts_len

        new_trx = TrxFile(
            nb_vertices=nb_vertices, nb_streamlines=nb_streamlines, init_as=ref_trx
        )
        if delete_dps:
            new_trx.data_per_streamline = {}
        if delete_dpv:
            new_trx.data_per_vertex = {}
        if delete_groups:
            new_trx.groups = {}

        tmp_dir = new_trx._uncompressed_folder_handle.name

        # When memory is allocated on the spot, groups and data_per_group can
        # be concatenated together
        for group_key in all_groups_len.keys():
            if not os.path.isdir(os.path.join(tmp_dir, "groups/")):
                os.mkdir(os.path.join(tmp_dir, "groups/"))
            dtype = all_groups_dtype[group_key]
            group_filename = os.path.join(
                tmp_dir, "groups/" "{}.{}".format(group_key, dtype.name)
            )
            group_len = all_groups_len[group_key]
            new_trx.groups[group_key] = _create_memmap(
                group_filename, mode="w+", shape=(group_len,), dtype=dtype
            )
            if delete_groups:
                continue
            pos = 0
            count = 0
            for curr_trx in trx_list:
                curr_len = len(curr_trx.groups[group_key])
                new_trx.groups[group_key][pos: pos +
                                          curr_len] = curr_trx.groups[group_key] + count
                pos += curr_len
                count += curr_trx.header["NB_STREAMLINES"]

        strs_end, pts_end = 0, 0
    else:
        new_trx = ref_trx
        strs_end, pts_end = new_trx._get_real_len()

    for curr_trx in to_concat_list:
        # Copy the TrxFile fixed-size info (the right chunk)
        strs_end, pts_end = new_trx._copy_fixed_arrays_from(
            curr_trx, strs_start=strs_end, pts_start=pts_end
        )
    return new_trx


def save(
    trx: "TrxFile", filename: str, compression_standard: Any = zipfile.ZIP_STORED
) -> None:
    """Save a TrxFile (compressed or not)

    Keyword arguments:
        trx -- The TrxFile to save
        filename -- The path to save the TrxFile to
        compression_standard -- The compression standard to use, as defined by the ZipFile library
    """
    if os.path.splitext(filename)[1] and not os.path.splitext(filename)[1] in [
        ".zip",
        ".trx",
    ]:
        raise ValueError("Unsupported extension.")

    copy_trx = trx.deepcopy()
    copy_trx.resize()

    tmp_dir_name = copy_trx._uncompressed_folder_handle.name
    if os.path.splitext(filename)[1] and os.path.splitext(filename)[1] in [
        ".zip",
        ".trx",
    ]:
        zip_from_folder(tmp_dir_name, filename, compression_standard)
    else:
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        shutil.copytree(tmp_dir_name, filename)
    copy_trx.close()


def zip_from_folder(
    directory: str, filename: str, compression_standard: Any = zipfile.ZIP_STORED
) -> None:
    """Utils function to zip on-disk memmaps

    Keyword arguments
        directory -- The path to the on-disk memmap
        filename -- The path where the zip file should be created
        compression_standard -- The compression standard to use, as defined by the ZipFile library

    """
    with zipfile.ZipFile(filename, mode="w", compression=compression_standard) as zf:
        for root, dirs, files in os.walk(directory):
            for name in files:
                tmp_filename = os.path.join(root, name)
                zf.write(tmp_filename, tmp_filename.replace(
                    directory + "/", ""))


class TrxFile:
    """Core class of the TrxFile"""

    header: dict
    streamlines: Type[ArraySequence]
    groups: dict
    data_per_streamline: dict
    data_per_vertex: dict
    data_per_group: dict

    def __init__(
        self,
        nb_vertices: Optional[int] = None,
        nb_streamlines: Optional[int] = None,
        init_as: Optional[Type["TrxFile"]] = None,
        reference: Union[
            str,
            dict,
            Type[Nifti1Image],
            Type[TrkFile],
            Type[Nifti1Header],
            None,
        ] = None,
    ) -> None:
        """Initialize an empty TrxFile, support preallocation

        Keyword Arguments:
            nb_vertices -- The number of vertices to use in the new TrxFile
            nb_streamlines -- The number of streamlines in the new TrxFile
            init_as -- A TrxFile to use as reference
            reference -- A Nifti or Trk file/obj to use as reference
        """
        if init_as is not None:
            affine = init_as.header["VOXEL_TO_RASMM"]
            dimensions = init_as.header["DIMENSIONS"]
        elif reference is not None:
            affine, dimensions, _, _ = get_reference_info_wrapper(reference)
        else:
            logging.debug(
                "No reference provided, using blank space "
                "attributes, please update them later."
            )
            affine = np.eye(4).astype(np.float32)
            dimensions = np.array([1, 1, 1], dtype=np.uint16)

        if nb_vertices is None and nb_streamlines is None:
            if init_as is not None:
                raise ValueError(
                    "Cant use init_as without declaring "
                    "nb_vertices AND nb_streamlines"
                )
            logging.debug("Intializing empty TrxFile.")
            self.header = {}
            # Using the new format default type
            tmp_strs = ArraySequence()
            tmp_strs._data = tmp_strs._data.astype(np.float32)
            tmp_strs._offsets = tmp_strs._offsets.astype(np.uint32)
            tmp_strs._lengths = tmp_strs._lengths.astype(np.uint32)
            self.streamlines = tmp_strs
            self.groups = {}
            self.data_per_streamline = {}
            self.data_per_vertex = {}
            self.data_per_group = {}
            self._uncompressed_folder_handle = None

            nb_vertices = 0
            nb_streamlines = 0

        elif nb_vertices is not None and nb_streamlines is not None:
            logging.debug(
                "Preallocating TrxFile with size {} streamlines"
                "and {} vertices.".format(nb_streamlines, nb_vertices)
            )
            trx = self._initialize_empty_trx(
                nb_streamlines, nb_vertices, init_as=init_as
            )
            self.__dict__ = trx.__dict__
        else:
            raise ValueError(
                "You must declare both nb_vertices AND " "NB_STREAMLINES")

        self.header["VOXEL_TO_RASMM"] = affine
        self.header["DIMENSIONS"] = dimensions
        self.header["NB_VERTICES"] = nb_vertices
        self.header["NB_STREAMLINES"] = nb_streamlines
        self._copy_safe = True

    def __str__(self) -> str:
        """Generate the string for printing"""
        affine = np.array(self.header["VOXEL_TO_RASMM"], dtype=np.float32)
        dimensions = np.array(self.header["DIMENSIONS"], dtype=np.uint16)
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = "".join(aff2axcodes(affine))

        text = "VOXEL_TO_RASMM: \n{}".format(
            np.array2string(affine, formatter={
                            "float_kind": lambda x: "%.6f" % x})
        )
        text += "\nDIMENSIONS: {}".format(np.array2string(dimensions))
        text += "\nVOX_SIZES: {}".format(
            np.array2string(vox_sizes, formatter={
                            "float_kind": lambda x: "%.2f" % x})
        )
        text += "\nVOX_ORDER: {}".format(vox_order)

        strs_size = self.header["NB_STREAMLINES"]
        pts_size = self.header["NB_VERTICES"]
        strs_len, pts_len = self._get_real_len()

        if strs_size != strs_len or pts_size != pts_len:
            text += "\nstreamline_size: {}".format(strs_size)
            text += "\nvertex_size: {}".format(pts_size)

        text += "\nstreamline_count: {}".format(strs_len)
        text += "\nvertex_count: {}".format(pts_len)
        text += "\ndata_per_vertex keys: {}".format(
            list(self.data_per_vertex.keys()))
        text += "\ndata_per_streamline keys: {}".format(
            list(self.data_per_streamline.keys())
        )

        text += "\ngroups keys: {}".format(list(self.groups.keys()))
        for group_key in self.groups.keys():
            if group_key in self.data_per_group:
                text += "\ndata_per_groups ({}) keys: {}".format(
                    group_key, list(self.data_per_group[group_key].keys())
                )

        text += "\ncopy_safe: {}".format(self._copy_safe)

        return text

    def __len__(self) -> int:
        """Define the length of the object"""
        return len(self.streamlines)

    def __getitem__(self, key) -> Any:
        """Slice all data in a consistent way"""
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            key = [key]
        elif isinstance(key, slice):
            key = [ii for ii in range(*key.indices(len(self)))]

        return self.select(key, keep_group=False)

    def __deepcopy__(self) -> Type["TrxFile"]:
        return self.deepcopy()

    def deepcopy(self) -> Type["TrxFile"]:
        """Create a deepcopy of the TrxFile

        Returns
            A deepcopied TrxFile of the current TrxFile
        """
        tmp_dir = tempfile.TemporaryDirectory()
        out_json = open(os.path.join(tmp_dir.name, "header.json"), "w")
        tmp_header = deepcopy(self.header)

        if not isinstance(tmp_header["VOXEL_TO_RASMM"], list):
            tmp_header["VOXEL_TO_RASMM"] = tmp_header["VOXEL_TO_RASMM"].tolist()
        if not isinstance(tmp_header["DIMENSIONS"], list):
            tmp_header["DIMENSIONS"] = tmp_header["DIMENSIONS"].tolist()

        # tofile() alway write in C-order
        if not self._copy_safe:
            to_dump = self.streamlines.copy()._data
            tmp_header["NB_STREAMLINES"] = len(self.streamlines)
            tmp_header["NB_VERTICES"] = len(to_dump)
        else:
            to_dump = self.streamlines._data
        json.dump(tmp_header, out_json)
        out_json.close()

        positions_filename = _generate_filename_from_data(
            to_dump, os.path.join(tmp_dir.name, "positions")
        )
        to_dump.tofile(positions_filename)

        if not self._copy_safe:
            to_dump = _append_last_offsets(self.streamlines.copy()._offsets,
                                           self.header["NB_VERTICES"])
        else:
            to_dump = _append_last_offsets(self.streamlines._offsets,
                                           self.header["NB_VERTICES"])
        offsets_filename = _generate_filename_from_data(
            self.streamlines._offsets, os.path.join(tmp_dir.name, "offsets")
        )
        to_dump.tofile(offsets_filename)

        if len(self.data_per_vertex.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir.name, "dpv/"))
        for dpv_key in self.data_per_vertex.keys():
            if not self._copy_safe:
                to_dump = self.data_per_vertex[dpv_key].copy()._data
            else:
                to_dump = self.data_per_vertex[dpv_key]._data

            dpv_filename = _generate_filename_from_data(
                to_dump, os.path.join(tmp_dir.name, "dpv/", dpv_key)
            )
            to_dump.tofile(dpv_filename)

        if len(self.data_per_streamline.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir.name, "dps/"))
        for dps_key in self.data_per_streamline.keys():
            to_dump = self.data_per_streamline[dps_key]
            dps_filename = _generate_filename_from_data(
                to_dump, os.path.join(tmp_dir.name, "dps/", dps_key)
            )
            to_dump.tofile(dps_filename)

        if len(self.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir.name, "groups/"))
        for group_key in self.groups.keys():
            to_dump = self.groups[group_key]
            group_filename = _generate_filename_from_data(
                to_dump, os.path.join(tmp_dir.name, "groups/", group_key)
            )
            to_dump.tofile(group_filename)

            if group_key not in self.data_per_group:
                continue
            for dpg_key in self.data_per_group[group_key].keys():
                # Creates 'dpg/' only if required
                if not os.path.isdir(os.path.join(tmp_dir.name, "dpg/")):
                    os.mkdir(os.path.join(tmp_dir.name, "dpg/"))
                if not os.path.isdir(os.path.join(tmp_dir.name, "dpg/", group_key)):
                    os.mkdir(os.path.join(tmp_dir.name, "dpg/", group_key))
                to_dump = self.data_per_group[group_key][dpg_key]
                dpg_filename = _generate_filename_from_data(
                    to_dump, os.path.join(
                        tmp_dir.name, "dpg/", group_key, dpg_key)
                )
                to_dump.tofile(dpg_filename)

        copy_trx = load_from_directory(tmp_dir.name)
        copy_trx._uncompressed_folder_handle = tmp_dir

        return copy_trx

    def _get_real_len(self) -> Tuple[int, int]:
        """Get the real size of data (ignoring zeros of preallocation)

        Returns
            A tuple representing the index of the last streamline and the total length of all the streamlines
        """
        if len(self.streamlines._lengths) == 0:
            return 0, 0

        last_elem_pos = _dichotomic_search(self.streamlines._lengths)
        if last_elem_pos != -1:
            strs_end = int(last_elem_pos + 1)
            pts_end = int(np.sum(self.streamlines._lengths[0:strs_end]))
            return strs_end, pts_end

        return 0, 0

    def _copy_fixed_arrays_from(
        self,
        trx: Type["TrxFile"],
        strs_start: int = 0,
        pts_start: int = 0,
        nb_strs_to_copy: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Fill a TrxFile using another and start indexes (preallocation)

        Keyword arguments:
            trx -- TrxFile to copy data from
            strs_start -- The start index of the streamline
            pts_start -- The start index of the point
            nb_strs_to_copy -- The number of streamlines to copy. If not set will copy all

        Returns
            A tuple representing the end of the copied streamlines and end of copied points
        """
        if nb_strs_to_copy is None:
            curr_strs_len, curr_pts_len = trx._get_real_len()
        else:
            curr_strs_len = int(nb_strs_to_copy)
            curr_pts_len = np.sum(trx.streamlines._lengths[0:curr_strs_len])
            curr_pts_len = int(curr_pts_len)

        strs_end = strs_start + curr_strs_len
        pts_end = pts_start + curr_pts_len

        if curr_pts_len == 0:
            return strs_start, pts_start

        # Mandatory arrays
        self.streamlines._data[pts_start:pts_end] = trx.streamlines._data[
            0:curr_pts_len
        ]
        self.streamlines._offsets[strs_start:strs_end] = (
            trx.streamlines._offsets[0:curr_strs_len] + pts_start
        )
        self.streamlines._lengths[strs_start:strs_end] = trx.streamlines._lengths[
            0:curr_strs_len
        ]

        # Optional fixed-sized arrays
        for dpv_key in self.data_per_vertex.keys():
            self.data_per_vertex[dpv_key]._data[
                pts_start:pts_end
            ] = trx.data_per_vertex[dpv_key]._data[0:curr_pts_len]
            self.data_per_vertex[dpv_key]._offsets = self.streamlines._offsets
            self.data_per_vertex[dpv_key]._lengths = self.streamlines._lengths

        for dps_key in self.data_per_streamline.keys():
            self.data_per_streamline[dps_key][
                strs_start:strs_end
            ] = trx.data_per_streamline[dps_key][0:curr_strs_len]

        return strs_end, pts_end

    @staticmethod
    def _initialize_empty_trx(
        nb_streamlines: int, nb_vertices: int, init_as: Optional[Type["TrxFile"]] = None
    ) -> Type["TrxFile"]:
        """Create on-disk memmaps of a certain size (preallocation)

        Keyword arguments:
            nb_streamlines -- The number of streamlines that the empty TrxFile will be initialized with
            nb_vertices -- The number of vertices that the empty TrxFile will be initialized with
            init_as -- A TrxFile to initialize the empty TrxFile with

        Returns:
            An empty TrxFile preallocated with a certain size
        """
        trx = TrxFile()
        tmp_dir = tempfile.TemporaryDirectory()
        logging.info("Temporary folder for memmaps: {}".format(tmp_dir.name))

        trx.header["NB_VERTICES"] = nb_vertices
        trx.header["NB_STREAMLINES"] = nb_streamlines

        if init_as is not None:
            trx.header["VOXEL_TO_RASMM"] = init_as.header["VOXEL_TO_RASMM"]
            trx.header["DIMENSIONS"] = init_as.header["DIMENSIONS"]
            positions_dtype = init_as.streamlines._data.dtype
            offsets_dtype = init_as.streamlines._offsets.dtype
            lengths_dtype = init_as.streamlines._lengths.dtype
        else:
            positions_dtype = np.dtype(np.float16)
            offsets_dtype = np.dtype(np.uint32)
            lengths_dtype = np.dtype(np.uint32)

        logging.debug(
            "Initializing positions with dtype:    {}".format(
                positions_dtype.name)
        )
        logging.debug(
            "Initializing offsets with dtype: {}".format(offsets_dtype.name))
        logging.debug(
            "Initializing lengths with dtype: {}".format(lengths_dtype.name))

        # A TrxFile without init_as only contain the essential arrays
        positions_filename = os.path.join(
            tmp_dir.name, "positions.3.{}".format(positions_dtype.name)
        )
        trx.streamlines._data = _create_memmap(
            positions_filename, mode="w+", shape=(nb_vertices, 3), dtype=positions_dtype
        )

        offsets_filename = os.path.join(
            tmp_dir.name, "offsets.{}".format(offsets_dtype.name)
        )
        trx.streamlines._offsets = _create_memmap(
            offsets_filename, mode="w+", shape=(nb_streamlines,), dtype=offsets_dtype
        )
        trx.streamlines._lengths = np.zeros(
            shape=(nb_streamlines,), dtype=lengths_dtype
        )

        # Only the structure of fixed-size arrays is copied
        if init_as is not None:
            if len(init_as.data_per_vertex.keys()) > 0:
                os.mkdir(os.path.join(tmp_dir.name, "dpv/"))
            if len(init_as.data_per_streamline.keys()) > 0:
                os.mkdir(os.path.join(tmp_dir.name, "dps/"))

            for dpv_key in init_as.data_per_vertex.keys():
                dtype = init_as.data_per_vertex[dpv_key]._data.dtype
                tmp_as = init_as.data_per_vertex[dpv_key]._data
                if tmp_as.ndim == 1:
                    dpv_filename = os.path.join(
                        tmp_dir.name, "dpv/" "{}.{}".format(
                            dpv_key, dtype.name)
                    )
                    shape = (nb_vertices, 1)
                elif tmp_as.ndim == 2:
                    dim = tmp_as.shape[-1]
                    shape = (nb_vertices, dim)
                    dpv_filename = os.path.join(
                        tmp_dir.name, "dpv/" "{}.{}.{}".format(
                            dpv_key, dim, dtype.name)
                    )
                else:
                    raise ValueError("Invalid dimensionality.")

                logging.debug(
                    "Initializing {} (dpv) with dtype: "
                    "{}".format(dpv_key, dtype.name)
                )
                trx.data_per_vertex[dpv_key] = ArraySequence()
                trx.data_per_vertex[dpv_key]._data = _create_memmap(
                    dpv_filename, mode="w+", shape=shape, dtype=dtype
                )
                trx.data_per_vertex[dpv_key]._offsets = trx.streamlines._offsets
                trx.data_per_vertex[dpv_key]._lengths = trx.streamlines._lengths

            for dps_key in init_as.data_per_streamline.keys():
                dtype = init_as.data_per_streamline[dps_key].dtype
                tmp_as = init_as.data_per_streamline[dps_key]
                if tmp_as.ndim == 1:
                    dps_filename = os.path.join(
                        tmp_dir.name, "dps/" "{}.{}".format(
                            dps_key, dtype.name)
                    )
                    shape = (nb_streamlines,)
                elif tmp_as.ndim == 2:
                    dim = tmp_as.shape[-1]
                    shape = (nb_streamlines, dim)
                    dps_filename = os.path.join(
                        tmp_dir.name, "dps/" "{}.{}.{}".format(
                            dps_key, dim, dtype.name)
                    )
                else:
                    raise ValueError("Invalid dimensionality.")

                logging.debug(
                    "Initializing {} (dps) with and dtype: "
                    "{}".format(dps_key, dtype.name)
                )
                trx.data_per_streamline[dps_key] = _create_memmap(
                    dps_filename, mode="w+", shape=shape, dtype=dtype
                )

        trx._uncompressed_folder_handle = tmp_dir

        return trx

    def _create_trx_from_pointer(
        header: dict,
        dict_pointer_size: dict,
        root_zip: Optional[str] = None,
        root: Optional[str] = None,
    ) -> Type["TrxFile"]:
        """After reading the structure of a zip/folder, create a TrxFile

        Keyword arguments:
            header -- A TrxFile header dictionary which will be used for the new TrxFile
            dict_pointer_size -- A dictionary containing the filenames of all the files within the TrxFile disk file/folder
            root_zip -- The path of the ZipFile pointer
            root -- The dirname of the ZipFile pointer

        Returns:
            A TrxFile constructer from the pointer provided
        """
        # TODO support empty positions, using optional tag?
        trx = TrxFile()
        trx.header = header
        positions, offsets = None, None
        for elem_filename in dict_pointer_size.keys():
            if root_zip:
                filename = root_zip
            else:
                filename = elem_filename

            folder = os.path.dirname(elem_filename)
            base, dim, ext = _split_ext_with_dimensionality(elem_filename)
            if ext == ".bit":
                ext = ".bool"
            mem_adress, size = dict_pointer_size[elem_filename]

            if root is not None and folder.startswith(root.rstrip("/")):
                folder = folder.replace(root, "").lstrip("/")

            # Parse/walk the directory tree
            if base == "positions" and folder == "":
                if size != trx.header["NB_VERTICES"] * 3 or dim != 3:
                    raise ValueError("Wrong data size/dimensionality.")
                positions = _create_memmap(
                    filename,
                    mode="r+",
                    offset=mem_adress,
                    shape=(trx.header["NB_VERTICES"], 3),
                    dtype=ext[1:],
                )
            elif base == "offsets" and folder == "":
                if size != trx.header["NB_STREAMLINES"]+1 or dim != 1:
                    raise ValueError("Wrong offsets size/dimensionality.")
                offsets = _create_memmap(
                    filename,
                    mode="r+",
                    offset=mem_adress,
                    shape=(trx.header["NB_STREAMLINES"]+1,),
                    dtype=ext[1:],
                )
                lengths = _compute_lengths(offsets)
            elif folder == "dps":
                nb_scalar = size / trx.header["NB_STREAMLINES"]
                if not nb_scalar.is_integer() or nb_scalar != dim:
                    raise ValueError("Wrong dps size/dimensionality.")
                else:
                    shape = (trx.header["NB_STREAMLINES"], int(nb_scalar))

                trx.data_per_streamline[base] = _create_memmap(
                    filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
                )
            elif folder == "dpv":
                nb_scalar = size / trx.header["NB_VERTICES"]
                if not nb_scalar.is_integer() or nb_scalar != dim:
                    raise ValueError("Wrong dpv size/dimensionality.")
                else:
                    shape = (trx.header["NB_VERTICES"], int(nb_scalar))

                trx.data_per_vertex[base] = _create_memmap(
                    filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
                )
            elif folder.startswith("dpg"):
                if int(size) != dim:
                    raise ValueError("Wrong dpg size/dimensionality.")
                else:
                    shape = (1, int(size))

                # Handle the two-layers architecture
                data_name = os.path.basename(base)
                sub_folder = os.path.basename(folder)
                if sub_folder not in trx.data_per_group:
                    trx.data_per_group[sub_folder] = {}
                trx.data_per_group[sub_folder][data_name] = _create_memmap(
                    filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
                )
            elif folder == "groups":
                # Groups are simply indices, nothing else
                # TODO Crash if not uint?
                if dim != 1:
                    raise ValueError("Wrong group dimensionality.")
                else:
                    shape = (int(size),)
                trx.groups[base] = _create_memmap(
                    filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
                )
            else:
                logging.error(
                    "{} is not part of a valid structure.".format(
                        elem_filename)
                )

        # All essential array must be declared
        if positions is not None and offsets is not None:
            trx.streamlines._data = positions
            trx.streamlines._offsets = offsets[:-1]
            trx.streamlines._lengths = lengths
        else:
            raise ValueError("Missing essential data.")

        for dpv_key in trx.data_per_vertex:
            tmp = trx.data_per_vertex[dpv_key]
            trx.data_per_vertex[dpv_key] = ArraySequence()
            trx.data_per_vertex[dpv_key]._data = tmp
            trx.data_per_vertex[dpv_key]._offsets = offsets[:-1]
            trx.data_per_vertex[dpv_key]._lengths = lengths
        return trx

    def resize(
        self,
        nb_streamlines: Optional[int] = None,
        nb_vertices: Optional[int] = None,
        delete_dpg: bool = False,
    ) -> None:
        """Remove the ununsed portion of preallocated memmaps

        Keyword arguments:
            nb_streamlines -- The number of streamlines to keep
            nb_vertices -- The number of vertices to keep
            delete_dpg -- Remove data_per_group when resizing
        """
        if not self._copy_safe:
            raise ValueError("Cannot resize a sliced datasets.")

        strs_end, pts_end = self._get_real_len()

        if nb_streamlines is not None and nb_streamlines < strs_end:
            strs_end = nb_streamlines
            logging.info(
                "Resizing (down) memmaps, less streamlines than it "
                "actually contains."
            )

        if nb_vertices is None:
            pts_end = int(np.sum(self.streamlines._lengths[0:nb_streamlines]))
            nb_vertices = pts_end
        elif nb_vertices < pts_end:
            # Resizing vertices only is too dangerous, not allowed
            logging.warning("Cannot resize (down) vertices for consistency.")
            return

        if nb_streamlines is None:
            nb_streamlines = strs_end

        if (
            nb_streamlines == self.header["NB_STREAMLINES"]
            and nb_vertices == self.header["NB_VERTICES"]
        ):
            logging.debug("TrxFile of the right size, no resizing.")
            return

        trx = self._initialize_empty_trx(
            nb_streamlines, nb_vertices, init_as=self)

        logging.info(
            "Resizing streamlines from size {} to {}".format(
                len(self.streamlines), nb_streamlines
            )
        )
        logging.info(
            "Resizing vertices from size {} to {}".format(
                len(self.streamlines._data), nb_vertices
            )
        )

        # Copy the fixed-sized info from the original TrxFile to the new
        # (resized) one.
        if nb_streamlines < self.header["NB_STREAMLINES"]:
            trx._copy_fixed_arrays_from(self, nb_strs_to_copy=nb_streamlines)
        else:
            trx._copy_fixed_arrays_from(self)

        tmp_dir = trx._uncompressed_folder_handle.name
        if len(self.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, "groups/"))

        for group_key in self.groups.keys():
            group_dtype = self.groups[group_key].dtype
            group_name = os.path.join(
                tmp_dir, "groups/", "{}.{}".format(group_key, group_dtype.name)
            )
            ori_len = len(self.groups[group_key])

            # Remove groups indices if resizing down
            tmp = self.groups[group_key][self.groups[group_key] < strs_end]
            trx.groups[group_key] = _create_memmap(
                group_name, mode="w+", shape=(len(tmp),), dtype=group_dtype
            )
            logging.debug(
                "{} group went from {} items to {}".format(
                    group_key, ori_len, len(tmp))
            )
            trx.groups[group_key][:] = tmp

        if delete_dpg:
            self.close()
            self.__dict__ = trx.__dict__
            return

        if len(self.data_per_group.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir, "dpg/"))
        for group_key in self.data_per_group:
            if not os.path.isdir(os.path.join(tmp_dir, "dpg/", group_key)):
                os.mkdir(os.path.join(tmp_dir, "dpg/", group_key))
            if group_key not in trx.data_per_group:
                trx.data_per_group[group_key] = {}

            for dpg_key in self.data_per_group[group_key].keys():
                dpg_dtype = self.data_per_group[group_key][dpg_key].dtype
                dpg_filename = _generate_filename_from_data(
                    self.data_per_group[group_key][dpg_key],
                    os.path.join(tmp_dir, "dpg/", group_key, dpg_key),
                )

                shape = self.data_per_group[group_key][dpg_key].shape
                if dpg_key not in trx.data_per_group[group_key]:
                    trx.data_per_group[group_key][dpg_key] = {}
                trx.data_per_group[group_key][dpg_key] = _create_memmap(
                    dpg_filename, mode="w+", shape=shape, dtype=dpg_dtype
                )

                trx.data_per_group[group_key][dpg_key][:] = self.data_per_group[
                    group_key
                ][dpg_key]

        self.close()
        self.__dict__ = trx.__dict__

    def append(self, trx: Type["TrxFile"], extra_buffer: int = 0) -> None:
        """Append a TrxFile to another (support buffer)

        Keyword arguments:
            trx -- The TrxFile to append to the current TrxFile
            extra_buffer -- The additional buffer space required to append data
        """
        strs_end, pts_end = self._get_real_len()

        nb_streamlines = strs_end + trx.header["NB_STREAMLINES"]
        nb_vertices = pts_end + trx.header["NB_VERTICES"]

        if (
            self.header["NB_STREAMLINES"] < nb_streamlines
            or self.header["NB_VERTICES"] < nb_vertices
        ):
            self.resize(
                nb_streamlines=nb_streamlines + extra_buffer,
                nb_vertices=nb_vertices + extra_buffer * 100,
            )
        _ = concatenate([self, trx], preallocation=True, delete_groups=True)

    def get_group(
        self, key: str, keep_group: bool = True, copy_safe: bool = False
    ) -> Type["TrxFile"]:
        """Get a particular group from the TrxFile

        Keyword arguments:
            key -- The group name to select
            keep_group -- Make sure group exists in returned TrxFile
            copy_safe -- Perform a deepcopy

        Returns
            A TrxFile exclusively containing data from said group
        """
        return self.select(self.groups[key], keep_group=keep_group, copy_safe=copy_safe)

    def select(
        self, indices: ArrayLike, keep_group: bool = True, copy_safe: bool = False
    ) -> Type["TrxFile"]:
        """Get a subset of items, always vertices to the same memmaps

        Keyword arguments:
            indices -- The list of indices of elements to return
            keep_group -- Ensure group is returned in output TrxFile
            copy_safe -- Perform a deep-copy

        Returns:
            A TrxFile containing data originating from the selected indices
        """
        indices = np.array(indices, dtype=np.uint32)

        new_trx = TrxFile()
        new_trx._copy_safe = copy_safe
        new_trx.header = deepcopy(self.header)

        if isinstance(indices, np.ndarray) and len(indices) == 0:
            # Even while empty, basic dtype and header must be coherent
            positions_dtype = self.streamlines._data.dtype
            offsets_dtype = self.streamlines._offsets.dtype
            lengths_dtype = self.streamlines._lengths.dtype
            new_trx.streamlines._data = new_trx.streamlines._data.reshape(
                (0, 3)
            ).astype(positions_dtype)
            new_trx.streamlines._offsets = new_trx.streamlines._offsets.astype(
                offsets_dtype
            )
            new_trx.streamlines._lengths = new_trx.streamlines._lengths.astype(
                lengths_dtype
            )
            new_trx.header["NB_VERTICES"] = len(new_trx.streamlines._data)
            new_trx.header["NB_STREAMLINES"] = len(
                new_trx.streamlines._lengths)

            return new_trx.deepcopy() if copy_safe else new_trx

        new_trx.streamlines = (
            self.streamlines[indices].copy(
            ) if copy_safe else self.streamlines[indices]
        )
        for dpv_key in self.data_per_vertex.keys():
            new_trx.data_per_vertex[dpv_key] = (
                self.data_per_vertex[dpv_key][indices].copy()
                if copy_safe
                else self.data_per_vertex[dpv_key][indices]
            )

        for dps_key in self.data_per_streamline.keys():
            new_trx.data_per_streamline[dps_key] = (
                self.data_per_streamline[dps_key][indices].copy()
                if copy_safe
                else self.data_per_streamline[dps_key][indices]
            )

        # Not keeping group is equivalent to the [] operator
        if keep_group:
            logging.warning(
                "Keeping dpg despite affecting the group " "items.")
            for group_key in self.groups.keys():
                # Keep the group indices even when fancy slicing
                index = np.argsort(indices)
                sorted_x = indices[index]
                sorted_index = np.searchsorted(
                    sorted_x, self.groups[group_key])
                yindex = np.take(index, sorted_index, mode="clip")
                mask = indices[yindex] != self.groups[group_key]
                intersect = yindex[~mask]

                if len(intersect) == 0:
                    continue

                new_trx.groups[group_key] = intersect
                if group_key in self.data_per_group:
                    for dpg_key in self.data_per_group[group_key].keys():
                        if group_key not in new_trx.data_per_group:
                            new_trx.data_per_group[group_key] = {}
                        new_trx.data_per_group[group_key][
                            dpg_key
                        ] = self.data_per_group[group_key][dpg_key]

        new_trx.header["NB_VERTICES"] = len(new_trx.streamlines._data)
        new_trx.header["NB_STREAMLINES"] = len(new_trx.streamlines._lengths)
        return new_trx.deepcopy() if copy_safe else new_trx

    @staticmethod
    def from_sft(sft, cast_position=np.float32):
        """Generate a valid TrxFile from a StatefulTractogram"""
        try:
            from dipy.io.stateful_tractogram import StatefulTractogram, Space
        except ImportError:
            logging.error('Dipy library is missing, cannot convert to '
                          'StatefulTractogram.')
            return None

        if not np.issubdtype(cast_position, np.floating):
            logging.warning(
                "Casting as {}, considering using a floating point "
                "dtype.".format(cast_position)
            )

        trx = TrxFile(nb_vertices=len(sft.streamlines._data),
                      nb_streamlines=len(sft.streamlines))
        trx.header = {
            "DIMENSIONS": sft.dimensions.tolist(),
            "VOXEL_TO_RASMM": sft.affine.tolist(),
            "NB_VERTICES": len(sft.streamlines._data),
            "NB_STREAMLINES": len(sft.streamlines),
        }

        old_space = deepcopy(sft.space)
        old_origin = deepcopy(sft.origin)

        # TrxFile are written on disk in RASMM/center convention
        sft.to_rasmm()
        sft.to_center()
        if cast_position != np.float32:
            tmp_streamlines = deepcopy(sft.streamlines)
        else:
            tmp_streamlines = sft.streamlines
        sft.to_space(old_space)
        sft.to_origin(old_origin)

        # Cast the int64 of Nibabel to uint32
        tmp_streamlines._offsets = tmp_streamlines._offsets.astype(np.uint32)
        if cast_position != np.float32:
            tmp_streamlines._data = tmp_streamlines._data.astype(cast_position)

        trx.streamlines = tmp_streamlines
        trx.data_per_streamline = sft.data_per_streamline
        trx.data_per_vertex = sft.data_per_point

        # For safety and for RAM, convert the whole object to memmaps
        tmpdir = tempfile.TemporaryDirectory()
        save(trx, tmpdir.name)
        trx = load_from_directory(tmpdir.name)
        trx._uncompressed_folder_handle = tmpdir

        return trx

    @staticmethod
    def from_tractogram(tractogram, reference, cast_position=np.float32):
        """Generate a valid TrxFile from a Nibabel Tractogram"""
        if not np.issubdtype(cast_position, np.floating):
            logging.warning(
                "Casting as {}, considering using a floating point "
                "dtype.".format(cast_position)
            )

        trx = TrxFile(
            nb_vertices=len(tractogram.streamlines._data),
            nb_streamlines=len(tractogram.streamlines),
        )

        affine, dimensions, _, _ = get_reference_info_wrapper(reference)
        trx.header = {
            "DIMENSIONS": dimensions,
            "VOXEL_TO_RASMM": affine,
            "NB_VERTICES": len(tractogram.streamlines._data),
            "NB_STREAMLINES": len(tractogram.streamlines),
        }

        if cast_position != np.float32:
            tmp_streamlines = deepcopy(tractogram.streamlines)
        else:
            tmp_streamlines = tractogram.streamlines

        # Cast the int64 of Nibabel to uint32
        tmp_streamlines._offsets = tmp_streamlines._offsets.astype(np.uint32)
        if cast_position != np.float32:
            tmp_streamlines._data = tmp_streamlines._data.astype(cast_position)

        trx.streamlines = tmp_streamlines
        trx.data_per_streamline = tractogram.data_per_streamline
        trx.data_per_vertex = tractogram.data_per_point

        # For safety and for RAM, convert the whole object to memmaps
        tmpdir = tempfile.TemporaryDirectory()
        save(trx, tmpdir.name)
        trx = load_from_directory(tmpdir.name)
        trx._uncompressed_folder_handle = tmpdir

        return trx

    def to_tractogram(self, resize=False):
        """Convert a TrxFile to a nibabel Tractogram (in RAM)"""
        if resize:
            self.resize()

        trx_obj = self.to_memory()
        tractogram = nib.streamlines.Tractogram([], affine_to_rasmm=np.eye(4))
        tractogram._set_streamlines(trx_obj.streamlines)
        tractogram._data_per_point = trx_obj.data_per_vertex
        tractogram._data_per_streamline = trx_obj.data_per_streamline

        return tractogram

    def to_memory(self, resize: bool = False) -> Type["TrxFile"]:
        """Convert a TrxFile to a RAM representation

        Keyword arguments:
            resize -- Resize TrxFile when converting to RAM representation

        Returns:
            A non memory mapped TrxFile
        """
        if resize:
            self.resize()

        trx_obj = TrxFile()
        trx_obj.header = deepcopy(self.header)
        trx_obj.streamlines = deepcopy(self.streamlines)

        for key in self.data_per_vertex:
            trx_obj.data_per_vertex[key] = deepcopy(self.data_per_vertex[key])

        for key in self.data_per_streamline:
            trx_obj.data_per_streamline[key] = deepcopy(
                self.data_per_streamline[key])

        for key in self.groups:
            trx_obj.groups[key] = deepcopy(self.groups[key])

        for key in self.data_per_group:
            trx_obj.data_per_group[key] = deepcopy(self.data_per_group[key])

        return trx_obj

    def to_sft(self, resize=False):
        """Convert a TrxFile to a valid StatefulTractogram (in RAM)"""
        try:
            from dipy.io.stateful_tractogram import StatefulTractogram, Space
        except ImportError:
            logging.error('Dipy library is missing, cannot convert to '
                          'StatefulTractogram.')
            return None

        affine = np.array(self.header["VOXEL_TO_RASMM"], dtype=np.float32)
        dimensions = np.array(self.header["DIMENSIONS"], dtype=np.uint16)
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = "".join(aff2axcodes(affine))
        space_attributes = (affine, dimensions, vox_sizes, vox_order)

        if resize:
            self.resize()
        sft = StatefulTractogram(
            self.streamlines,
            space_attributes,
            Space.RASMM,
            data_per_point=self.data_per_vertex,
            data_per_streamline=self.data_per_streamline,
        )

        return sft

    def close(self) -> None:
        """Cleanup on-disk temporary folder and initialize an empty TrxFile"""
        if self._uncompressed_folder_handle is not None:
            self._uncompressed_folder_handle.cleanup()
        self.__init__()
        logging.debug("Deleted memmaps and intialized empty TrxFile.")
