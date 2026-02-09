# -*- coding: utf-8 -*-
"""Core TrxFile class with memory-mapped data access."""

from copy import deepcopy
import json
import logging
import os
import shutil
import struct
from typing import Any, List, Optional, Tuple, Type, Union
import zipfile

import nibabel as nib
from nibabel.affines import voxel_sizes
from nibabel.nifti1 import Nifti1Header, Nifti1Image
from nibabel.orientations import aff2axcodes
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import LazyTractogram, Tractogram
from nibabel.streamlines.trk import TrkFile
import numpy as np

from trx.io import get_trx_tmp_dir
from trx.utils import (
    append_generator_to_dict,
    close_or_delete_mmap,
    convert_data_dict_to_tractogram,
    get_reference_info_wrapper,
)

try:
    import dipy  # noqa: F401

    dipy_available = True
except ImportError:
    dipy_available = False


def _get_dtype_little_endian(dtype: Union[np.dtype, str, type]) -> np.dtype:
    """Convert a dtype to its little-endian equivalent.

    The TRX file format uses little-endian byte order for cross-platform
    compatibility. This function ensures that dtypes are always interpreted
    as little-endian when reading/writing TRX files.

    Parameters
    ----------
    dtype : np.dtype, str, or type
        Input dtype specification (e.g., np.float32, 'float32', '>f4').

    Returns
    -------
    np.dtype
        Little-endian dtype. For single-byte types (uint8, int8, bool),
        returns the original dtype as endianness is not applicable.
    """
    dt = np.dtype(dtype)
    # Single-byte types don't have endianness
    if dt.byteorder == "|" or dt.itemsize == 1:
        return dt
    # Already little-endian
    if dt.byteorder == "<":
        return dt
    # Convert to little-endian
    return dt.newbyteorder("<")


def _ensure_little_endian(arr: np.ndarray) -> np.ndarray:
    """Ensure array data is in little-endian byte order for writing.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array with little-endian byte order. Returns a copy if conversion
        was needed, otherwise returns the original array.
    """
    dt = arr.dtype
    # Single-byte types don't have endianness
    if dt.byteorder == "|" or dt.itemsize == 1:
        return arr
    # Already little-endian
    if dt.byteorder == "<":
        return arr
    # Native byte order on little-endian system
    if dt.byteorder == "=" and np.little_endian:
        return arr
    # Convert to little-endian
    return arr.astype(dt.newbyteorder("<"))


def _append_last_offsets(nib_offsets: np.ndarray, nb_vertices: int) -> np.ndarray:
    """Append the last element of offsets from header information.

    Parameters
    ----------
    nib_offsets : np.ndarray
        Array of offsets with the last element being the start of the last
        streamline (nibabel convention).
    nb_vertices : int
        Total number of vertices in the streamlines.

    Returns
    -------
    np.ndarray
        Offsets array (VTK convention).
    """

    def is_sorted(a):
        """Return True if array is sorted non-decreasing.

        Parameters
        ----------
        a : np.ndarray
            1D array of numeric offsets.

        Returns
        -------
        bool
            True when ``a`` is monotonically non-decreasing.
        """
        return np.all(a[:-1] <= a[1:])

    if not is_sorted(nib_offsets):
        raise ValueError("Offsets must be sorted values.")
    return np.append(nib_offsets, nb_vertices).astype(nib_offsets.dtype)


def _generate_filename_from_data(arr: np.ndarray, filename: str) -> str:
    """Determine the data type from array data and generate the appropriate filename.

    Parameters
    ----------
    arr : np.ndarray
        A NumPy array (1-2D, otherwise ValueError raised).
    filename : str
        The original filename.

    Returns
    -------
    str
        An updated filename with appropriate extension.
    """
    base, ext = os.path.splitext(filename)
    if ext:
        logging.warning("Will overwrite provided extension if needed.")

    dtype = arr.dtype
    dtype = "bit" if dtype is np.dtype(bool) else dtype.name

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
    """Take a filename and split it into its components.

    Parameters
    ----------
    filename : str
        Input filename.

    Returns
    -------
    tuple
        A tuple of (basename, dimension, extension).
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


def _compute_lengths(offsets: np.ndarray) -> np.ndarray:
    """Compute lengths from offsets.

    Parameters
    ----------
    offsets : np.ndarray
        An array of offsets.

    Returns
    -------
    np.ndarray
        An array of lengths.
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
    """Verify that filename extension is a valid datatype.

    Parameters
    ----------
    ext : str
        Filename extension.

    Returns
    -------
    bool
        True if the provided datatype is valid, False otherwise.
    """
    if ext.replace(".", "") == "bit":
        return True
    try:
        isinstance(np.dtype(ext.replace(".", "")), np.dtype)
        return True
    except TypeError:
        return False


def _dichotomic_search(
    x: np.ndarray, l_bound: Optional[int] = None, r_bound: Optional[int] = None
) -> int:
    """Find where data of a contiguous array is actually ending.

    Parameters
    ----------
    x : np.ndarray
        Array of values.
    l_bound : int, optional
        Lower bound index for search.
    r_bound : int, optional
        Upper bound index for search.

    Returns
    -------
    int
        Index at which array value is 0 (if possible), otherwise returns -1.
    """
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
) -> np.ndarray:
    """Wrap memmap creation to support empty arrays.

    Parameters
    ----------
    filename : str
        Filename where the empty memmap should be created.
    mode : str, optional
        File open mode (see np.memmap for options). Default is 'r'.
    shape : tuple, optional
        Shape of memmapped array. Default is (1,).
    dtype : np.dtype, optional
        Datatype of memmapped array. Default is np.float32.
    offset : int, optional
        Offset of the data within the file. Default is 0.
    order : str, optional
        Data representation on disk ('C' or 'F'). Default is 'C'.

    Returns
    -------
    np.ndarray
        Memory-mapped array or a zero-filled array if shape[0] is 0.
    """
    if np.dtype(dtype) == bool:
        filename = filename.replace(".bool", ".bit")

    # TRX format uses little-endian byte order for cross-platform compatibility
    dtype = _get_dtype_little_endian(dtype)

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
    """Load a TrxFile (compressed or not).

    Parameters
    ----------
    input_obj : str
        A directory name or filepath to the TRX data.
    check_dpg : bool, optional
        Whether to check group metadata. Default is True.

    Returns
    -------
    TrxFile
        TrxFile object representing the read data.
    """
    # TODO Check if 0 streamlines, then 0 vertices is expected (vice-versa)
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
                tmp_dir = get_trx_tmp_dir()
                zf.extractall(tmp_dir.name)
                trx = load_from_directory(tmp_dir.name)
                trx._uncompressed_folder_handle = tmp_dir
                logging.info(
                    "File was compressed, call the close() function before exiting."
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
                    "An undeclared group ({}) has data_per_group.".format(dpg)
                )
    return trx


def load_from_zip(filename: str) -> Type["TrxFile"]:
    """Load a TrxFile from a single zipfile.

    Note: Does not work with compressed zipfiles.

    Parameters
    ----------
    filename : str
        Path of the zipped TrxFile.

    Returns
    -------
    TrxFile
        TrxFile representing the read data.
    """
    with zipfile.ZipFile(filename, mode="r") as zf:
        with zf.open("header.json") as zf_header:
            header = json.load(zf_header)
            header["VOXEL_TO_RASMM"] = np.reshape(
                header["VOXEL_TO_RASMM"], (4, 4)
            ).astype(np.float32)
            header["DIMENSIONS"] = np.array(header["DIMENSIONS"], dtype=np.uint16)

        files_pointer_size = {}
        for zip_info in zf.filelist:
            elem_filename = zip_info.filename
            _, ext = os.path.splitext(elem_filename)
            if ext == ".json" or zip_info.is_dir():
                continue

            if not _is_dtype_valid(ext):
                continue
                raise ValueError("The dtype {} is not supported".format(elem_filename))

            if ext == ".bit":
                ext = ".bool"

            # Read actual local file header to get correct data offset.
            # We can't use zip_info.FileHeader() because ZIP spec allows local
            # headers to differ from central directory entries.
            # See: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
            _ZIP_LOCAL_HEADER_SIZE = 30
            _ZIP_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"

            zf.fp.seek(zip_info.header_offset)
            local_header = zf.fp.read(_ZIP_LOCAL_HEADER_SIZE)
            if len(local_header) < _ZIP_LOCAL_HEADER_SIZE:
                raise ValueError(f"Truncated local file header for {elem_filename}")
            if local_header[:4] != _ZIP_LOCAL_HEADER_SIGNATURE:
                raise ValueError(
                    f"Invalid local file header signature for {elem_filename}"
                )
            fname_len, extra_len = struct.unpack("<HH", local_header[26:30])

            mem_adress = (
                zip_info.header_offset + _ZIP_LOCAL_HEADER_SIZE + fname_len + extra_len
            )

            dtype_size = np.dtype(ext[1:]).itemsize
            size = zip_info.file_size / dtype_size

            if size.is_integer():
                files_pointer_size[elem_filename] = mem_adress, int(size)
            else:
                raise ValueError("Wrong size or datatype")

    return TrxFile._create_trx_from_pointer(
        header, files_pointer_size, root_zip=filename
    )


def load_from_directory(directory: str) -> Type["TrxFile"]:
    """Load a TrxFile from a folder containing memmaps.

    Parameters
    ----------
    directory : str
        Path of the directory containing TRX data.

    Returns
    -------
    TrxFile
        TrxFile representing the read data.
    """

    directory = os.path.abspath(directory)
    with open(os.path.join(directory, "header.json")) as header:
        header = json.load(header)
        header["VOXEL_TO_RASMM"] = np.reshape(header["VOXEL_TO_RASMM"], (4, 4)).astype(
            np.float32
        )
        header["DIMENSIONS"] = np.array(header["DIMENSIONS"], dtype=np.uint16)
    files_pointer_size = {}
    for root, _dirs, files in os.walk(directory):
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


def _filter_empty_trx_files(trx_list: List["TrxFile"]) -> List["TrxFile"]:
    """Remove empty TrxFiles from the list.

    Parameters
    ----------
    trx_list : list of TrxFile class instances
        Collection of tractograms to filter.

    Returns
    -------
    list of TrxFile class instances
        Only entries containing at least one streamline.
    """
    return [curr_trx for curr_trx in trx_list if curr_trx.header["NB_STREAMLINES"] > 0]


def _get_all_data_keys(trx_list: List["TrxFile"]) -> Tuple[set, set]:
    """Get all dps and dpv keys from the TrxFile list.

    Parameters
    ----------
    trx_list : list of TrxFile class instances
        Collection of tractograms.

    Returns
    -------
    tuple of set
        Sets of `data_per_streamline` keys and `data_per_vertex` keys.
    """
    all_dps = []
    all_dpv = []
    for curr_trx in trx_list:
        all_dps.extend(list(curr_trx.data_per_streamline.keys()))
        all_dpv.extend(list(curr_trx.data_per_vertex.keys()))
    return set(all_dps), set(all_dpv)


def _check_space_attributes(trx_list: List["TrxFile"]) -> None:
    """Verify that space attributes are consistent across TrxFiles.

    Parameters
    ----------
    trx_list : list of TrxFile
        Tractograms to compare for affine and dimension consistency.

    Raises
    ------
    ValueError
        If voxel-to-RASMM matrices or dimensions differ.
    """
    ref_trx = trx_list[0]
    for curr_trx in trx_list[1:]:
        if not np.allclose(
            ref_trx.header["VOXEL_TO_RASMM"], curr_trx.header["VOXEL_TO_RASMM"]
        ) or not np.array_equal(
            ref_trx.header["DIMENSIONS"], curr_trx.header["DIMENSIONS"]
        ):
            raise ValueError("Wrong space attributes.")


def _verify_dpv_coherence(
    trx_list: List["TrxFile"], all_dpv: set, ref_trx: "TrxFile", delete_dpv: bool
) -> None:
    """Verify dpv coherence across TrxFiles.

    Parameters
    ----------
    trx_list : list of TrxFile class instances
        Tractograms being concatenated.
    all_dpv : set
        Union of `data_per_vertex` keys across tractograms.
    ref_trx : TrxFile class instance
        Reference tractogram for dtype/key checks.
    delete_dpv : bool
        Drop mismatched dpv keys instead of raising when True.

    Raises
    ------
    ValueError
        If dpv keys or dtypes differ and `delete_dpv` is False.
    """
    for curr_trx in trx_list:
        for key in all_dpv:
            if (
                key not in ref_trx.data_per_vertex.keys()
                or key not in curr_trx.data_per_vertex.keys()
            ):
                if not delete_dpv:
                    logging.debug(
                        "{} dpv key does not exist in all TrxFile.".format(key)
                    )
                    raise ValueError("TrxFile must be sharing identical dpv keys.")
            elif (
                ref_trx.data_per_vertex[key]._data.dtype
                != curr_trx.data_per_vertex[key]._data.dtype
            ):
                logging.debug(
                    "{} dpv key is not declared with the same dtype "
                    "in all TrxFile.".format(key)
                )
                raise ValueError("Shared dpv key, has different dtype.")


def _verify_dps_coherence(
    trx_list: List["TrxFile"], all_dps: set, ref_trx: "TrxFile", delete_dps: bool
) -> None:
    """Verify dps coherence across TrxFiles.

    Parameters
    ----------
    trx_list : list of TrxFile class instances
        Tractograms being concatenated.
    all_dps : set
        Union of data_per_streamline keys across tractograms.
    ref_trx : TrxFile class instance
        Reference tractogram for dtype/key checks.
    delete_dps : bool
        Drop mismatched dps keys instead of raising when True.

    Raises
    ------
    ValueError
        If dps keys or dtypes differ and `delete_dps` is False.
    """
    for curr_trx in trx_list:
        for key in all_dps:
            if (
                key not in ref_trx.data_per_streamline.keys()
                or key not in curr_trx.data_per_streamline.keys()
            ):
                if not delete_dps:
                    logging.debug(
                        "{} dps key does not exist in all TrxFile.".format(key)
                    )
                    raise ValueError("TrxFile must be sharing identical dps keys.")
            elif (
                ref_trx.data_per_streamline[key].dtype
                != curr_trx.data_per_streamline[key].dtype
            ):
                logging.debug(
                    "{} dps key is not declared with the same dtype "
                    "in all TrxFile.".format(key)
                )
                raise ValueError("Shared dps key, has different dtype.")


def _compute_groups_info(trx_list: List["TrxFile"]) -> Tuple[dict, dict]:
    """Compute group length and dtype information.

    Parameters
    ----------
    trx_list : list of TrxFile class instances
        Tractograms being concatenated.

    Returns
    -------
    tuple of dict
        (group lengths, group dtypes) keyed by group name.
    """
    all_groups_len = {}
    all_groups_dtype = {}

    for trx_1 in trx_list:
        for group_key in trx_1.groups.keys():
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

    return all_groups_len, all_groups_dtype


def _create_new_trx_for_concatenation(
    trx_list: List["TrxFile"],
    ref_trx: "TrxFile",
    delete_dps: bool,
    delete_dpv: bool,
    delete_groups: bool,
) -> "TrxFile":
    """Create a new TrxFile for concatenation.

    Parameters
    ----------
    trx_list : list of TrxFile class instances
        Input tractograms to concatenate.
    ref_trx : TrxFile class instance
        Reference tractogram for header/dtype template.
    delete_dps : bool
        Drop `data_per_streamline` keys not shared.
    delete_dpv : bool
        Drop `data_per_vertex` keys not shared.
    delete_groups : bool
        Drop groups when metadata differ.

    Returns
    -------
    TrxFile
        Empty TRX ready to receive concatenated data.
    """
    nb_vertices = 0
    nb_streamlines = 0
    for curr_trx in trx_list:
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

    return new_trx


def _setup_groups_for_concatenation(
    new_trx: "TrxFile",
    trx_list: List["TrxFile"],
    all_groups_len: dict,
    all_groups_dtype: dict,
    delete_groups: bool,
) -> None:
    """Setup groups in the new TrxFile for concatenation.

    Parameters
    ----------
    new_trx : TrxFile class instance
        Destination tractogram.
    trx_list : list of TrxFile class instances
        Source tractograms.
    all_groups_len : dict
        Mapping of group name to total length.
    all_groups_dtype : dict
        Mapping of group name to dtype.
    delete_groups : bool
        If True, skip creating group arrays.
    """
    if delete_groups:
        return

    tmp_dir = new_trx._uncompressed_folder_handle.name

    for group_key in all_groups_len.keys():
        if not os.path.isdir(os.path.join(tmp_dir, "groups/")):
            os.mkdir(os.path.join(tmp_dir, "groups/"))

        dtype = all_groups_dtype[group_key]
        group_filename = os.path.join(
            tmp_dir, "groups/{}.{}".format(group_key, dtype.name)
        )
        group_len = all_groups_len[group_key]
        new_trx.groups[group_key] = _create_memmap(
            group_filename, mode="w+", shape=(group_len,), dtype=dtype
        )

        pos = 0
        count = 0
        for curr_trx in trx_list:
            curr_len = len(curr_trx.groups[group_key])
            new_trx.groups[group_key][pos : pos + curr_len] = (
                curr_trx.groups[group_key] + count
            )
            pos += curr_len
            count += curr_trx.header["NB_STREAMLINES"]


def concatenate(
    trx_list: List["TrxFile"],
    delete_dpv: bool = False,
    delete_dps: bool = False,
    delete_groups: bool = False,
    check_space_attributes: bool = True,
    preallocation: bool = False,
) -> "TrxFile":
    """Concatenate multiple TrxFile together, with support for preallocation.

    Parameters
    ----------
    trx_list : list of TrxFile
        A list containing TrxFiles to concatenate.
    delete_dpv : bool, optional
        Delete dpv keys that do not exist in all the provided TrxFiles.
        Default is False.
    delete_dps : bool, optional
        Delete dps keys that do not exist in all the provided TrxFiles.
        Default is False.
    delete_groups : bool, optional
        Delete all the groups that currently exist in the TrxFiles.
        Default is False.
    check_space_attributes : bool, optional
        Verify that dimensions and size of data are similar between all
        the TrxFiles. Default is True.
    preallocation : bool, optional
        Preallocated TrxFile has already been generated and is the first
        element in trx_list. Note: delete_groups must be set to True as well.
        Default is False.

    Returns
    -------
    TrxFile
        TrxFile representing the concatenated data.
    """
    trx_list = _filter_empty_trx_files(trx_list)
    if len(trx_list) == 0:
        logging.warning("Inputs of concatenation were empty.")
        return TrxFile()

    ref_trx = trx_list[0]
    all_dps, all_dpv = _get_all_data_keys(trx_list)

    if check_space_attributes:
        _check_space_attributes(trx_list)

    if preallocation and not delete_groups:
        raise ValueError("Groups are variables, cannot be handled with preallocation")

    _verify_dpv_coherence(trx_list, all_dpv, ref_trx, delete_dpv)
    _verify_dps_coherence(trx_list, all_dps, ref_trx, delete_dps)

    all_groups_len, all_groups_dtype = _compute_groups_info(trx_list)

    to_concat_list = trx_list[1:] if preallocation else trx_list
    if not preallocation:
        new_trx = _create_new_trx_for_concatenation(
            to_concat_list, ref_trx, delete_dps, delete_dpv, delete_groups
        )
        _setup_groups_for_concatenation(
            new_trx, trx_list, all_groups_len, all_groups_dtype, delete_groups
        )
        strs_end, pts_end = 0, 0
    else:
        new_trx = ref_trx
        strs_end, pts_end = new_trx._get_real_len()

    for curr_trx in to_concat_list:
        strs_end, pts_end = new_trx._copy_fixed_arrays_from(
            curr_trx, strs_start=strs_end, pts_start=pts_end
        )
    return new_trx


def save(
    trx: "TrxFile", filename: str, compression_standard: Any = zipfile.ZIP_STORED
) -> None:
    """Save a TrxFile (compressed or not).

    Parameters
    ----------
    trx : TrxFile
        The TrxFile to save.
    filename : str
        The path to save the TrxFile to.
    compression_standard : int, optional
        The compression standard to use, as defined by the ZipFile library.
        Default is zipfile.ZIP_STORED.
    """
    _, ext = os.path.splitext(filename)
    if ext not in [".zip", ".trx", ""]:
        raise ValueError("Unsupported extension.")

    copy_trx = trx.deepcopy()
    copy_trx.resize()
    tmp_dir_name = copy_trx._uncompressed_folder_handle.name
    if ext in [".zip", ".trx"]:
        zip_from_folder(tmp_dir_name, filename, compression_standard)
    else:
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        shutil.copytree(tmp_dir_name, filename)
    copy_trx.close()


def zip_from_folder(
    directory: str, filename: str, compression_standard: Any = zipfile.ZIP_STORED
) -> None:
    """Zip on-disk memmaps into a single file.

    Parameters
    ----------
    directory : str
        The path to the on-disk memmap directory.
    filename : str
        The path where the zip file should be created.
    compression_standard : int, optional
        The compression standard to use, as defined by the ZipFile library.
        Default is zipfile.ZIP_STORED.
    """
    with zipfile.ZipFile(filename, mode="w", compression=compression_standard) as zf:
        for root, _, files in os.walk(directory):
            for name in files:
                curr_filename = os.path.join(root, name)
                tmp_filename = curr_filename.replace(directory, "")[1:]
                zf.write(curr_filename, tmp_filename)


class TrxFile:
    """Core class of the TrxFile.

    Parameters
    ----------
    nb_vertices : int, optional
        The number of vertices to use in the new TrxFile.
    nb_streamlines : int, optional
        The number of streamlines in the new TrxFile.
    init_as : TrxFile class instance, optional
        A TrxFile to use as reference.

    reference : str, dict, Nifti1Image, TrkFile, or Nifti1Header, optional
        A Nifti or Trk file/obj to use as reference.
    """

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
        """Initialize an empty TrxFile with support for preallocation.

        Parameters
        ----------
        nb_vertices : int, optional
            The number of vertices to use in the new TrxFile.
        nb_streamlines : int, optional
            The number of streamlines in the new TrxFile.
        init_as : TrxFile, optional
            A TrxFile to use as reference.
        reference : str, dict, Nifti1Image, TrkFile, Nifti1Header, optional
            A Nifti or Trk file/obj to use as reference.
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
                    "Can't use init_as without declaring nb_vertices AND nb_streamlines"
                )
            logging.debug("Initializing empty TrxFile.")
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
                "Preallocating TrxFile with size {} streamlinesand {} vertices.".format(
                    nb_streamlines, nb_vertices
                )
            )
            trx = self._initialize_empty_trx(
                nb_streamlines, nb_vertices, init_as=init_as
            )
            self.__dict__ = trx.__dict__
        else:
            raise ValueError("You must declare both nb_vertices AND NB_STREAMLINES")

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
            np.array2string(affine, formatter={"float_kind": lambda x: "%.6f" % x})
        )
        text += "\nDIMENSIONS: {}".format(np.array2string(dimensions))
        text += "\nVOX_SIZES: {}".format(
            np.array2string(vox_sizes, formatter={"float_kind": lambda x: "%.2f" % x})
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

        dpv_keys = list(self.data_per_vertex.keys())
        if dpv_keys:
            text += "\ndata_per_vertex keys: {}".format(dpv_keys)
        else:
            text += "\nNo data per vertex (dpv) keys"

        dps_keys = list(self.data_per_streamline.keys())
        if dps_keys:
            text += "\ndata_per_streamline keys: {}".format(dps_keys)
        else:
            text += "\nNo data per streamline (dps) keys"

        group_keys = list(self.groups.keys())
        if group_keys:
            text += "\ngroups keys: {}".format(group_keys)
        else:
            text += "\nNo group keys"
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
            key = list(range(*key.indices(len(self))))

        return self.select(key, keep_group=False)

    def __deepcopy__(self) -> Type["TrxFile"]:
        """Return a deep copy of the TrxFile.

        Parameters
        ----------
        self
            TrxFile class instance.

        Returns
        -------
        TrxFile class instance
            Deep-copied instance.
        """
        return self.deepcopy()

    def deepcopy(self) -> Type["TrxFile"]:  # noqa: C901
        """Create a deepcopy of the TrxFile.

        Returns
        -------
        TrxFile
            A deepcopied TrxFile of the current TrxFile.
        """
        tmp_dir = get_trx_tmp_dir()
        out_json = open(os.path.join(tmp_dir.name, "header.json"), "w")
        tmp_header = deepcopy(self.header)

        if not isinstance(tmp_header["VOXEL_TO_RASMM"], list):
            tmp_header["VOXEL_TO_RASMM"] = tmp_header["VOXEL_TO_RASMM"].tolist()
        if not isinstance(tmp_header["DIMENSIONS"], list):
            tmp_header["DIMENSIONS"] = tmp_header["DIMENSIONS"].tolist()

        # tofile() always write in C-order
        # Ensure little-endian byte order for cross-platform compatibility
        if not self._copy_safe:
            to_dump = self.streamlines.copy()._data
            tmp_header["NB_STREAMLINES"] = len(self.streamlines)
            tmp_header["NB_VERTICES"] = len(to_dump)
        else:
            to_dump = self.streamlines._data
        json.dump(tmp_header, out_json)
        out_json.close()

        # Only write positions and offsets if TRX is not empty
        if tmp_header["NB_STREAMLINES"] > 0 and tmp_header["NB_VERTICES"] > 0:
            positions_filename = _generate_filename_from_data(
                to_dump, os.path.join(tmp_dir.name, "positions")
            )
            _ensure_little_endian(to_dump).tofile(positions_filename)

            if not self._copy_safe:
                to_dump = _append_last_offsets(
                    self.streamlines.copy()._offsets, self.header["NB_VERTICES"]
                )
            else:
                to_dump = _append_last_offsets(
                    self.streamlines._offsets, self.header["NB_VERTICES"]
                )
            offsets_filename = _generate_filename_from_data(
                self.streamlines._offsets, os.path.join(tmp_dir.name, "offsets")
            )
            _ensure_little_endian(to_dump).tofile(offsets_filename)

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
            _ensure_little_endian(to_dump).tofile(dpv_filename)

        if len(self.data_per_streamline.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir.name, "dps/"))
        for dps_key in self.data_per_streamline.keys():
            to_dump = self.data_per_streamline[dps_key]
            dps_filename = _generate_filename_from_data(
                to_dump, os.path.join(tmp_dir.name, "dps/", dps_key)
            )
            _ensure_little_endian(to_dump).tofile(dps_filename)

        if len(self.groups.keys()) > 0:
            os.mkdir(os.path.join(tmp_dir.name, "groups/"))
        for group_key in self.groups.keys():
            to_dump = self.groups[group_key]
            group_filename = _generate_filename_from_data(
                to_dump, os.path.join(tmp_dir.name, "groups/", group_key)
            )
            _ensure_little_endian(to_dump).tofile(group_filename)

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
                    to_dump, os.path.join(tmp_dir.name, "dpg/", group_key, dpg_key)
                )
                _ensure_little_endian(to_dump).tofile(dpg_filename)

        copy_trx = load_from_directory(tmp_dir.name)
        copy_trx._uncompressed_folder_handle = tmp_dir

        return copy_trx

    def _get_real_len(self) -> Tuple[int, int]:
        """Get the real size of data (ignoring zeros of preallocation).

        Returns
        -------
        tuple of int
            A tuple (strs_end, pts_end) representing the index of the last
            streamline and the total length of all the streamlines.
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
        """Fill a TrxFile using another and start indexes (preallocation).

        Parameters
        ----------
        trx : TrxFile
            TrxFile to copy data from.
        strs_start : int, optional
            The start index of the streamline. Default is 0.
        pts_start : int, optional
            The start index of the point. Default is 0.
        nb_strs_to_copy : int, optional
            The number of streamlines to copy. If not set, will copy all.

        Returns
        -------
        tuple of int
            A tuple (strs_end, pts_end) representing the end of the copied
            streamlines and end of copied points.
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
            self.data_per_vertex[dpv_key]._data[pts_start:pts_end] = (
                trx.data_per_vertex[dpv_key]._data[0:curr_pts_len]
            )
            self.data_per_vertex[dpv_key]._offsets = self.streamlines._offsets
            self.data_per_vertex[dpv_key]._lengths = self.streamlines._lengths

        for dps_key in self.data_per_streamline.keys():
            self.data_per_streamline[dps_key][strs_start:strs_end] = (
                trx.data_per_streamline[dps_key][0:curr_strs_len]
            )

        return strs_end, pts_end

    @staticmethod
    def _initialize_empty_trx(  # noqa: C901
        nb_streamlines: int,
        nb_vertices: int,
        init_as: Optional[Type["TrxFile"]] = None,
    ) -> Type["TrxFile"]:
        """Create on-disk memmaps of a certain size (preallocation).

        Parameters
        ----------
        nb_streamlines : int
            The number of streamlines that the empty TrxFile will be
            initialized with.
        nb_vertices : int
            The number of vertices that the empty TrxFile will be
            initialized with.
        init_as : TrxFile, optional
            A TrxFile to initialize the empty TrxFile with.

        Returns
        -------
        TrxFile
            An empty TrxFile preallocated with a certain size.
        """
        trx = TrxFile()
        tmp_dir = get_trx_tmp_dir()
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
            "Initializing positions with dtype:    {}".format(positions_dtype.name)
        )
        logging.debug("Initializing offsets with dtype: {}".format(offsets_dtype.name))
        logging.debug("Initializing lengths with dtype: {}".format(lengths_dtype.name))

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
                        tmp_dir.name, "dpv/{}.{}".format(dpv_key, dtype.name)
                    )
                    shape = (nb_vertices, 1)
                elif tmp_as.ndim == 2:
                    dim = tmp_as.shape[-1]
                    shape = (nb_vertices, dim)
                    dpv_filename = os.path.join(
                        tmp_dir.name, "dpv/{}.{}.{}".format(dpv_key, dim, dtype.name)
                    )
                else:
                    raise ValueError("Invalid dimensionality.")

                logging.debug(
                    "Initializing {} (dpv) with dtype: {}".format(dpv_key, dtype.name)
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
                        tmp_dir.name, "dps/{}.{}".format(dps_key, dtype.name)
                    )
                    shape = (nb_streamlines,)
                elif tmp_as.ndim == 2:
                    dim = tmp_as.shape[-1]
                    shape = (nb_streamlines, dim)
                    dps_filename = os.path.join(
                        tmp_dir.name, "dps/{}.{}.{}".format(dps_key, dim, dtype.name)
                    )
                else:
                    raise ValueError("Invalid dimensionality.")

                logging.debug(
                    "Initializing {} (dps) with and dtype: {}".format(
                        dps_key, dtype.name
                    )
                )
                trx.data_per_streamline[dps_key] = _create_memmap(
                    dps_filename, mode="w+", shape=shape, dtype=dtype
                )

        trx._uncompressed_folder_handle = tmp_dir

        return trx

    def _create_trx_from_pointer(  # noqa: C901
        header: dict,
        dict_pointer_size: dict,
        root_zip: Optional[str] = None,
        root: Optional[str] = None,
    ) -> Type["TrxFile"]:
        """Create a TrxFile after reading the structure of a zip/folder.

        Parameters
        ----------
        header : dict
            A TrxFile header dictionary which will be used for the new TrxFile.
        dict_pointer_size : dict
            A dictionary containing the filenames of all the files within the
            TrxFile disk file/folder.
        root_zip : str, optional
            The path of the ZipFile pointer.
        root : str, optional
            The dirname of the ZipFile pointer.

        Returns
        -------
        TrxFile
            A TrxFile constructed from the pointer provided.
        """
        trx = TrxFile()
        trx.header = header

        # Handle empty TRX files early - no positions/offsets to load
        if header["NB_STREAMLINES"] == 0 or header["NB_VERTICES"] == 0:
            return trx

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

            if root is not None:
                # This is for Unix
                if os.name != "nt" and folder.startswith(root.rstrip("/")):
                    folder = folder.replace(root, "").lstrip("/")
                # These three are for Windows
                elif os.path.isdir(folder) and os.path.basename(folder) in [
                    "dpv",
                    "dps",
                    "groups",
                ]:
                    folder = os.path.basename(folder)
                elif os.path.basename(os.path.dirname(folder)) == "dpg":
                    folder = os.path.join("dpg", os.path.basename(folder))
                else:
                    folder = ""

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
                if size != trx.header["NB_STREAMLINES"] + 1 or dim != 1:
                    raise ValueError("Wrong offsets size/dimensionality.")
                offsets = _create_memmap(
                    filename,
                    mode="r+",
                    offset=mem_adress,
                    shape=(trx.header["NB_STREAMLINES"] + 1,),
                    dtype=ext[1:],
                )
                if offsets[-1] != 0:
                    lengths = _compute_lengths(offsets)
                else:
                    lengths = [0]
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
                    "{} is not part of a valid structure.".format(elem_filename)
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

    def resize(  # noqa: C901
        self,
        nb_streamlines: Optional[int] = None,
        nb_vertices: Optional[int] = None,
        delete_dpg: bool = False,
    ) -> None:
        """Remove the unused portion of preallocated memmaps.

        Parameters
        ----------
        nb_streamlines : int, optional
            The number of streamlines to keep.
        nb_vertices : int, optional
            The number of vertices to keep.
        delete_dpg : bool, optional
            Remove data_per_group when resizing. Default is False.
        """
        if not self._copy_safe:
            raise ValueError("Cannot resize a sliced datasets.")

        strs_end, pts_end = self._get_real_len()

        if nb_streamlines is not None and nb_streamlines < strs_end:
            strs_end = nb_streamlines
            logging.info(
                "Resizing (down) memmaps, less streamlines than it actually contains."
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

        trx = self._initialize_empty_trx(nb_streamlines, nb_vertices, init_as=self)

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
                "{} group went from {} items to {}".format(group_key, ori_len, len(tmp))
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

    def get_dtype_dict(self):
        """Get the dtype dictionary for the TrxFile.

        Returns
        -------
        dict
            A dictionary containing the dtype for each data element.
        """
        dtype_dict = {
            "positions": self.streamlines._data.dtype,
            "offsets": self.streamlines._offsets.dtype,
            "dpv": {},
            "dps": {},
            "dpg": {},
            "groups": {},
        }

        for key in self.data_per_vertex.keys():
            dtype_dict["dpv"][key] = self.data_per_vertex[key]._data.dtype
        for key in self.data_per_streamline.keys():
            dtype_dict["dps"][key] = self.data_per_streamline[key].dtype

        for group_key in self.data_per_group.keys():
            dtype_dict["groups"][group_key] = self.groups[group_key].dtype

        for group_key in self.data_per_group.keys():
            dtype_dict["dpg"][group_key] = {}
            for dpg_key in self.data_per_group[group_key].keys():
                dtype_dict["dpg"][group_key][dpg_key] = self.data_per_group[group_key][
                    dpg_key
                ].dtype

        return dtype_dict

    def append(self, obj, extra_buffer: int = 0) -> None:
        """Append another tractogram-like object to this TRX.

        Parameters
        ----------
        obj : TrxFile or Tractogram or StatefulTractogram class instance
            Object whose streamlines and associated data will be appended.
        extra_buffer : int, optional
            Additional preallocation buffer for streamlines (in count).

        Returns
        -------
        None
            Mutates the current TrxFile in-place.
        """
        curr_dtype_dict = self.get_dtype_dict()
        if dipy_available:
            from dipy.io.stateful_tractogram import StatefulTractogram

        if not isinstance(obj, (TrxFile, Tractogram)) and (
            dipy_available and not isinstance(obj, StatefulTractogram)
        ):
            raise TypeError(
                "{} is not a supported object type for appending.".format(type(obj))
            )
        elif isinstance(obj, Tractogram):
            obj = self.from_tractogram(
                obj, reference=self.header, dtype_dict=curr_dtype_dict
            )
        elif dipy_available and isinstance(obj, StatefulTractogram):
            obj = self.from_sft(obj, dtype_dict=curr_dtype_dict)

        self._append_trx(obj, extra_buffer=extra_buffer)

    def _append_trx(self, trx: Type["TrxFile"], extra_buffer: int = 0) -> None:
        """Append a TrxFile to another (with buffer support).

        Parameters
        ----------
        trx : TrxFile
            The TrxFile to append to the current TrxFile.
        extra_buffer : int, optional
            The additional buffer space required to append data. Default is 0.
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
        """Get a particular group from the TrxFile.

        Parameters
        ----------
        key : str
            The group name to select.
        keep_group : bool, optional
            Make sure group exists in returned TrxFile. Default is True.
        copy_safe : bool, optional
            Perform a deepcopy. Default is False.

        Returns
        -------
        TrxFile
            A TrxFile exclusively containing data from said group.
        """
        return self.select(self.groups[key], keep_group=keep_group, copy_safe=copy_safe)

    def select(
        self, indices: np.ndarray, keep_group: bool = True, copy_safe: bool = False
    ) -> Type["TrxFile"]:
        """Get a subset of items, always pointing to the same memmaps.

        Parameters
        ----------
        indices : np.ndarray
            The list of indices of elements to return.
        keep_group : bool, optional
            Ensure group is returned in output TrxFile. Default is True.
        copy_safe : bool, optional
            Perform a deep-copy. Default is False.

        Returns
        -------
        TrxFile
            A TrxFile containing data originating from the selected indices.
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
            new_trx.header["NB_STREAMLINES"] = len(new_trx.streamlines._lengths)

            return new_trx.deepcopy() if copy_safe else new_trx

        new_trx.streamlines = (
            self.streamlines[indices].copy() if copy_safe else self.streamlines[indices]
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
            logging.warning("Keeping dpg despite affecting the group items.")
            for group_key in self.groups.keys():
                # Keep the group indices even when fancy slicing
                index = np.argsort(indices)
                sorted_x = indices[index]
                sorted_index = np.searchsorted(sorted_x, self.groups[group_key])
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
                        new_trx.data_per_group[group_key][dpg_key] = (
                            self.data_per_group[group_key][dpg_key]
                        )

        new_trx.header["NB_VERTICES"] = len(new_trx.streamlines._data)
        new_trx.header["NB_STREAMLINES"] = len(new_trx.streamlines._lengths)
        return new_trx.deepcopy() if copy_safe else new_trx

    @staticmethod
    def from_lazy_tractogram(
        obj: ["LazyTractogram"],
        reference,
        extra_buffer: int = 0,
        chunk_size: int = 10000,
        dtype_dict: dict = None,
    ) -> Type["TrxFile"]:
        """Create a TrxFile from a LazyTractogram with buffer support.

        Parameters
        ----------
        obj : LazyTractogram
            The LazyTractogram to convert.
        reference : object
            Reference for spatial information.
        extra_buffer : int, optional
            The buffer space between reallocation. This number should be a
            number of streamlines. Use 0 for no buffer. Default is 0.
        chunk_size : int, optional
            The number of streamlines to save at a time. Default is 10000.
        dtype_dict : dict, optional
            Dictionary specifying dtypes for positions, offsets, dpv, and dps.

        Returns
        -------
        TrxFile
            A TrxFile created from the LazyTractogram.
        """
        if dtype_dict is None:
            dtype_dict = {
                "positions": np.float32,
                "offsets": np.uint32,
                "dpv": {},
                "dps": {},
            }

        data = {"strs": [], "dpv": {}, "dps": {}}
        concat = None
        count = 0
        iterator = iter(obj)
        while True:
            if count < chunk_size:
                try:
                    i = next(iterator)
                    count += 1
                except StopIteration:
                    obj = convert_data_dict_to_tractogram(data)
                    if concat is None:
                        if len(obj.streamlines) == 0:
                            concat = TrxFile()
                        else:
                            concat = TrxFile.from_tractogram(
                                obj, reference=reference, dtype_dict=dtype_dict
                            )
                    elif len(obj.streamlines) > 0:
                        curr_obj = TrxFile.from_tractogram(
                            obj, reference=reference, dtype_dict=dtype_dict
                        )
                        concat.append(curr_obj)
                    break
                append_generator_to_dict(i, data)
            else:
                obj = convert_data_dict_to_tractogram(data)
                if concat is None:
                    concat = TrxFile.from_tractogram(
                        obj, reference=reference, dtype_dict=dtype_dict
                    )
                else:
                    curr_obj = TrxFile.from_tractogram(
                        obj, reference=reference, dtype_dict=dtype_dict
                    )
                    concat.append(curr_obj, extra_buffer=extra_buffer)
                data = {"strs": [], "dpv": {}, "dps": {}}
                count = 0

        concat.resize()
        return concat

    @staticmethod
    def from_sft(sft, dtype_dict=None):
        """Generate a TrxFile from a StatefulTractogram.

        Parameters
        ----------
        sft : StatefulTractogram class instance
            Input tractogram.
        dtype_dict : dict or None, optional
            Mapping of target dtypes for positions, offsets, dpv, and dps. When
            None, uses ``sft.dtype_dict`` or sensible defaults.

        Returns
        -------
        TrxFile
            TRX representation of the StatefulTractogram.
        """
        if dtype_dict is None:
            dtype_dict = {}

        if len(sft.dtype_dict) > 0:
            dtype_dict = sft.dtype_dict
        if "dpp" in dtype_dict:
            dtype_dict["dpv"] = dtype_dict.pop("dpp")
        elif len(dtype_dict) == 0:
            dtype_dict = {
                "positions": np.float32,
                "offsets": np.uint32,
                "dpv": {},
                "dps": {},
            }

        positions_dtype = dtype_dict["positions"]
        offsets_dtype = dtype_dict["offsets"]

        if not np.issubdtype(positions_dtype, np.floating):
            logging.warning(
                "Casting positions as {}, considering using a floating point "
                "dtype.".format(positions_dtype)
            )

        if not np.issubdtype(offsets_dtype, np.integer):
            logging.warning(
                "Casting offsets as {}, considering using a integer dtype.".format(
                    offsets_dtype
                )
            )

        trx = TrxFile(
            nb_vertices=len(sft.streamlines._data), nb_streamlines=len(sft.streamlines)
        )
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

        tmp_streamlines = deepcopy(sft.streamlines)

        # Cast the int64 of Nibabel to uint32
        tmp_streamlines._offsets = tmp_streamlines._offsets.astype(offsets_dtype)
        tmp_streamlines._data = tmp_streamlines._data.astype(positions_dtype)

        trx.streamlines = tmp_streamlines
        for key in sft.data_per_point:
            dtype_to_use = (
                dtype_dict["dpv"][key] if key in dtype_dict["dpv"] else np.float32
            )
            trx.data_per_vertex[key] = sft.data_per_point[key]
            trx.data_per_vertex[key]._data = sft.data_per_point[key]._data.astype(
                dtype_to_use
            )

        for key in sft.data_per_streamline:
            dtype_to_use = (
                dtype_dict["dps"][key] if key in dtype_dict["dps"] else np.float32
            )
            trx.data_per_streamline[key] = sft.data_per_streamline[key].astype(
                dtype_to_use
            )

        # For safety and for RAM, convert the whole object to memmaps
        tmp_dir = get_trx_tmp_dir()
        save(trx, tmp_dir.name)
        trx.close()
        trx = load_from_directory(tmp_dir.name)
        trx._uncompressed_folder_handle = tmp_dir

        sft.to_space(old_space)
        sft.to_origin(old_origin)
        del tmp_streamlines

        return trx

    @staticmethod
    def from_tractogram(
        tractogram,
        reference,
        dtype_dict=None,
    ):
        """Generate a TrxFile from a nibabel Tractogram.

        Parameters
        ----------
        tractogram : nibabel.streamlines.Tractogram class instance
            Input tractogram to convert.
        reference : object
            Reference anatomy used to populate header fields.
        dtype_dict : dict or None, optional
            Mapping of target dtypes for positions, offsets, dpv, and dps.

        Returns
        -------
        TrxFile class instance
            TRX representation of the tractogram.
        """
        if dtype_dict is None:
            dtype_dict = {
                "positions": np.float32,
                "offsets": np.uint32,
                "dpv": {},
                "dps": {},
            }

        positions_dtype = (
            dtype_dict["positions"] if "positions" in dtype_dict else np.float32
        )
        offsets_dtype = dtype_dict["offsets"] if "offsets" in dtype_dict else np.uint32

        if not np.issubdtype(positions_dtype, np.floating):
            logging.warning(
                "Casting positions as {}, considering using a floating point "
                "dtype.".format(positions_dtype)
            )

        if not np.issubdtype(offsets_dtype, np.integer):
            logging.warning(
                "Casting offsets as {}, considering using a integer dtype.".format(
                    offsets_dtype
                )
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

        tmp_streamlines = deepcopy(tractogram.streamlines)

        # Cast the int64 of Nibabel to uint32
        tmp_streamlines._offsets = tmp_streamlines._offsets.astype(offsets_dtype)
        tmp_streamlines._data = tmp_streamlines._data.astype(positions_dtype)

        trx.streamlines = tmp_streamlines
        for key in tractogram.data_per_point:
            dtype_to_use = (
                dtype_dict["dpv"][key] if key in dtype_dict["dpv"] else np.float32
            )
            trx.data_per_vertex[key] = tractogram.data_per_point[key]
            trx.data_per_vertex[key]._data = tractogram.data_per_point[
                key
            ]._data.astype(dtype_to_use)

        for key in tractogram.data_per_streamline:
            dtype_to_use = (
                dtype_dict["dps"][key] if key in dtype_dict["dps"] else np.float32
            )
            trx.data_per_streamline[key] = tractogram.data_per_streamline[key].astype(
                dtype_to_use
            )

        # For safety and for RAM, convert the whole object to memmaps
        tmp_dir = get_trx_tmp_dir()
        save(trx, tmp_dir.name)
        trx.close()

        trx = load_from_directory(tmp_dir.name)
        del tmp_streamlines

        return trx

    def to_tractogram(self, resize=False):
        """Convert this TrxFile to a nibabel Tractogram.

        Parameters
        ----------
        resize : bool, optional
            If True, resize to actual data length before conversion.

        Returns
        -------
        nibabel.streamlines.Tractogram class instance
            Tractogram containing streamlines and metadata.
        """
        if resize:
            self.resize()

        trx_obj = self.to_memory()
        tractogram = nib.streamlines.Tractogram([], affine_to_rasmm=np.eye(4))
        tractogram._set_streamlines(trx_obj.streamlines)
        tractogram._data_per_point = trx_obj.data_per_vertex
        tractogram._data_per_streamline = trx_obj.data_per_streamline

        return tractogram

    def to_memory(self, resize: bool = False) -> Type["TrxFile"]:
        """Convert a TrxFile to a RAM representation.

        Parameters
        ----------
        resize : bool, optional
            Resize TrxFile when converting to RAM representation.
            Default is False.

        Returns
        -------
        TrxFile
            A non memory-mapped TrxFile.
        """
        if resize:
            self.resize()

        trx_obj = TrxFile()
        trx_obj.header = deepcopy(self.header)
        trx_obj.streamlines = deepcopy(self.streamlines)

        for key in self.data_per_vertex:
            trx_obj.data_per_vertex[key] = deepcopy(self.data_per_vertex[key])

        for key in self.data_per_streamline:
            trx_obj.data_per_streamline[key] = deepcopy(self.data_per_streamline[key])

        for key in self.groups:
            trx_obj.groups[key] = deepcopy(self.groups[key])

        for key in self.data_per_group:
            trx_obj.data_per_group[key] = deepcopy(self.data_per_group[key])

        return trx_obj

    def to_sft(self, resize=False):
        """Convert this TrxFile to a StatefulTractogram.

        Parameters
        ----------
        resize : bool, optional
            If True, resize to actual data length before conversion.

        Returns
        -------
        StatefulTractogram class instance or None
            StatefulTractogram object, or None if dipy is unavailable.
        """
        try:
            from dipy.io.stateful_tractogram import Space, StatefulTractogram
        except ImportError:
            logging.error(
                "Dipy library is missing, cannot convert to StatefulTractogram."
            )
            return None

        affine = np.array(self.header["VOXEL_TO_RASMM"], dtype=np.float32)
        dimensions = np.array(self.header["DIMENSIONS"], dtype=np.uint16)
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = "".join(aff2axcodes(affine))
        space_attributes = (affine, dimensions, vox_sizes, vox_order)

        if resize:
            self.resize()
        sft = StatefulTractogram(
            deepcopy(self.streamlines),
            space_attributes,
            Space.RASMM,
            data_per_point=deepcopy(self.data_per_vertex),
            data_per_streamline=deepcopy(self.data_per_streamline),
        )
        tmp_dict = self.get_dtype_dict()
        if "dpv" in tmp_dict:
            tmp_dict["dpp"] = tmp_dict.pop("dpv")
        sft.dtype_dict = self.get_dtype_dict()

        return sft

    def close(self) -> None:
        """Cleanup on-disk temporary folder and memmaps.

        Returns
        -------
        None
            Releases file handles and removes temporary storage.
        """
        if self._uncompressed_folder_handle is not None:
            close_or_delete_mmap(self.streamlines)

            # # Close or delete attributes in dictionaries
            for key in self.data_per_vertex:
                close_or_delete_mmap(self.data_per_vertex[key])

            for key in self.data_per_streamline:
                close_or_delete_mmap(self.data_per_streamline[key])

            for key in self.groups:
                close_or_delete_mmap(self.groups[key])

            for key in self.data_per_group:
                for dpg in self.data_per_group[key]:
                    close_or_delete_mmap(self.data_per_group[key][dpg])

            try:
                self._uncompressed_folder_handle.cleanup()
            except PermissionError:
                logging.error(
                    "Windows PermissionError, temporary directory %s was not deleted!",
                    self._uncompressed_folder_handle.name,
                )
        self.__init__()
        logging.debug("Deleted memmaps and initialized empty TrxFile.")
