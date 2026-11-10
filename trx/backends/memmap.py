# -*- coding: utf-8 -*-
"""The TRX specification's native container: a zip file or folder of memmaps.

This module holds the code that used to live in :mod:`trx.trx_file_memmap`
for walking a ``.trx`` archive or directory, memory-mapping each array and
assembling a :class:`~trx.trx_file_memmap.TrxFile`, and for writing one back
out. The behaviour is unchanged; it has only moved behind the
:class:`~trx.backends.base.TrxBackend` protocol so that other containers can
sit beside it.

The functions ``load_from_zip``, ``load_from_directory`` and
``zip_from_folder`` keep their names and are re-exported from
:mod:`trx.trx_file_memmap`.
"""

import json
import logging
import os
import shutil
import struct
from typing import Any, Dict, Optional, Tuple
import zipfile

from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np

from trx.backends.base import register_backend
from trx.io import get_trx_tmp_dir

_ZIP_LOCAL_HEADER_SIZE = 30
_ZIP_LOCAL_HEADER_SIGNATURE = b"PK\x03\x04"


def _core():
    """Import :mod:`trx.trx_file_memmap` lazily to avoid a circular import."""
    from trx import trx_file_memmap

    return trx_file_memmap


def _read_header(fobj) -> dict:
    """Parse ``header.json`` into the dict shape ``TrxFile`` expects."""
    header = json.load(fobj)
    header["VOXEL_TO_RASMM"] = np.reshape(header["VOXEL_TO_RASMM"], (4, 4)).astype(
        np.float32
    )
    header["DIMENSIONS"] = np.array(header["DIMENSIONS"], dtype=np.uint16)
    return header


def load_from_zip(filename: str):
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
    tmm = _core()
    with zipfile.ZipFile(filename, mode="r") as zf:
        with zf.open("header.json") as zf_header:
            header = _read_header(zf_header)

        files_pointer_size = {}
        for zip_info in zf.filelist:
            elem_filename = zip_info.filename
            _, ext = os.path.splitext(elem_filename)
            if ext == ".json" or zip_info.is_dir():
                continue

            if not tmm._is_dtype_valid(ext):
                continue

            if ext == ".bit":
                ext = ".bool"

            # Read actual local file header to get correct data offset.
            # We can't use zip_info.FileHeader() because ZIP spec allows local
            # headers to differ from central directory entries.
            # See: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
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

    return create_trx_from_pointer(header, files_pointer_size, root_zip=filename)


def load_from_directory(directory: str):
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
    tmm = _core()
    directory = os.path.abspath(directory)
    with open(os.path.join(directory, "header.json")) as header:
        header = _read_header(header)
    files_pointer_size = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            elem_filename = os.path.join(root, name)
            _, ext = os.path.splitext(elem_filename)
            if ext == ".json":
                continue

            if not tmm._is_dtype_valid(ext):
                raise ValueError(f"The dtype of {elem_filename} is not supported")

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

    return create_trx_from_pointer(header, files_pointer_size, root=directory)


def create_trx_from_pointer(  # noqa: C901
    header: dict,
    dict_pointer_size: Dict[str, Tuple[int, int]],
    root_zip: Optional[str] = None,
    root: Optional[str] = None,
):
    """Create a TrxFile after reading the structure of a zip/folder.

    Parameters
    ----------
    header : dict
        A TrxFile header dictionary which will be used for the new TrxFile.
    dict_pointer_size : dict
        A dictionary containing the filenames of all the files within the
        TrxFile disk file/folder, mapped to ``(byte offset, element count)``.
    root_zip : str, optional
        The path of the ZipFile pointer.
    root : str, optional
        The dirname of the ZipFile pointer.

    Returns
    -------
    TrxFile
        A TrxFile constructed from the pointer provided.
    """
    tmm = _core()
    tgm = tmm.TrxFile()
    tgm.header = header

    # Handle empty TRX files early - no positions/offsets to load
    if header["NB_STREAMLINES"] == 0 or header["NB_VERTICES"] == 0:
        return tgm

    positions, offsets = None, None
    for elem_filename in dict_pointer_size.keys():
        if root_zip:
            filename = root_zip
        else:
            filename = elem_filename

        folder = os.path.dirname(elem_filename)
        base, dim, ext = tmm._split_ext_with_dimensionality(elem_filename)
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
            if size != tgm.header["NB_VERTICES"] * 3 or dim != 3:
                raise ValueError("Wrong data size/dimensionality.")
            positions = tmm._create_memmap(
                filename,
                mode="r+",
                offset=mem_adress,
                shape=(tgm.header["NB_VERTICES"], 3),
                dtype=ext[1:],
            )
        elif base == "offsets" and folder == "":
            if size != tgm.header["NB_STREAMLINES"] + 1 or dim != 1:
                raise ValueError("Wrong offsets size/dimensionality.")
            offsets = tmm._create_memmap(
                filename,
                mode="r+",
                offset=mem_adress,
                shape=(tgm.header["NB_STREAMLINES"] + 1,),
                dtype=ext[1:],
            )
            if offsets[-1] != 0:
                lengths = tmm._compute_lengths(offsets)
            else:
                lengths = [0]
        elif folder == "dps":
            nb_scalar = size / tgm.header["NB_STREAMLINES"]
            if not nb_scalar.is_integer() or nb_scalar != dim:
                raise ValueError("Wrong dps size/dimensionality.")
            else:
                shape = (tgm.header["NB_STREAMLINES"], int(nb_scalar))

            tgm.data_per_streamline[base] = tmm._create_memmap(
                filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
            )
        elif folder == "dpv":
            nb_scalar = size / tgm.header["NB_VERTICES"]
            if not nb_scalar.is_integer() or nb_scalar != dim:
                raise ValueError("Wrong dpv size/dimensionality.")
            else:
                shape = (tgm.header["NB_VERTICES"], int(nb_scalar))

            tgm.data_per_vertex[base] = tmm._create_memmap(
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
            if sub_folder not in tgm.data_per_group:
                tgm.data_per_group[sub_folder] = {}
            tgm.data_per_group[sub_folder][data_name] = tmm._create_memmap(
                filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
            )
        elif folder == "groups":
            # Groups are simply indices, nothing else
            # TODO Crash if not uint?
            if dim != 1:
                raise ValueError("Wrong group dimensionality.")
            else:
                shape = (int(size),)
            tgm.groups[base] = tmm._create_memmap(
                filename, mode="r+", offset=mem_adress, shape=shape, dtype=ext[1:]
            )
        else:
            logging.error(f"{elem_filename} is not part of a valid structure.")

    # All essential array must be declared
    if positions is not None and offsets is not None:
        tgm.streamlines._data = positions
        tgm.streamlines._offsets = offsets[:-1]
        tgm.streamlines._lengths = lengths
    else:
        raise ValueError("Missing essential data.")

    for dpv_key in tgm.data_per_vertex:
        tmp = tgm.data_per_vertex[dpv_key]
        tgm.data_per_vertex[dpv_key] = ArraySequence()
        tgm.data_per_vertex[dpv_key]._data = tmp
        tgm.data_per_vertex[dpv_key]._offsets = offsets[:-1]
        tgm.data_per_vertex[dpv_key]._lengths = lengths
    return tgm


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
        Default is ZIP_STORED.
    """
    with zipfile.ZipFile(filename, mode="w", compression=compression_standard) as zf:
        for root, _, files in os.walk(directory):
            for name in files:
                curr_filename = os.path.join(root, name)
                tmp_filename = curr_filename.replace(directory, "")[1:]
                zf.write(curr_filename, tmp_filename)


@register_backend
class MemmapBackend:
    """Zip-or-folder TRX container with memory-mapped arrays (the TRX spec)."""

    name = "memmap"

    @classmethod
    def can_load(cls, path: str) -> bool:
        """A regular file (zip) or a directory holding ``header.json``."""
        if os.path.isfile(path):
            return zipfile.is_zipfile(path)
        if os.path.isdir(path):
            return os.path.isfile(os.path.join(path, "header.json"))
        return False

    @classmethod
    def can_save(cls, path: str) -> bool:
        """``.trx``, ``.zip`` or an extension-less folder path."""
        _, ext = os.path.splitext(path)
        return ext.lower() in [".zip", ".trx", ""]

    @classmethod
    def load(cls, path: str):
        """Load a TrxFile (compressed or not) from a zip file or a folder.

        A deflated zip is extracted to a temporary directory first; the
        returned object keeps a handle to it, so call ``close()`` when done.
        """
        if os.path.isfile(path):
            was_compressed = False
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    if info.compress_type != 0:
                        was_compressed = True
                        break
            if was_compressed:
                with zipfile.ZipFile(path, "r") as zf:
                    tmp_dir = get_trx_tmp_dir()
                    zf.extractall(tmp_dir.name)
                    tgm = load_from_directory(tmp_dir.name)
                    tgm._uncompressed_folder_handle = tmp_dir
                    logging.info(
                        "File was compressed, call the close() function before exiting."
                    )
            else:
                tgm = load_from_zip(path)
        elif os.path.isdir(path):
            tgm = load_from_directory(path)
        else:
            raise ValueError("File/Folder does not exist")
        return tgm

    @classmethod
    def save(
        cls, tgm, path: str, compression_standard: Any = zipfile.ZIP_STORED, **_: Any
    ) -> None:
        """Write ``tgm`` as a zip (``.trx``/``.zip``) or as a plain folder."""
        _, ext = os.path.splitext(path)
        if ext.lower() not in [".zip", ".trx", ""]:
            raise ValueError("Unsupported extension.")

        copy_trx = tgm.deepcopy()
        copy_trx.resize()
        tmp_dir_name = copy_trx._uncompressed_folder_handle.name
        if ext.lower() in [".zip", ".trx"]:
            zip_from_folder(tmp_dir_name, path, compression_standard)
        else:
            if os.path.isdir(path):
                shutil.rmtree(path)
            shutil.copytree(tmp_dir_name, path)
        copy_trx.close()
