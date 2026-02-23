# -*- coding: utf-8 -*-

import json
import os
import struct
import tempfile
import zipfile

from nibabel.streamlines import LazyTractogram
from nibabel.streamlines.tests.test_tractogram import make_dummy_streamline
import numpy as np
import pytest

try:
    import dipy  # noqa: F401

    dipy_available = True
except ImportError:
    dipy_available = False

from trx.fetcher import fetch_data, get_home, get_testing_files_dict
from trx.io import get_trx_tmp_dir
import trx.trx_file_memmap as tmm

fetch_data(get_testing_files_dict(), keys=["memmap_test_data.zip"])
tmp_dir = get_trx_tmp_dir()


@pytest.mark.parametrize(
    "arr,expected,value_error",
    [
        (np.ones((5, 5, 5), dtype=np.int16), None, True),
        (np.ones((5, 4), dtype=np.int16), "mean_fa.4.int16", False),
        (np.ones((5, 4), dtype=np.float64), "mean_fa.4.float64", False),
        (np.ones((5, 1), dtype=np.float64), "mean_fa.float64", False),
        (np.ones((1), dtype=np.float64), "mean_fa.float64", False),
    ],
)
def test__generate_filename_from_data(
    arr, expected, value_error, filename="mean_fa.bit"
):
    if value_error:
        with pytest.raises(ValueError):
            new_fn = tmm._generate_filename_from_data(arr=arr, filename=filename)
            assert new_fn is None
    else:
        new_fn = tmm._generate_filename_from_data(arr=arr, filename=filename)
        assert new_fn == expected


@pytest.mark.parametrize(
    "filename,expected,value_error",
    [
        ("mean_fa.float64", ("mean_fa", 1, ".float64"), False),
        ("mean_fa.5.int32", ("mean_fa", 5, ".int32"), False),
        ("mean_fa", None, True),
        ("mean_fa.5.4.int32", None, True),
        pytest.param(
            "mean_fa.fa", None, True, marks=pytest.mark.xfail, id="invalid extension"
        ),
    ],
)
def test__split_ext_with_dimensionality(filename, expected, value_error):
    if value_error:
        with pytest.raises(ValueError):
            assert tmm._split_ext_with_dimensionality(filename) == expected
    else:
        assert tmm._split_ext_with_dimensionality(filename) == expected


@pytest.mark.parametrize(
    "offsets,nb_vertices,expected",
    [
        (np.array(range(5), dtype=np.int16), 4, np.array([1, 1, 1, 1, 0])),
        (np.array([0, 1, 1, 3, 4], dtype=np.int32), 4, np.array([1, 0, 2, 1, 0])),
        (np.array(range(4), dtype=np.uint64), 4, np.array([1, 1, 1, 1])),
        pytest.param(
            np.array([0, 1, 0, 3, 4], dtype=np.int16),
            4,
            np.array([1, 3, 0, 1, 0]),
            marks=pytest.mark.xfail,
            id="offsets not sorted",
        ),
    ],
)
def test__compute_lengths(offsets, nb_vertices, expected):
    offsets = tmm._append_last_offsets(offsets, nb_vertices)
    lengths = tmm._compute_lengths(offsets=offsets)
    assert np.array_equal(lengths, expected)


@pytest.mark.parametrize(
    "ext,expected",
    [
        (".bit", True),
        (".int16", True),
        (".float32", True),
        (".ushort", True),
        (".txt", False),
    ],
)
def test__is_dtype_valid(ext, expected):
    assert tmm._is_dtype_valid(ext) == expected


@pytest.mark.parametrize(
    "arr,l_bound,r_bound,expected",
    [
        (np.array(range(5), dtype=np.int16), None, None, 4),
        (np.array([0, 1, 0, 3, 4], dtype=np.int16), None, None, 1),
        (np.array([0, 1, 2, 0, 4], dtype=np.int16), None, None, 2),
        (np.array(range(5), dtype=np.int16), 1, 2, 2),
        (np.array(range(5), dtype=np.int16), 3, 3, 3),
        (np.zeros((5), dtype=np.int16), 3, 3, -1),
    ],
)
def test__dichotomic_search(arr, l_bound, r_bound, expected):
    end_idx = tmm._dichotomic_search(arr, l_bound=l_bound, r_bound=r_bound)
    assert end_idx == expected


@pytest.mark.parametrize(
    "basename, create, expected",
    [
        ("offsets.int16", True, np.array(range(12), dtype=np.int16).reshape((3, 4))),
        ("offsets.float32", False, None),
    ],
)
def test__create_memmap(basename, create, expected):
    if create:
        with get_trx_tmp_dir() as dirname:
            filename = os.path.join(dirname, basename)
            fp = tmm._create_memmap(
                filename=filename, mode="w+", shape=(3, 4), dtype=np.int16
            )
            fp[:] = expected[:]
            mmarr = tmm._create_memmap(filename=filename, shape=(3, 4), dtype=np.int16)
            assert np.array_equal(mmarr, expected)

    else:
        with get_trx_tmp_dir() as dirname:
            filename = os.path.join(dirname, basename)
            mmarr = tmm._create_memmap(filename=filename, shape=(0,), dtype=np.int16)
            assert os.path.isfile(filename)
            assert np.array_equal(mmarr, np.zeros(shape=(0,), dtype=np.float32))


# need dpg test with missing keys
@pytest.mark.parametrize(
    "path,check_dpg,value_error",
    [
        ("small_compressed.trx", False, False),
        ("small.trx", True, False),
        ("small_fldr.trx", False, False),
        ("dontexist.trx", False, True),
    ],
)
def test_load(path, check_dpg, value_error):
    path = os.path.join(get_home(), "memmap_test_data", path)
    # Need to perhaps improve test
    if value_error:
        with pytest.raises(ValueError):
            assert not isinstance(
                tmm.load(input_obj=path, check_dpg=check_dpg), tmm.TrxFile
            )
    else:
        assert isinstance(tmm.load(input_obj=path, check_dpg=check_dpg), tmm.TrxFile)


@pytest.mark.parametrize("path", [("small.trx")])
def test_load_zip(path):
    path = os.path.join(get_home(), "memmap_test_data", path)
    assert isinstance(tmm.load_from_zip(path), tmm.TrxFile)


@pytest.mark.parametrize("path", [("small_fldr.trx")])
def test_load_directory(path):
    path = os.path.join(get_home(), "memmap_test_data", path)
    assert isinstance(tmm.load_from_directory(path), tmm.TrxFile)


@pytest.mark.parametrize("path", [("small.trx")])
def test_concatenate(path):
    path = os.path.join(get_home(), "memmap_test_data", path)
    trx1 = tmm.load(path)
    trx2 = tmm.load(path)
    concat = tmm.concatenate([trx1, trx2])

    assert len(concat) == 2 * len(trx2)
    trx1.close()
    trx2.close()
    concat.close()


@pytest.mark.parametrize("path", [("small.trx")])
def test_resize(path):
    path = os.path.join(get_home(), "memmap_test_data", path)
    trx1 = tmm.load(path)
    concat = tmm.TrxFile(nb_vertices=1000000, nb_streamlines=10000, init_as=trx1)

    tmm.concatenate([concat, trx1], preallocation=True, delete_groups=True)
    concat.resize()

    assert len(concat) == len(trx1)
    trx1.close()
    concat.close()


@pytest.mark.parametrize("path, buffer", [("small.trx", 10000), ("small.trx", 0)])
def test_append(path, buffer):
    path = os.path.join(get_home(), "memmap_test_data", path)
    trx1 = tmm.load(path)
    concat = tmm.TrxFile(nb_vertices=1, nb_streamlines=1, init_as=trx1)

    concat.append(trx1, extra_buffer=buffer)
    if buffer > 0:
        concat.resize()

    assert len(concat) == len(trx1)
    trx1.close()
    concat.close()


@pytest.mark.parametrize("path, buffer", [("small.trx", 10000)])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed")
def test_append_StatefulTractogram(path, buffer):
    path = os.path.join(get_home(), "memmap_test_data", path)
    trx = tmm.load(path)
    obj = trx.to_sft()
    concat = tmm.TrxFile(nb_vertices=1, nb_streamlines=1, init_as=trx)

    concat.append(obj, extra_buffer=buffer)
    if buffer > 0:
        concat.resize()

    assert len(concat) == len(obj)
    trx.close()
    concat.close()


@pytest.mark.parametrize("path, buffer", [("small.trx", 10000)])
def test_append_Tractogram(path, buffer):
    path = os.path.join(get_home(), "memmap_test_data", path)
    trx = tmm.load(path)
    obj = trx.to_tractogram()
    concat = tmm.TrxFile(nb_vertices=1, nb_streamlines=1, init_as=trx)

    concat.append(obj, extra_buffer=buffer)
    if buffer > 0:
        concat.resize()

    assert len(concat) == len(obj)
    trx.close()
    concat.close()


@pytest.mark.parametrize(
    "path, size, buffer",
    [
        ("small.trx", 50, 10000),
        ("small.trx", 0, 10000),
        ("small.trx", 25000, 10000),
        ("small.trx", 50, 0),
        ("small.trx", 0, 0),
        ("small.trx", 25000, 10000),
    ],
)
def test_from_lazy_tractogram(path, size, buffer):
    _ = np.random.RandomState(1776)
    streamlines = []
    fa = []
    commit_weights = []
    clusters_QB = []
    gen_range = [1, 2, 5, 2, 1] * (size // 5)
    for i in gen_range:
        data = make_dummy_streamline(i)
        streamline, data_per_point, data_for_streamline = data
        streamlines.append(streamline)
        fa.append(data_per_point["fa"].astype(np.float16))
        commit_weights.append(data_for_streamline["mean_curvature"].astype(np.float32))
        clusters_QB.append(data_for_streamline["mean_torsion"].astype(np.uint16))

    def streamlines_func():
        return (e for e in streamlines)

    data_per_point_func = {"fa": lambda: (e for e in fa)}
    data_per_streamline_func = {
        "commit_weights": lambda: (e for e in commit_weights),
        "clusters_QB": lambda: (e for e in clusters_QB),
    }

    obj = LazyTractogram(
        streamlines_func,
        data_per_streamline_func,
        data_per_point_func,
        affine_to_rasmm=np.eye(4),
    )

    dtype_dict = {
        "positions": np.float32,
        "offsets": np.uint32,
        "dpv": {"fa": np.float16},
        "dps": {"commit_weights": np.float32, "clusters_QB": np.uint16},
    }
    path = os.path.join(get_home(), "memmap_test_data", path)
    trx = tmm.TrxFile.from_lazy_tractogram(
        obj, reference=path, extra_buffer=buffer, chunk_size=1000, dtype_dict=dtype_dict
    )

    assert len(trx) == len(gen_range)


def test_zip_from_folder():
    pass


def test_trxfile_init():
    pass


def test_trxfile_print():
    pass


def test_trxfile_len():
    fake = tmm.TrxFile(nb_vertices=100, nb_streamlines=10)
    assert len(fake) == 10


def test_trxfile_getitem():
    pass


def test_trxfile_deepcopy():
    pass


def test_get_real_len():
    fake = tmm.TrxFile(nb_vertices=100, nb_streamlines=10)
    assert fake._get_real_len() == (0, 0)


def test_copy_fixed_arrays_from():
    pass


def test_initialize_empty_trx():
    """Test creating, saving, and loading an empty TRX file."""
    trx = tmm.TrxFile()
    assert trx.header["NB_STREAMLINES"] == 0
    assert trx.header["NB_VERTICES"] == 0
    assert len(trx.streamlines) == 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = os.path.join(tmp_dir, "empty.trx")
        tmm.save(trx, out_path)

        assert os.path.exists(out_path)
        file_size = os.path.getsize(out_path)
        assert file_size < 500  # Should be very small, just header.json in zip

        with zipfile.ZipFile(out_path, "r") as zf:
            filenames = [info.filename for info in zf.filelist]
            assert "header.json" in filenames
            positions_files = [f for f in filenames if f.startswith("positions")]
            offsets_files = [f for f in filenames if f.startswith("offsets")]
            assert len(positions_files) == 0
            assert len(offsets_files) == 0

        loaded_trx = tmm.load(out_path)
        assert loaded_trx.header["NB_STREAMLINES"] == 0
        assert loaded_trx.header["NB_VERTICES"] == 0
        assert len(loaded_trx.streamlines) == 0
        assert len(loaded_trx.groups) == 0
        assert len(loaded_trx.data_per_streamline) == 0
        assert len(loaded_trx.data_per_vertex) == 0
        assert len(loaded_trx.data_per_group) == 0
        loaded_trx.close()


def test_create_trx_from_pointer():
    pass


def test_trxfile_getgroup():
    pass


def test_trxfile_select():
    pass


def test_trxfile_to_memory():
    pass


def test_trxfile_close():
    pass


# Endianness tests for cross-platform compatibility (Issue #83)
@pytest.mark.parametrize(
    "dtype_input,expected_byteorder",
    [
        # Native dtypes should be converted to little-endian
        (np.float32, "<"),
        (np.float64, "<"),
        (np.int32, "<"),
        (np.int64, "<"),
        (np.uint32, "<"),
        (np.uint64, "<"),
        ("float32", "<"),
        ("float64", "<"),
        # Big-endian dtypes should be converted to little-endian
        (">f4", "<"),
        (">f8", "<"),
        (">i4", "<"),
        (">u4", "<"),
        # Little-endian dtypes should remain little-endian
        ("<f4", "<"),
        ("<i4", "<"),
        # Single-byte types don't have endianness (byteorder is '|')
        (np.uint8, "|"),
        (np.int8, "|"),
        (np.bool_, "|"),
    ],
)
def test__get_dtype_little_endian(dtype_input, expected_byteorder):
    """Test that _get_dtype_little_endian correctly converts dtypes."""
    result = tmm._get_dtype_little_endian(dtype_input)
    assert result.byteorder == expected_byteorder or (
        result.byteorder == "=" and np.little_endian and expected_byteorder == "<"
    )


@pytest.mark.parametrize(
    "dtype,test_value",
    [
        (np.float32, 3.14159),
        (np.float64, 2.71828),
        (np.int32, 12345),
        (np.int64, 9876543210),
        (np.uint32, 0xDEADBEEF),
        (np.uint64, 0xDEADBEEFCAFEBABE),
    ],
)
def test__ensure_little_endian(dtype, test_value):
    """Test that _ensure_little_endian correctly converts arrays."""
    # Create array in native byte order
    arr = np.array([test_value], dtype=dtype)

    # Ensure little endian
    result = tmm._ensure_little_endian(arr)

    # Result should be little-endian (or native if system is little-endian)
    assert result.dtype.byteorder in ("<", "=", "|")

    # Values should be preserved (use isclose for float types due to precision)
    if np.issubdtype(dtype, np.floating):
        assert np.isclose(result[0], test_value, rtol=1e-6)
    else:
        assert result[0] == test_value


def test__ensure_little_endian_big_endian_input():
    """Test _ensure_little_endian with explicitly big-endian input."""
    # Create a big-endian array
    big_endian_dtype = np.dtype(">u4")
    arr = np.array([0x12345678], dtype=big_endian_dtype)

    # Ensure little endian
    result = tmm._ensure_little_endian(arr)

    # Result should be little-endian
    assert result.dtype.byteorder == "<"

    # Value should be preserved
    assert result[0] == 0x12345678


def test_load_zip_with_local_header_extra_field():
    """Test loading ZIP where local header has extra field not in central dir.

    Regression test for a bug where zip_info.FileHeader() was used to calculate
    data offset. The ZIP spec allows local headers to have different extra
    fields than central directory entries. The fix reads the actual local
    file header to get the correct offset.
    """
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    offsets = np.array([0, 2], dtype=np.uint64)
    header = {
        "DIMENSIONS": [10, 10, 10],
        "VOXEL_TO_RASMM": np.eye(4).tolist(),
        "NB_VERTICES": 2,
        "NB_STREAMLINES": 1,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        trx_path = os.path.join(tmp_dir, "test.trx")

        # Build ZIP with extra bytes in local headers but not central directory
        with open(trx_path, "wb") as f:
            local_info = []
            extra = b"\x00\x00\x04\x00TEST"  # 8-byte extra field

            for name, data in [
                ("header.json", json.dumps(header).encode()),
                ("positions.3.float32", positions.tobytes()),
                ("offsets.uint64", offsets.tobytes()),
            ]:
                offset = f.tell()
                fname = name.encode()
                crc = zipfile.crc32(data)
                # Local header WITH extra field
                f.write(
                    struct.pack(
                        "<4sHHHHHIIIHH",
                        b"PK\x03\x04",
                        20,
                        0,
                        0,
                        0,
                        0,
                        crc,
                        len(data),
                        len(data),
                        len(fname),
                        len(extra),
                    )
                )
                f.write(fname)
                f.write(extra)
                f.write(data)
                local_info.append((name, offset, crc, len(data)))

            cd_start = f.tell()
            for name, offset, crc, size in local_info:
                fname = name.encode()
                # Central directory WITHOUT extra field (mismatch!)
                f.write(
                    struct.pack(
                        "<4sHHHHHHIIIHHHHHII",
                        b"PK\x01\x02",
                        20,
                        20,
                        0,
                        0,
                        0,
                        0,
                        crc,
                        size,
                        size,
                        len(fname),
                        0,
                        0,
                        0,
                        0,
                        0,
                        offset,
                    )
                )
                f.write(fname)

            # End of central directory
            f.write(
                struct.pack(
                    "<4sHHHHIIH",
                    b"PK\x05\x06",
                    0,
                    0,
                    3,
                    3,
                    f.tell() - cd_start,
                    cd_start,
                    0,
                )
            )

        trx = tmm.load_from_zip(trx_path)
        np.testing.assert_array_almost_equal(trx.streamlines._data, positions)
        assert trx.header["NB_VERTICES"] == 2
        assert trx.header["NB_STREAMLINES"] == 1

        trx.close()


def test_endianness_roundtrip():
    """Test that data survives write/read cycle with correct endianness."""
    with get_trx_tmp_dir() as dirname:
        # Test values that would be corrupted if endianness is wrong
        test_positions = np.array(
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32
        )
        test_offsets = np.array([0, 3], dtype=np.uint32)

        # Write as little-endian
        pos_file = os.path.join(dirname, "test_positions.3.float32")
        off_file = os.path.join(dirname, "test_offsets.uint32")

        tmm._ensure_little_endian(test_positions).tofile(pos_file)
        tmm._ensure_little_endian(test_offsets).tofile(off_file)

        # Read back using _create_memmap (which enforces little-endian)
        read_positions = tmm._create_memmap(
            pos_file, mode="r", shape=(3, 3), dtype="float32"
        )
        read_offsets = tmm._create_memmap(
            off_file, mode="r", shape=(2,), dtype="uint32"
        )

        # Values should match
        np.testing.assert_array_almost_equal(read_positions, test_positions)
        np.testing.assert_array_equal(read_offsets, test_offsets)
