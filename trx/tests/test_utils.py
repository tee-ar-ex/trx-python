# -*- coding: utf-8 -*-
"""Tests for utility functions in trx.utils."""

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import Tractogram, TractogramItem
import numpy as np
import pytest

from trx.utils import (
    append_generator_to_dict,
    close_or_delete_mmap,
    convert_data_dict_to_tractogram,
    flip_sft,
    get_axis_flip_vector,
    get_axis_shift_vector,
    get_reference_info_wrapper,
    get_reverse_enum,
    get_shift_vector,
    is_header_compatible,
    load_matrix_in_any_format,
    split_name_with_gz,
    verify_trx_dtype,
)

# Optional dipy import
try:
    import dipy  # noqa: F401
    from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram

    dipy_available = True
except ImportError:  # pragma: no cover
    dipy_available = False

    class Space:
        RASMM = None
        VOXMM = None
        VOX = None

    class Origin:
        NIFTI = None
        TRACKVIS = None


def test_close_or_delete_mmap_np_memmap(tmp_path):
    """Test close_or_delete_mmap with a numpy.memmap."""
    tmp_name = tmp_path / "test.mmap"
    mmap_arr = np.memmap(tmp_name, dtype="float32", mode="w+", shape=(10,))
    close_or_delete_mmap(mmap_arr)
    assert tmp_name.exists()


def test_close_or_delete_mmap_array_sequence(tmp_path):
    """Test close_or_delete_mmap with an ArraySequence."""
    tmp1_name = tmp_path / "test1.mmap"
    tmp2_name = tmp_path / "test2.mmap"
    data = np.memmap(tmp1_name, dtype="float32", mode="w+", shape=(10, 3))
    offsets = np.memmap(tmp2_name, dtype="uint32", mode="w+", shape=(5,))

    seq = ArraySequence()
    seq._data = data
    seq._offsets = offsets
    seq._lengths = np.array([2, 2, 2, 2, 2], dtype="uint32")

    close_or_delete_mmap(seq)

    assert tmp1_name.exists()
    assert tmp2_name.exists()


def test_close_or_delete_mmap_with_mmap_attr():
    """Test close_or_delete_mmap with an object having _mmap attribute."""
    mock_obj = MagicMock()
    mock_mmap = MagicMock()
    mock_obj._mmap = mock_mmap

    close_or_delete_mmap(mock_obj)
    mock_mmap.close.assert_called_once()


def test_close_or_delete_mmap_other_type(caplog):
    """Test close_or_delete_mmap with an unsupported type."""
    with caplog.at_level(logging.DEBUG):
        close_or_delete_mmap("not a memmap")
    assert "Object to be close or deleted must be np.memmap" in caplog.text


@pytest.mark.parametrize(
    "filename,expected_base,expected_ext",
    [
        ("test.nii.gz", "test", ".nii.gz"),
        ("test.trk.gz", "test", ".trk.gz"),
        ("test.nii", "test", ".nii"),
        ("test.trk", "test", ".trk"),
        ("test.txt", "test", ".txt"),
        ("my.file.with.dots.nii.gz", "my.file.with.dots", ".nii.gz"),
        ("no_ext", "no_ext", ""),
    ],
)
def test_split_name_with_gz(filename, expected_base, expected_ext):
    """Test split_name_with_gz with various extensions."""
    base, ext = split_name_with_gz(filename)
    assert base == expected_base
    assert ext == expected_ext


def test_load_matrix_in_any_format_txt():
    """Test loading a matrix from a .txt file."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tmp:
        tmp.write("1 2 3\n4 5 6")
        tmp_name = tmp.name

    try:
        matrix = load_matrix_in_any_format(tmp_name)
        np.testing.assert_allclose(matrix, [[1, 2, 3], [4, 5, 6]])
    finally:
        os.remove(tmp_name)


def test_load_matrix_in_any_format_npy():
    """Test loading a matrix from a .npy file."""
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        data = np.array([[1, 2], [3, 4]])
        np.save(tmp.name, data)
        tmp_name = tmp.name

    try:
        matrix = load_matrix_in_any_format(tmp_name)
        np.testing.assert_array_equal(matrix, data)
    finally:
        os.remove(tmp_name)


def test_load_matrix_in_any_format_error():
    """Test load_matrix_in_any_format with unsupported extension."""
    with pytest.raises(ValueError, match="Extension .invalid is not supported"):
        load_matrix_in_any_format("test.invalid")


# --- Spatial Reference Tests ---


@pytest.fixture
def nifti_ref():
    """Create a synthetic Nifti1Image for testing."""
    data = np.zeros((10, 20, 30), dtype=np.float32)
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    affine[0:3, 3] = [1.1, 2.2, 3.3]
    img = nib.Nifti1Image(data, affine)
    return img


@pytest.fixture
def trk_header():
    """Create a synthetic TRK header for testing."""
    return {
        "voxel_to_rasmm": np.diag([1.0, 2.0, 3.0, 1.0]),
        "dimensions": np.array([10, 20, 30], dtype=np.int16),
        "voxel_sizes": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "voxel_order": "RAS",
        "magic_number": "TRACK",
    }


def test_get_reference_info_wrapper_nifti_obj(nifti_ref):
    """Test get_reference_info_wrapper with a Nifti1Image object."""
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(nifti_ref)
    assert np.allclose(affine, nifti_ref.affine)
    assert np.array_equal(dimensions, [10, 20, 30])
    assert np.allclose(voxel_sizes, [1.0, 2.0, 3.0])
    assert voxel_order == "RAS"


def test_get_reference_info_wrapper_nifti_header(nifti_ref):
    """Test get_reference_info_wrapper with a Nifti1Header object."""
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(
        nifti_ref.header
    )
    assert np.allclose(affine, nifti_ref.affine)
    assert np.array_equal(dimensions, [10, 20, 30])


def test_get_reference_info_wrapper_nifti_file(tmp_path, nifti_ref):
    """Test get_reference_info_wrapper with a Nifti filename."""
    path = os.path.join(tmp_path, "test.nii.gz")
    nib.save(nifti_ref, path)
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(path)
    assert np.allclose(affine, nifti_ref.affine)
    assert np.array_equal(dimensions, [10, 20, 30])


@patch("nibabel.streamlines.load")
def test_get_reference_info_wrapper_trk_file(mock_load):
    """Test get_reference_info_wrapper with a TRK filename."""
    mock_trk = MagicMock()
    mock_trk.header = {
        "voxel_to_rasmm": np.diag([1.0, 2.0, 3.0, 1.0]),
        "dimensions": np.array([10, 20, 30], dtype=np.int16),
        "voxel_sizes": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "voxel_order": "RAS",
        "magic_number": "TRACK",
    }
    mock_load.return_value = mock_trk

    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(
        "test.trk"
    )
    assert np.allclose(affine, mock_trk.header["voxel_to_rasmm"])


def test_get_reference_info_wrapper_trk_obj():
    """Test get_reference_info_wrapper with a TrkFile object."""
    mock_trk = MagicMock(spec=nib.streamlines.trk.TrkFile)
    mock_trk.header = {
        "voxel_to_rasmm": np.diag([1.0, 2.0, 3.0, 1.0]),
        "dimensions": np.array([10, 20, 30], dtype=np.int16),
        "voxel_sizes": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "voxel_order": "RAS",
        "magic_number": "TRACK",
    }
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(mock_trk)
    assert np.allclose(affine, mock_trk.header["voxel_to_rasmm"])


def test_get_reference_info_wrapper_trk_dict(trk_header):
    """Test get_reference_info_wrapper with a TRK header dict."""
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(
        trk_header
    )
    assert np.allclose(affine, trk_header["voxel_to_rasmm"])
    assert np.array_equal(dimensions, trk_header["dimensions"])


@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_get_reference_info_wrapper_sft(nifti_ref):
    """Test get_reference_info_wrapper with a StatefulTractogram."""
    streamlines = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    sft = StatefulTractogram(streamlines, nifti_ref, Space.RASMM)
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(sft)
    assert np.allclose(affine, nifti_ref.affine)
    assert np.array_equal(dimensions, [10, 20, 30])


def test_get_reference_info_wrapper_trx_obj():
    """Test get_reference_info_wrapper with a TrxFile object mock."""
    from trx.trx_file_memmap import TrxFile

    mock_trx = MagicMock(spec=TrxFile)
    mock_trx.header = {
        "VOXEL_TO_RASMM": np.diag([1.0, 1.0, 1.0, 1.0]),
        "DIMENSIONS": np.array([10, 10, 10], dtype=np.uint16),
    }
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(mock_trx)
    assert np.allclose(affine, mock_trx.header["VOXEL_TO_RASMM"])
    assert np.array_equal(dimensions, mock_trx.header["DIMENSIONS"])


@patch("trx.trx_file_memmap.load")
def test_get_reference_info_wrapper_trx_file(mock_load):
    """Test get_reference_info_wrapper with a TRX filename."""
    mock_trx = MagicMock()
    mock_trx.header = {
        "VOXEL_TO_RASMM": np.diag([1.0, 1.0, 1.0, 1.0]),
        "DIMENSIONS": np.array([10, 10, 10], dtype=np.uint16),
    }
    mock_load.return_value = mock_trx

    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(
        "test.trx"
    )
    assert np.allclose(affine, mock_trx.header["VOXEL_TO_RASMM"])


def test_get_reference_info_wrapper_trx_dict():
    """Test get_reference_info_wrapper with a TRX header dict."""
    header = {
        "VOXEL_TO_RASMM": np.diag([1.0, 1.0, 1.0, 1.0]),
        "DIMENSIONS": np.array([10, 10, 10], dtype=np.uint16),
        "NB_VERTICES": 0,
    }
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(header)
    assert np.allclose(affine, header["VOXEL_TO_RASMM"])
    assert np.array_equal(dimensions, header["DIMENSIONS"])


def test_get_reference_info_wrapper_zero_affine():
    """Test get_reference_info_wrapper with an all-zero affine."""
    mock_img = MagicMock(spec=nib.Nifti1Image)
    mock_header = MagicMock(spec=nib.Nifti1Header)
    mock_img.header = mock_header
    mock_header.get_best_affine.return_value = np.zeros((4, 4))
    mock_header.__getitem__.side_effect = lambda x: (
        [10, 10, 10] if x == "dim" else [1, 1, 1]
    )

    with pytest.raises(ValueError, match="Invalid affine, contains only zeros"):
        get_reference_info_wrapper(mock_img)


def test_get_reference_info_wrapper_binary_order():
    """Test get_reference_info_wrapper with binary voxel order."""
    header = {
        "voxel_to_rasmm": np.diag([1.0, 1.0, 1.0, 1.0]),
        "dimensions": [10, 10, 10],
        "voxel_sizes": [1, 1, 1],
        "voxel_order": np.bytes_(b"RAS"),  # numpy bytes
        "magic_number": "TRACK",
    }
    affine, dimensions, voxel_sizes, voxel_order = get_reference_info_wrapper(header)
    assert voxel_order == "RAS"


def test_get_reference_info_wrapper_error():
    """Test get_reference_info_wrapper with unsupported type."""
    with pytest.raises(
        TypeError, match="Input reference is not one of the supported format"
    ):
        get_reference_info_wrapper(123)


def test_is_header_compatible_identical(nifti_ref):
    """Test is_header_compatible with identical headers."""
    assert is_header_compatible(nifti_ref, nifti_ref)


def test_is_header_compatible_different(nifti_ref):
    """Test is_header_compatible with different headers."""
    data2 = np.zeros((10, 20, 31), dtype=np.float32)
    img2 = nib.Nifti1Image(data2, nifti_ref.affine)
    assert not is_header_compatible(nifti_ref, img2)


def test_is_header_compatible_affine_diff(nifti_ref, caplog):
    """Test is_header_compatible with different affines."""
    affine2 = nifti_ref.affine.copy()
    affine2[0, 0] = 5.0
    img2 = nib.Nifti1Image(nifti_ref.get_fdata(), affine2)
    with caplog.at_level(logging.ERROR):
        assert not is_header_compatible(nifti_ref, img2)
    assert "Affine not equal" in caplog.text or "Voxel_size not equal" in caplog.text


def test_is_header_compatible_order_diff(caplog):
    """Test is_header_compatible with different voxel orders."""
    affine1 = np.diag([1.0, 1.0, 1.0, 1.0])
    affine2 = np.diag([-1.0, 1.0, 1.0, 1.0])  # LAS instead of RAS

    header1 = {
        "voxel_to_rasmm": affine1,
        "dimensions": [10, 10, 10],
        "voxel_sizes": [1, 1, 1],
        "voxel_order": "RAS",
        "magic_number": "TRACK",
    }
    header2 = header1.copy()
    header2["voxel_to_rasmm"] = affine2
    header2["voxel_order"] = "LAS"

    with caplog.at_level(logging.ERROR):
        assert not is_header_compatible(header1, header2)
    assert "Voxel_order not equal" in caplog.text


# --- Transformation & Vector Tests ---


def test_get_axis_shift_vector():
    """Test get_axis_shift_vector."""
    assert np.array_equal(get_axis_shift_vector(["x", "y"]), [-1.0, -1.0, 0.0])
    assert np.array_equal(get_axis_shift_vector(["z"]), [0.0, 0.0, -1.0])


def test_get_axis_flip_vector():
    """Test get_axis_flip_vector."""
    assert np.array_equal(get_axis_flip_vector(["x", "z"]), [-1.0, 1.0, -1.0])
    assert np.array_equal(get_axis_flip_vector([]), [1.0, 1.0, 1.0])


@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_get_shift_vector(nifti_ref):
    """Test get_shift_vector."""
    sft = StatefulTractogram([], nifti_ref, Space.RASMM)
    shift = get_shift_vector(sft)
    assert np.array_equal(shift, [-5.0, -10.0, -15.0])


@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_flip_sft(nifti_ref):
    """Test flip_sft."""
    streamlines = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    sft = StatefulTractogram(streamlines, nifti_ref, Space.VOX)

    # Flipping X axis. Center of X is 5.0 (dim[0]=10).
    # 0 -> (0 - 5) * -1 - (-5) = -5 * -1 + 5 = 10
    # 1 -> (1 - 5) * -1 - (-5) = -4 * -1 + 5 = 9

    flipped_sft = flip_sft(sft, ["x"])
    assert np.allclose(flipped_sft.streamlines[0][0, 0], 10.0)
    assert np.allclose(flipped_sft.streamlines[0][1, 0], 9.0)


@patch("trx.utils.dipy_available", False)
def test_flip_sft_no_dipy(caplog):
    """Test flip_sft when dipy is missing."""
    with caplog.at_level(logging.ERROR):
        result = flip_sft(None, ["x"])
    assert result is None
    assert "Dipy library is missing" in caplog.text


@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
@pytest.mark.parametrize(
    "space_str,origin_str,expected_space,expected_origin",
    [
        ("rasmm", "nifti", Space.RASMM, Origin.NIFTI),
        ("voxmm", "trackvis", Space.VOXMM, Origin.TRACKVIS),
        ("vox", "nifti", Space.VOX, Origin.NIFTI),
    ],
)
def test_get_reverse_enum(space_str, origin_str, expected_space, expected_origin):
    """Test get_reverse_enum."""
    space, origin = get_reverse_enum(space_str, origin_str)
    assert space == expected_space
    assert origin == expected_origin


@patch("trx.utils.dipy_available", False)
def test_get_reverse_enum_no_dipy(caplog):
    """Test get_reverse_enum when dipy is missing."""
    with caplog.at_level(logging.ERROR):
        result = get_reverse_enum("rasmm", "nifti")
    assert result is None
    assert "Dipy library is missing" in caplog.text


# --- Data Conversion & Dtype Verification Tests ---


def test_convert_data_dict_to_tractogram():
    """Test convert_data_dict_to_tractogram."""
    data = {
        "strs": [np.array([[0, 0, 0], [1, 1, 1]]), np.array([[2, 2, 2]])],
        "dps": {"test_dps": [1, 2]},
        "dpv": {"test_dpv": [0.1, 0.2, 0.3]},
    }
    obj = convert_data_dict_to_tractogram(data)
    assert isinstance(obj, nib.streamlines.tractogram.Tractogram)
    assert len(obj.streamlines) == 2
    assert np.array_equal(obj.data_per_streamline["test_dps"], [[1], [2]])
    # Data per vertex is returned as ArraySequence
    assert np.allclose(obj.data_per_point["test_dpv"][0], [[0.1], [0.2]])


def test_append_generator_to_dict_array():
    """Test append_generator_to_dict with numpy array."""
    data = {"strs": [], "dpv": {}, "dps": {}}
    append_generator_to_dict(np.array([[0, 0, 0]]), data)
    assert len(data["strs"]) == 1


def test_append_generator_to_dict_item():
    """Test append_generator_to_dict with TractogramItem."""
    data = {"strs": [], "dpv": {}, "dps": {}}
    # TractogramItem(streamline, data_for_streamline=None, data_for_points=None)
    item = TractogramItem(np.array([[0, 0, 0]]), {"s": 1}, {"v": [0.1]})
    append_generator_to_dict(item, data)
    assert len(data["strs"]) == 1
    assert "v" in data["dpv"]
    assert "s" in data["dps"]


def test_verify_trx_dtype():
    """Test verify_trx_dtype."""
    # Create a mock TRX object
    mock_trx = MagicMock(spec=Tractogram)
    mock_trx.streamlines._data.dtype = np.float32
    mock_trx.streamlines._offsets.dtype = np.uint32

    mock_dpv = MagicMock()
    mock_dpv._data.dtype = np.uint16
    mock_trx.data_per_vertex = {"v1": mock_dpv}

    mock_trx.data_per_streamline = {"s1": np.array([1], dtype="int16")}

    # Define expected dtype dict
    dtype_dict = {
        "positions": np.float32,
        "offsets": np.uint32,
        "dpv": {"v1": np.uint16},
        "dps": {"s1": np.int16},
    }

    assert verify_trx_dtype(mock_trx, dtype_dict)

    # Test mismatches for warnings
    with patch("logging.warning") as mock_log:
        dtype_dict["positions"] = np.float64
        assert not verify_trx_dtype(mock_trx, dtype_dict)
        mock_log.assert_any_call("Positions dtype is different")

        dtype_dict["positions"] = np.float32
        dtype_dict["offsets"] = np.uint64
        assert not verify_trx_dtype(mock_trx, dtype_dict)
        mock_log.assert_any_call("Offsets dtype is different")

        dtype_dict["offsets"] = np.uint32
        dtype_dict["dpv"]["v1"] = np.uint32
        assert not verify_trx_dtype(mock_trx, dtype_dict)
        mock_log.assert_any_call("Data per vertex (v1) dtype is different")

        dtype_dict["dpv"]["v1"] = np.uint16
        dtype_dict["dps"]["s1"] = np.int32
        assert not verify_trx_dtype(mock_trx, dtype_dict)
        mock_log.assert_any_call("Data per streamline (s1) dtype is different")


def test_verify_trx_dtype_groups():
    """Test verify_trx_dtype with groups and dpg."""
    mock_trx = MagicMock(spec=Tractogram)
    mock_trx.streamlines._data.dtype = np.float32
    mock_trx.streamlines._offsets.dtype = np.uint32

    mock_g1 = MagicMock()
    mock_g1._data.dtype = np.int32

    mock_dpg_val = MagicMock()
    mock_dpg_val.dtype = np.float32

    # verify_trx_dtype expects trx.data_per_point to contain groups and dpg
    mock_trx.data_per_point = {"g1": mock_g1, "g2": {"d1": mock_dpg_val}}

    dtype_dict = {"groups": {"g1": np.int32}, "dpg": {"g2": {"d1": np.float32}}}

    assert verify_trx_dtype(mock_trx, dtype_dict)

    # Test mismatches for dpg and groups
    with patch("logging.warning") as mock_log:
        dtype_dict["groups"]["g1"] = np.int16
        assert not verify_trx_dtype(mock_trx, dtype_dict)
        mock_log.assert_any_call("Data per group (g1) dtype is different")

        dtype_dict["groups"]["g1"] = np.int32
        dtype_dict["dpg"]["g2"]["d1"] = np.float64
        assert not verify_trx_dtype(mock_trx, dtype_dict)
        mock_log.assert_any_call("Data per group (d1) dtype is different")
