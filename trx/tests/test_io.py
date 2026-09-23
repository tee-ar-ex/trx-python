# -*- coding: utf-8 -*-

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from numpy.testing import assert_allclose
import psutil
import pytest

try:
    from dipy.io.streamline import load_tractogram, save_tractogram

    dipy_available = True
except ImportError:
    dipy_available = False

from trx.fetcher import fetch_data, get_home, get_testing_files_dict
from trx.io import get_trx_tmp_dir, load, save
import trx.trx_file_memmap as tmm
from trx.trx_file_memmap import TrxFile

fetch_data(get_testing_files_dict(), keys=["gold_standard.zip"])


@pytest.mark.parametrize("path", ["gs.trk", "gs.tck", "gs.vtk"])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_seq_ops_sft(tmp_path, path):
    gs_dir = get_home() / "gold_standard"
    path = tmp_path / path

    obj = load(gs_dir / "gs.trx", gs_dir / "gs.nii")
    sft_1 = obj.to_sft()
    save_tractogram(sft_1, path)
    obj.close()
    save_tractogram(sft_1, tmp_path / "tmp.trk")

    _ = load_tractogram(tmp_path / "tmp.trk", "same")


def test_seq_ops_trx(tmp_path):
    gs_dir = get_home() / "gold_standard"
    path = gs_dir / "gs.trx"

    trx_1 = tmm.load(path)
    tmm.save(trx_1, tmp_path / "tmp.trx")
    trx_1.close()
    trx_2 = tmm.load(tmp_path / "tmp.trx")
    trx_2.close()


@pytest.mark.parametrize("path", ["gs.trx", "gs.trk", "gs.tck", "gs.vtk"])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_load_vox(path):
    from dipy.io.stateful_tractogram import Space

    gs_dir = get_home() / "gold_standard"
    path = gs_dir / path
    coord = np.loadtxt(get_home() / "gold_standard" / "gs_vox_space.txt")
    from_space = Space.LPSMM if path.name.endswith("gs.vtk") else None
    obj = load(path, gs_dir / "gs.nii", from_space=from_space)

    sft = obj.to_sft() if isinstance(obj, TrxFile) else obj
    sft.to_vox()

    assert_allclose(sft.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", ["gs.trx", "gs.trk", "gs.tck", "gs.vtk"])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_load_voxmm(path):
    from dipy.io.stateful_tractogram import Space

    gs_dir = get_home() / "gold_standard"
    path = gs_dir / path
    coord = np.loadtxt(get_home() / "gold_standard" / "gs_voxmm_space.txt")
    from_space = Space.LPSMM if path.name.endswith("gs.vtk") else None
    obj = load(path, gs_dir / "gs.nii", from_space=from_space)

    sft = obj.to_sft() if isinstance(obj, TrxFile) else obj
    sft.to_voxmm()

    assert_allclose(sft.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", ["gs.trk", "gs.trx", "gs_fldr.trx"])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_multi_load_save_rasmm(tmp_path, path):
    gs_dir = get_home() / "gold_standard"
    basename = Path(path).stem
    ext = Path(path).suffix.lstrip(".")

    path = gs_dir / path
    coord = np.loadtxt(get_home() / "gold_standard" / "gs_rasmm_space.txt")

    obj = load(path, gs_dir / "gs.nii")
    for i in range(3):
        out_path = tmp_path / f"{basename}_tmp{i}_{ext}"
        save(obj, out_path)

        if isinstance(obj, TrxFile):
            obj.close()
        obj = load(out_path, gs_dir / "gs.nii")

    assert_allclose(obj.streamlines._data, coord, rtol=1e-04, atol=1e-06)
    if isinstance(obj, TrxFile):
        obj.close()


@pytest.mark.parametrize("path", ["gs.trx", "gs_fldr.trx"])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_delete_tmp_gs_dir(path):
    gs_dir = get_home() / "gold_standard"
    path = gs_dir / path

    trx1 = tmm.load(path)
    if path.is_file():
        tmp_gs_dir = deepcopy(trx1._uncompressed_folder_handle.name)
        assert tmp_gs_dir.is_dir()
    sft = trx1.to_sft()
    trx1.close()

    coord_rasmm = np.loadtxt(get_home() / "gold_standard" / "gs_rasmm_space.txt")
    coord_vox = np.loadtxt(get_home() / "gold_standard" / "gs_vox_space.txt")

    # The folder trx representation does not need tmp files
    if path.is_file():
        assert not tmp_gs_dir.is_dir()

    assert_allclose(sft.streamlines._data, coord_rasmm, rtol=1e-04, atol=1e-06)

    # Reloading the TRX and checking its data, then closing
    trx2 = tmm.load(path)
    assert_allclose(
        trx2.streamlines._data, sft.streamlines._data, rtol=1e-04, atol=1e-06
    )
    trx2.close()

    sft.to_vox()
    assert_allclose(sft.streamlines._data, coord_vox, rtol=1e-04, atol=1e-06)

    trx3 = tmm.load(path)
    assert_allclose(trx3.streamlines._data, coord_rasmm, rtol=1e-04, atol=1e-06)
    trx3.close()


@pytest.mark.parametrize("path", ["gs.trx"])
@pytest.mark.skipif(not dipy_available, reason="Dipy is not installed.")
def test_close_tmp_files(path):
    gs_dir = get_home() / "gold_standard"
    path = gs_dir / path

    tgm = tmm.load(path)
    process = psutil.Process(os.getpid())
    open_files = process.open_files()

    expected_content = [
        "offsets.uint32",
        "positions.3.float32",
        "header.json",
        "random_coord.3.float32",
        "color_y.float32",
        "color_x.float32",
        "color_z.float32",
    ]

    count = 0
    for open_file in open_files:
        basename = Path(open_file.path).name
        if basename in expected_content:
            count += 1

    assert count == 6
    tgm.close()

    open_files = process.open_files()
    count = 0
    for open_file in open_files:
        basename = Path(open_file.path).name
        if basename in expected_content:
            count += 1
    assert not count


@pytest.mark.parametrize(
    "env_value, expected_parent_fn",
    [
        ("use_working_dir", lambda: Path.cwd()),
        (Path("~").expanduser(), lambda: Path("~").expanduser()),
        (None, lambda: Path(tempfile.gettempdir())),
    ],
)
def test_get_trx_tmp_dir(env_value, expected_parent_fn, monkeypatch):
    if env_value is None:
        monkeypatch.delenv("TRX_TMPDIR", raising=False)
    else:
        monkeypatch.setenv("TRX_TMPDIR", str(env_value))

    td = get_trx_tmp_dir()
    tmp_path = Path(td.name)
    try:
        assert tmp_path.parent == expected_parent_fn()
        assert tmp_path.is_dir()
    finally:
        td.cleanup()

    assert not tmp_path.is_dir()


@pytest.mark.parametrize(
    "trx_tmpdir_env, expected_parent",
    [
        ("use_working_dir", lambda: Path.cwd()),
        (Path("~").expanduser(), lambda: Path("~").expanduser()),
        (None, lambda: Path(tempfile.gettempdir())),
    ],
)
def test_change_tmp_dir(trx_tmpdir_env, expected_parent, monkeypatch):
    """Integration test through tmm.load(path), assuming that it
    eventually calls get_trx_tmp_dir()."""
    gs_dir = get_home() / "gold_standard"
    path = gs_dir / "gs.trx"

    if trx_tmpdir_env is None:
        monkeypatch.delenv("TRX_TMPDIR", raising=False)
    else:
        monkeypatch.setenv("TRX_TMPDIR", str(trx_tmpdir_env))

    tgm = tmm.load(path)
    tmp_gs_dir = deepcopy(tgm._uncompressed_folder_handle.name)

    assert tmp_gs_dir.parent == expected_parent()

    tgm.close()
    assert not tmp_gs_dir.is_dir()


@pytest.mark.parametrize("path", ["gs.trx", "gs_fldr.trx"])
def test_complete_dir_from_trx(path):
    gs_dir = get_home() / "gold_standard"
    path = gs_dir / path

    tgm = tmm.load(path)
    if tgm._uncompressed_folder_handle is None:
        dir_to_check = path
    else:
        dir_to_check = tgm._uncompressed_folder_handle.name

    file_paths = []
    for dirpath, _, filenames in os.walk(dir_to_check):
        for filename in filenames:
            full_path = Path(dirpath) / filename
            cut_path = full_path.relative_to(dir_to_check).as_posix()
            file_paths.append(cut_path)

    expected_content = [
        "offsets.uint32",
        "positions.3.float32",
        "header.json",
        "dps/random_coord.3.float32",
        "dpv/color_y.float32",
        "dpv/color_x.float32",
        "dpv/color_z.float32",
    ]
    assert set(file_paths) == set(expected_content)


def test_complete_zip_from_trx():
    gs_dir = get_home() / "gold_standard"
    path = gs_dir / "gs.trx"

    with zipfile.ZipFile(path, mode="r") as zf:
        zip_file_list = zf.namelist()

    expected_content = [
        "offsets.uint32",
        "positions.3.float32",
        "header.json",
        "dps/random_coord.3.float32",
        "dpv/color_y.float32",
        "dpv/color_x.float32",
        "dpv/color_z.float32",
    ]
    assert set(zip_file_list) == set(expected_content)
