# -*- coding: utf-8 -*-
"""Tests for the storage-backend registry and the default memmap backend."""

import os
import zipfile

import numpy as np
import pytest

import trx.backends as backends
from trx.backends.memmap import MemmapBackend
import trx.trx_file_memmap as tmm


def _small_trx():
    """A tiny in-memory TrxFile: two streamlines, one dps and one dpv key."""
    trx = tmm.TrxFile(nb_vertices=5, nb_streamlines=2)
    trx.streamlines._data[:] = np.arange(15, dtype=np.float32).reshape(5, 3)
    trx.streamlines._offsets[:] = np.array([0, 2], dtype=np.uint32)
    trx.streamlines._lengths[:] = np.array([2, 3], dtype=np.uint32)
    trx.data_per_streamline["w"] = np.array([[1.0], [2.0]], dtype=np.float32)
    return trx


def test_memmap_backend_is_registered_first():
    registered = backends.registered_backends()
    assert registered[0] is MemmapBackend
    assert isinstance(MemmapBackend, type)
    assert MemmapBackend.name == "memmap"


def test_backend_for_saving_by_extension(tmp_path):
    assert backends.backend_for_saving(str(tmp_path / "a.trx")) is MemmapBackend
    assert backends.backend_for_saving(str(tmp_path / "a.zip")) is MemmapBackend
    assert backends.backend_for_saving(str(tmp_path / "folder")) is MemmapBackend
    with pytest.raises(ValueError, match="Unsupported extension"):
        backends.backend_for_saving(str(tmp_path / "a.trk"))


def test_backend_for_loading_zip_and_folder(tmp_path):
    trx = _small_trx()
    zip_path = str(tmp_path / "small.trx")
    dir_path = str(tmp_path / "small_dir")
    tmm.save(trx, zip_path)
    tmm.save(trx, dir_path)

    assert zipfile.is_zipfile(zip_path)
    assert os.path.isfile(os.path.join(dir_path, "header.json"))
    assert backends.backend_for_loading(zip_path) is MemmapBackend
    assert backends.backend_for_loading(dir_path) is MemmapBackend

    not_trx = tmp_path / "plain.txt"
    not_trx.write_text("hello")
    with pytest.raises(ValueError, match="No TRX storage backend"):
        backends.backend_for_loading(str(not_trx))


def test_load_round_trip_through_registry(tmp_path):
    trx = _small_trx()
    zip_path = str(tmp_path / "rt.trx")
    tmm.save(trx, zip_path)
    loaded = tmm.load(zip_path)
    try:
        assert loaded.header["NB_STREAMLINES"] == 2
        assert loaded.header["NB_VERTICES"] == 5
        np.testing.assert_array_equal(loaded.streamlines._data, trx.streamlines._data)
        np.testing.assert_array_equal(
            np.asarray(loaded.data_per_streamline["w"]).ravel(), [1.0, 2.0]
        )
    finally:
        loaded.close()


def test_legacy_names_still_exported():
    # Downstream code imports these from trx.trx_file_memmap.
    for name in (
        "load",
        "save",
        "load_from_zip",
        "load_from_directory",
        "zip_from_folder",
        "concatenate",
        "TrxFile",
    ):
        assert hasattr(tmm, name), name
    assert hasattr(tmm.TrxFile, "_create_trx_from_pointer")


def test_custom_backend_can_be_registered(tmp_path):
    class DummyBackend:
        name = "dummy"
        saved = []

        @classmethod
        def can_load(cls, path):
            return path.endswith(".dummy")

        @classmethod
        def can_save(cls, path):
            return path.endswith(".dummy")

        @classmethod
        def load(cls, path):
            return tmm.TrxFile()

        @classmethod
        def save(cls, tgm, path, **options):
            cls.saved.append((path, options))

    backends.register_backend(DummyBackend)
    try:
        assert isinstance(DummyBackend, backends.TrxBackend)
        target = str(tmp_path / "x.dummy")
        assert backends.backend_for_saving(target) is DummyBackend
        tmm.save(_small_trx(), target)
        assert DummyBackend.saved and DummyBackend.saved[0][0] == target
        # The memmap backend still wins for its own extensions.
        assert backends.backend_for_saving(str(tmp_path / "y.trx")) is MemmapBackend
    finally:
        backends.base._REGISTRY.remove(DummyBackend)
