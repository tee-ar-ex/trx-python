# -*- coding: utf-8 -*-

import hashlib
import logging
import os
import shutil
import urllib.request

TEST_DATA_REPO = "tee-ar-ex/trx-test-data"
TEST_DATA_TAG = "v0.1.0"
# GitHub release API entrypoint for metadata (asset list, sizes, etc.).
TEST_DATA_API_URL = (
    f"https://api.github.com/repos/{TEST_DATA_REPO}/releases/tags/{TEST_DATA_TAG}"
)
# Direct download base for release assets.
TEST_DATA_BASE_URL = (
    f"https://github.com/{TEST_DATA_REPO}/releases/download/{TEST_DATA_TAG}"
)


def get_home():
    """Set a user-writeable file-system location to put files"""
    if "TRX_HOME" in os.environ:
        trx_home = os.environ["TRX_HOME"]
    else:
        trx_home = os.path.join(os.path.expanduser("~"), ".tee_ar_ex")
    return trx_home


def get_testing_files_dict():
    """Get dictionary linking zip file to their GitHub release URL & checksums.

    Assets are hosted under the v0.1.0 release of tee-ar-ex/trx-test-data.
    If URLs change, check TEST_DATA_API_URL to discover the latest asset
    locations.
    """
    return {
        "DSI.zip": (
            f"{TEST_DATA_BASE_URL}/DSI.zip",
            "b847f053fc694d55d935c0be0e5268f7",  # md5
            "1b09ce8b4b47b2600336c558fdba7051218296e8440e737364f2c4b8ebae666c",
        ),
        "memmap_test_data.zip": (
            f"{TEST_DATA_BASE_URL}/memmap_test_data.zip",
            "03f7651a0f9e3eeabee9aed0ad5f69e1",  # md5
            "98ba89d7a9a7baa2d37956a0a591dce9bb4581bd01296ad5a596706ee90a52ef",
        ),
        "trx_from_scratch.zip": (
            f"{TEST_DATA_BASE_URL}/trx_from_scratch.zip",
            "d9f220a095ce7f027772fcd9451a2ee5",  # md5
            "f98ab6da6a6065527fde4b0b6aa40f07583e925d952182e9bbd0febd55c0f6b2",
        ),
        "gold_standard.zip": (
            f"{TEST_DATA_BASE_URL}/gold_standard.zip",
            "57e3f9951fe77245684ede8688af3ae8",  # md5
            "35a0b633560cc2b0d8ecda885aa72d06385499e0cd1ca11a956b0904c3358f01",
        ),
    }


def md5sum(filename):
    """Compute the MD5 checksum of a file.

    Parameters
    ----------
    filename : str
        Path to file to hash.

    Returns
    -------
    str
        Hexadecimal MD5 digest.
    """
    h = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * h.block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256sum(filename):
    """Compute the SHA256 checksum of a file.

    Parameters
    ----------
    filename : str
        Path to file to hash.

    Returns
    -------
    str
        Hexadecimal SHA256 digest.
    """
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(128 * h.block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_data(files_dict, keys=None):  # noqa: C901
    """Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files_dict : dictionary
        For each file in `files_dict` the value should be (url, md5).
        The file will be downloaded from url, if the file does not already
        exist or if the file exists but the md5 checksum does not match.
        Zip files are automatically unzipped and its content* are md5 checked.

    Raises
    ------
    ValueError
        Raises if the md5 checksum of the file does not match the expected
        value. The downloaded file is not deleted when this error is raised.
    """
    trx_home = get_home()

    if not os.path.exists(trx_home):
        os.makedirs(trx_home)

    if keys is None:
        keys = files_dict.keys()
    elif isinstance(keys, str):
        keys = [keys]

    for f in keys:
        file_entry = files_dict[f]
        if len(file_entry) == 2:
            url, expected_md5 = file_entry
            expected_sha = None
        else:
            url, expected_md5, expected_sha = file_entry
        full_path = os.path.join(trx_home, f)

        logging.info("Downloading {} to {}".format(f, trx_home))
        if not os.path.exists(full_path):
            urllib.request.urlretrieve(url, full_path)

        actual_md5 = md5sum(full_path)
        if expected_md5 != actual_md5:
            raise ValueError(
                f"Md5sum for {f} does not match. "
                "Please remove the file to download it again: " + full_path
            )

        if expected_sha is not None:
            actual_sha = sha256sum(full_path)
            if expected_sha != actual_sha:
                raise ValueError(
                    f"SHA256 for {f} does not match. "
                    "Please remove the file to download it again: " + full_path
                )

        if f.endswith(".zip"):
            dst_dir = os.path.join(trx_home, f[:-4])
            shutil.unpack_archive(full_path, extract_dir=dst_dir, format="zip")
