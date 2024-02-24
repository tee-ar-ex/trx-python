# -*- coding: utf-8 -*-

import hashlib
import logging
import os
import shutil
import urllib.request


def get_home():
    """ Set a user-writeable file-system location to put files """
    if 'TRX_HOME' in os.environ:
        trx_home = os.environ['TRX_HOME']
    else:
        trx_home = os.path.join(os.path.expanduser('~'), '.tee_ar_ex')
    return trx_home


def get_testing_files_dict():
    """ Get dictionary linking zip file to their Figshare URL & MD5SUM """
    return {
        'DSI.zip':
        ('https://figshare.com/ndownloader/files/37624154',
         'b847f053fc694d55d935c0be0e5268f7'),  # V1 (27.09.2022)
        'memmap_test_data.zip':
        ('https://figshare.com/ndownloader/files/37624148',
         '03f7651a0f9e3eeabee9aed0ad5f69e1'),  # V2 (27.09.2022)
        'trx_from_scratch.zip':
        ('https://figshare.com/ndownloader/files/37624151',
         'd9f220a095ce7f027772fcd9451a2ee5'),  # V2 (27.09.2022)
        'gold_standard.zip':
        ('https://figshare.com/ndownloader/files/38146098',
         '57e3f9951fe77245684ede8688af3ae8')  # V1 (8.11.2022)
    }


def md5sum(filename):
    """ Compute one md5 checksum for a file """
    h = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(128 * h.block_size), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_data(files_dict, keys=None):
    """ Downloads files to folder and checks their md5 checksums

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
        url, expected_md5 = files_dict[f]
        full_path = os.path.join(trx_home, f)

        logging.info('Downloading {} to {}'.format(f, trx_home))
        if not os.path.exists(full_path):
            urllib.request.urlretrieve(url, full_path)

        actual_md5 = md5sum(full_path)
        if expected_md5 != actual_md5:
            raise ValueError(
                f'Md5sum for {f} does not match. '
                'Please remove the file to download it again: ' +
                full_path
            )

        if f.endswith('.zip'):
            dst_dir = os.path.join(trx_home, f[:-4])
            shutil.unpack_archive(full_path,
                                  extract_dir=dst_dir,
                                  format='zip')
