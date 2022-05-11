#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os

from gdown import cached_download, extractall

GOOGLE_URL = "https://drive.google.com/uc?id="


def get_home():
    """ Set a user-writeable file-system location to put files """
    if 'TRX_HOME' in os.environ:
        trx_home = os.environ['TRX_HOME']
    else:
        trx_home = os.path.join(os.path.expanduser('~'), '.tee_ar_ex/')
    return trx_home


def get_testing_files_dict():
    """ Get dictionary linking zip file to their GDrive ID & MD5SUM """
    return {'DSI.zip':
            ['18i9aAuMmPPaH6D03CH2hY4tDrAVBKTRe',
             'a984e4f5a37273063a713ee578901127']}


def fetch_data(files_dict, keys=None):
    """ Downloads files to folder and checks their md5 checksums

    Parameters
    ----------
    files_dict : dictionary
        For each file in `files_dict` the value should be (url, md5).
        The file will be downloaded from url, if the file does not already
        exist or if the file exists but the md5 checksum does not match.

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
        url, md5 = files_dict[f]
        full_path = os.path.join(trx_home, f)

        logging.info('Downloading {} to {}'.format(f, trx_home))
        cached_download(url=GOOGLE_URL+url,
                        path=full_path,
                        md5=md5,
                        quiet=True,
                        postprocess=extractall)
