# -*- coding: utf-8 -*-

import os
import logging
import tempfile
import sys

try:
    import dipy
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.utils import split_name_with_gz


def get_trx_tmp_dir():
    if os.getenv('TRX_TMPDIR') is not None:
        if os.getenv('TRX_TMPDIR') == 'use_working_dir':
            trx_tmp_dir = os.getcwd()
        else:
            trx_tmp_dir = os.getenv('TRX_TMPDIR')
    else:
        trx_tmp_dir = tempfile.gettempdir()

    if sys.version_info[1] >= 10:
        return tempfile.TemporaryDirectory(dir=trx_tmp_dir, prefix='trx_',
                                           ignore_cleanup_errors=True)
    else:
        return tempfile.TemporaryDirectory(dir=trx_tmp_dir, prefix='trx_')


def load_sft_with_reference(filepath, reference=None,
                            bbox_check=True):
    if not dipy_available:
        logging.error('Dipy library is missing, cannot use functions related '
                      'to the StatefulTractogram.')
        return None
    from dipy.io.streamline import load_tractogram

    # Force the usage of --reference for all file formats without an header
    _, ext = os.path.splitext(filepath)
    if ext == '.trk':
        if reference is not None and reference != 'same':
            logging.warning('Reference is discarded for this file format '
                            '{}.'.format(filepath))
        sft = load_tractogram(filepath, 'same',
                              bbox_valid_check=bbox_check)
    elif ext in ['.tck', '.fib', '.vtk', '.dpy']:
        if reference is None or reference == 'same':
            raise IOError('--reference is required for this file format '
                          '{}.'.format(filepath))
        else:
            sft = load_tractogram(filepath, reference,
                                  bbox_valid_check=bbox_check)

    else:
        raise IOError('{} is an unsupported file format'.format(filepath))

    return sft


def load(tractogram_filename, reference):
    import trx.trx_file_memmap as tmm
    in_ext = split_name_with_gz(tractogram_filename)[1]
    if in_ext != '.trx' and not os.path.isdir(tractogram_filename):
        tractogram_obj = load_sft_with_reference(tractogram_filename,
                                                 reference,
                                                 bbox_check=False)
    else:
        tractogram_obj = tmm.load(tractogram_filename)

    return tractogram_obj


def save(tractogram_obj, tractogram_filename, bbox_valid_check=False):
    if not dipy_available:
        logging.error('Dipy library is missing, cannot use functions related '
                      'to the StatefulTractogram.')
        return None
    from dipy.io.stateful_tractogram import StatefulTractogram
    from dipy.io.streamline import save_tractogram
    import trx.trx_file_memmap as tmm

    out_ext = split_name_with_gz(tractogram_filename)[1]

    if out_ext != '.trx':
        if not isinstance(tractogram_obj, StatefulTractogram):
            tractogram_obj = tractogram_obj.to_sft()
        save_tractogram(tractogram_obj, tractogram_filename,
                        bbox_valid_check=bbox_valid_check)
    else:
        if not isinstance(tractogram_obj, tmm.TrxFile):
            tractogram_obj = tmm.TrxFile.from_sft(tractogram_obj)
        tmm.save(tractogram_obj, tractogram_filename)
        tractogram_obj.close()
