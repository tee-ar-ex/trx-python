#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import tempfile

try:
    import dipy
    dipy_available = True
except ImportError:
    dipy_available = False

from trx.utils import split_name_with_gz


def get_trx_tmpdir():
    if os.getenv('TRX_TMPDIR') is not None:
        if os.getenv('TRX_TMPDIR') == 'use_working_dir':
            trx_tmp_dir = os.getcwd()
        else:
            trx_tmp_dir = os.getenv('TRX_TMPDIR')
    else:
        trx_tmp_dir = tempfile.gettempdir()
    return tempfile.TemporaryDirectory(dir=trx_tmp_dir, prefix='trx_')
    # Step 1: Traverse the directory tree
    for dirpath, dirnames, filenames in os.walk(tmpdir_path, topdown=False):
        
        # Step 2 and 3: Identify and remove symlinks (for files)
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.islink(filepath):
                os.unlink(filepath)
        
        # Step 2 and 3: Identify and remove symlinks (for directories)
        for dirname in dirnames:
            dir_full_path = os.path.join(dirpath, dirname)
            if os.path.islink(dir_full_path):
                os.unlink(dir_full_path)
    
    # Step 4: Remove the temporary directory
    shutil.rmtree(tmpdir_path)

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
