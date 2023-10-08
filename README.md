# trx-python

This is a Python implementation of the trx file-format for tractography data.

For details, please visit the documentation web-page at https://tee-ar-ex.github.io/trx-python/.

To install this, you can run:

    pip install trx-python

Or, to install from source:

    git clone https://github.com/tee-ar-ex/trx-python.git
    cd trx-python
    pip install .

### Temporary Directory
The TRX file format uses memmaps to limit RAM usage. When dealing with large files this means several gigabytes could be required on disk (instead of RAM). 

By default, the temporary directory on Linux and MacOS is `/tmp` and on Windows it should be `C:\WINDOWS\Temp`.

If you wish to change the directory add the following variable to your script or to your .bashrc or .bash_profile:
`export TRX_TMPDIR=/WHERE/I/WANT/MY/TMP/DATA` (a)
OR
`export TRX_TMPDIR=use_working_dir` (b)

The provided folder must already exists (a). `use_working_dir` will be the directory where the code is being executed from (b).

The temporary folders should be automatically cleaned. But, if the code crash unexpectedly, make sure the folders are deleted.
