:html_theme.sidebar_secondary.remove:

Command-line Interface
======================

The TRX toolkit provides a unified command-line interface ``trx`` as well as individual standalone commands for backward compatibility. All commands become available on your ``PATH`` after installing ``trx-python``.

Each command supports ``--help`` for full options.

Unified CLI: ``trx``
--------------------

The recommended way to use TRX commands is through the unified ``trx`` CLI:

.. code-block:: bash

    trx --help              # Show all available commands
    trx <command> --help    # Show help for a specific command

Available subcommands:

- ``trx info`` - Display detailed TRX file information
- ``trx concatenate`` - Concatenate multiple tractograms
- ``trx convert`` - Convert between tractography formats
- ``trx convert-dsi`` - Fix DSI-Studio TRK files
- ``trx generate`` - Generate TRX from raw data files
- ``trx manipulate-dtype`` - Change array data types
- ``trx compare`` - Simple tractogram comparison
- ``trx validate`` - Validate and clean TRX files
- ``trx verify-header`` - Check header compatibility
- ``trx visualize`` - Visualize tractogram overlap

Standalone Commands
-------------------

For backward compatibility, standalone commands are also available:

trx_info
~~~~~~~~
Display detailed information about a TRX file, including file size, compression status, header metadata (affine, dimensions, voxel sizes), streamline/vertex counts, data keys (dpv, dps, dpg), groups, and archive contents.

- Only ``.trx`` files are supported.

.. code-block:: bash

    # Using unified CLI
    trx info tractogram.trx

    # Using standalone command
    trx_info tractogram.trx

trx_concatenate_tractograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Concatenate multiple tractograms into a single output.

- Supports ``trk``, ``tck``, ``vtk``, ``fib``, ``dpy``, and ``trx`` inputs.
- Flags: ``--delete-dpv``, ``--delete-dps``, ``--delete-groups`` to drop mismatched metadata; ``--reference`` for formats needing an anatomy reference; ``-f`` to overwrite.

.. code-block:: bash

    # Using unified CLI
    trx concatenate in1.trk in2.trk merged.trx

    # Using standalone command
    trx_concatenate_tractograms in1.trk in2.trk merged.trx

trx_convert_dsi_studio
~~~~~~~~~~~~~~~~~~~~~~
Convert a DSI Studio ``.trk`` with accompanying ``.nii.gz`` reference into a cleaned ``.trk`` or TRX.

.. code-block:: bash

    # Using unified CLI
    trx convert-dsi input.trk reference.nii.gz cleaned.trk

    # Using standalone command
    trx_convert_dsi_studio input.trk reference.nii.gz cleaned.trk

trx_convert_tractogram
~~~~~~~~~~~~~~~~~~~~~~
General-purpose converter between ``trk``, ``tck``, ``vtk``, ``fib``, ``dpy``, and ``trx``.

- Flags: ``--reference`` for formats needing a NIfTI, ``--positions-dtype``, ``--offsets-dtype``, ``-f`` to overwrite.

.. code-block:: bash

    # Using unified CLI
    trx convert input.trk output.trx --positions-dtype float32 --offsets-dtype uint64

    # Using standalone command
    trx_convert_tractogram input.trk output.trx --positions-dtype float32 --offsets-dtype uint64

trx_generate_from_scratch
~~~~~~~~~~~~~~~~~~~~~~~~~
Build a TRX file from raw NumPy arrays or CSV streamline coordinates.

- Flags: ``--positions``, ``--offsets``, ``--positions-dtype``, ``--offsets-dtype``, spatial options (``--space``, ``--origin``), and metadata loaders for dpv/dps/groups/dpg.

.. code-block:: bash

    # Using unified CLI
    trx generate fa.nii.gz output.trx --positions positions.npy --offsets offsets.npy

    # Using standalone command
    trx_generate_from_scratch fa.nii.gz output.trx --positions positions.npy --offsets offsets.npy

trx_manipulate_datatype
~~~~~~~~~~~~~~~~~~~~~~~
Rewrite TRX datasets with new dtypes for positions/offsets/dpv/dps/dpg/groups.

- Accepts per-field dtype arguments and overwrites with ``-f``.

.. code-block:: bash

    # Using unified CLI
    trx manipulate-dtype input.trx output.trx --positions-dtype float16 --dpv color,uint8

    # Using standalone command
    trx_manipulate_datatype input.trx output.trx --positions-dtype float16 --dpv color,uint8

trx_simple_compare
~~~~~~~~~~~~~~~~~~
Compare two tractograms for quick difference checks.

.. code-block:: bash

    # Using unified CLI
    trx compare first.trk second.trk

    # Using standalone command
    trx_simple_compare first.trk second.trk

trx_validate
~~~~~~~~~~~~
Validate a TRX file for consistency and remove invalid streamlines.

.. code-block:: bash

    # Using unified CLI
    trx validate data.trx --out cleaned.trx

    # Using standalone command
    trx_validate data.trx --out cleaned.trx

trx_verify_header_compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Check whether tractogram headers are compatible for operations such as concatenation.

.. code-block:: bash

    # Using unified CLI
    trx verify-header file1.trk file2.trk

    # Using standalone command
    trx_verify_header_compatibility file1.trk file2.trk

trx_visualize_overlap
~~~~~~~~~~~~~~~~~~~~~
Visualize streamline overlap between tractograms (requires visualization dependencies).

.. code-block:: bash

    # Using unified CLI
    trx visualize tractogram.trk reference.nii.gz

    # Using standalone command
    trx_visualize_overlap tractogram.trk reference.nii.gz

Troubleshooting
---------------

If the ``trx`` command is not working as expected, run ``trx --debug`` to print
diagnostic information about the Python interpreter, package location, and
whether all required and optional dependencies are installed:

.. code-block:: bash

    trx --debug

    # Example output:
    # Environment diagnostics:
    #   Python executable : /Users/you/myenv/bin/python
    #   sys.prefix        : /Users/you/myenv
    #   trx-python version: 0.3.1
    #   trx package       : /Users/you/myenv/lib/python3.11/site-packages/trx
    #
    # Required dependencies:
    #   deepdiff     found
    #   nibabel      found
    #   numpy        found
    #   typer        found
    #
    # Optional dependencies:
    #   dipy         found
    #   fury         not found
    #   vtk          not found

Notes
-----
- Test datasets for examples can be fetched with ``python -m trx.fetcher`` helpers: ``fetch_data(get_testing_files_dict())`` downloads to ``$TRX_HOME`` (default ``~/.tee_ar_ex``).
- All commands print detailed usage with ``--help``.
- The unified ``trx`` CLI uses `Typer <https://typer.tiangolo.com/>`_ for beautiful terminal output with colors and rich formatting.
