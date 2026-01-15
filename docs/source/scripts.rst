Command-line Interface
======================

The TRX toolkit provides a unified command-line interface ``tff`` as well as individual standalone commands for backward compatibility. All commands become available on your ``PATH`` after installing ``trx-python``.

Each command supports ``--help`` for full options.

Unified CLI: ``tff``
--------------------

The recommended way to use TRX commands is through the unified ``tff`` CLI:

.. code-block:: bash

    tff --help              # Show all available commands
    tff <command> --help    # Show help for a specific command

Available subcommands:

- ``tff concatenate`` - Concatenate multiple tractograms
- ``tff convert`` - Convert between tractography formats
- ``tff convert-dsi`` - Fix DSI-Studio TRK files
- ``tff generate`` - Generate TRX from raw data files
- ``tff manipulate-dtype`` - Change array data types
- ``tff compare`` - Simple tractogram comparison
- ``tff validate`` - Validate and clean TRX files
- ``tff verify-header`` - Check header compatibility
- ``tff visualize`` - Visualize tractogram overlap

Standalone Commands
-------------------

For backward compatibility, standalone commands are also available:

tff_concatenate_tractograms
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Concatenate multiple tractograms into a single output.

- Supports ``trk``, ``tck``, ``vtk``, ``fib``, ``dpy``, and ``trx`` inputs.
- Flags: ``--delete-dpv``, ``--delete-dps``, ``--delete-groups`` to drop mismatched metadata; ``--reference`` for formats needing an anatomy reference; ``-f`` to overwrite.

.. code-block:: bash

    # Using unified CLI
    tff concatenate in1.trk in2.trk merged.trx

    # Using standalone command
    tff_concatenate_tractograms in1.trk in2.trk merged.trx

tff_convert_dsi_studio
~~~~~~~~~~~~~~~~~~~~~~
Convert a DSI Studio ``.trk`` with accompanying ``.nii.gz`` reference into a cleaned ``.trk`` or TRX.

.. code-block:: bash

    # Using unified CLI
    tff convert-dsi input.trk reference.nii.gz cleaned.trk

    # Using standalone command
    tff_convert_dsi_studio input.trk reference.nii.gz cleaned.trk

tff_convert_tractogram
~~~~~~~~~~~~~~~~~~~~~~
General-purpose converter between ``trk``, ``tck``, ``vtk``, ``fib``, ``dpy``, and ``trx``.

- Flags: ``--reference`` for formats needing a NIfTI, ``--positions-dtype``, ``--offsets-dtype``, ``-f`` to overwrite.

.. code-block:: bash

    # Using unified CLI
    tff convert input.trk output.trx --positions-dtype float32 --offsets-dtype uint64

    # Using standalone command
    tff_convert_tractogram input.trk output.trx --positions-dtype float32 --offsets-dtype uint64

tff_generate_trx_from_scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Build a TRX file from raw NumPy arrays or CSV streamline coordinates.

- Flags: ``--positions``, ``--offsets``, ``--positions-dtype``, ``--offsets-dtype``, spatial options (``--space``, ``--origin``), and metadata loaders for dpv/dps/groups/dpg.

.. code-block:: bash

    # Using unified CLI
    tff generate fa.nii.gz output.trx --positions positions.npy --offsets offsets.npy

    # Using standalone command
    tff_generate_trx_from_scratch fa.nii.gz output.trx --positions positions.npy --offsets offsets.npy

tff_manipulate_datatype
~~~~~~~~~~~~~~~~~~~~~~~
Rewrite TRX datasets with new dtypes for positions/offsets/dpv/dps/dpg/groups.

- Accepts per-field dtype arguments and overwrites with ``-f``.

.. code-block:: bash

    # Using unified CLI
    tff manipulate-dtype input.trx output.trx --positions-dtype float16 --dpv color,uint8

    # Using standalone command
    tff_manipulate_datatype input.trx output.trx --positions-dtype float16 --dpv color,uint8

tff_simple_compare
~~~~~~~~~~~~~~~~~~
Compare two tractograms for quick difference checks.

.. code-block:: bash

    # Using unified CLI
    tff compare first.trk second.trk

    # Using standalone command
    tff_simple_compare first.trk second.trk

tff_validate_trx
~~~~~~~~~~~~~~~~
Validate a TRX file for consistency and remove invalid streamlines.

.. code-block:: bash

    # Using unified CLI
    tff validate data.trx --out cleaned.trx

    # Using standalone command
    tff_validate_trx data.trx --out cleaned.trx

tff_verify_header_compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Check whether tractogram headers are compatible for operations such as concatenation.

.. code-block:: bash

    # Using unified CLI
    tff verify-header file1.trk file2.trk

    # Using standalone command
    tff_verify_header_compatibility file1.trk file2.trk

tff_visualize_overlap
~~~~~~~~~~~~~~~~~~~~~
Visualize streamline overlap between tractograms (requires visualization dependencies).

.. code-block:: bash

    # Using unified CLI
    tff visualize tractogram.trk reference.nii.gz

    # Using standalone command
    tff_visualize_overlap tractogram.trk reference.nii.gz

Notes
-----
- Test datasets for examples can be fetched with ``python -m trx.fetcher`` helpers: ``fetch_data(get_testing_files_dict())`` downloads to ``$TRX_HOME`` (default ``~/.tee_ar_ex``).
- All commands print detailed usage with ``--help``.
- The unified ``tff`` CLI uses `Typer <https://typer.tiangolo.com/>`_ for beautiful terminal output with colors and rich formatting.
