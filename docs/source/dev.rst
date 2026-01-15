Developer Guide
===============

This guide provides detailed information for developers working on TRX-Python.

Installation for Development
----------------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.11 or later (Python 3.12+ recommended)
- Git
- pip

Setting Up Your Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the repository**:

   .. code-block:: bash

       # If you're a contributor, fork first then clone your fork
       git clone https://github.com/YOUR_USERNAME/trx-python.git
       cd trx-python

2. **Install with all development dependencies**:

   .. code-block:: bash

       pip install -e ".[all]"

   This installs:

   - Core dependencies (numpy, nibabel, deepdiff, typer)
   - Development tools (spin, setuptools_scm)
   - Documentation tools (sphinx, numpydoc)
   - Style tools (ruff, pre-commit)
   - Testing tools (pytest, pytest-cov)

3. **Set up the development environment**:

   .. code-block:: bash

       spin setup

   This command:

   - Adds upstream remote if missing
   - Fetches version tags for correct ``setuptools_scm`` version detection

Using Spin
----------

We use `spin <https://github.com/scientific-python/spin>`_ for development workflow.
Spin provides a consistent interface for common development tasks.

Available Commands
~~~~~~~~~~~~~~~~~~

Run ``spin`` without arguments to see all available commands:

.. code-block:: bash

    spin

**Setup Commands:**

.. code-block:: bash

    spin setup          # Configure development environment

**Build Commands:**

.. code-block:: bash

    spin install        # Install package in editable mode

**Test Commands:**

.. code-block:: bash

    spin test           # Run all tests
    spin test -m NAME   # Run tests matching pattern
    spin test -v        # Verbose output
    spin lint           # Run ruff linting
    spin lint --fix     # Auto-fix linting issues

**Documentation Commands:**

.. code-block:: bash

    spin docs           # Build documentation
    spin docs --clean   # Clean and rebuild
    spin docs --open    # Build and open in browser

**Cleanup Commands:**

.. code-block:: bash

    spin clean          # Remove temporary files and build artifacts

Code Quality
------------

Linting with Ruff
~~~~~~~~~~~~~~~~~

We use `ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting.
Configuration is in ``ruff.toml``.

.. code-block:: bash

    # Check for issues
    spin lint

    # Auto-fix issues
    spin lint --fix

    # Format code
    ruff format .

    # Check formatting without changes
    ruff format --check .

Pre-commit Hooks
~~~~~~~~~~~~~~~~

We recommend using pre-commit hooks to catch issues before committing:

.. code-block:: bash

    # Install pre-commit hooks
    pre-commit install

    # Run hooks manually on all files
    pre-commit run --all-files

The hooks run:

- ``ruff`` - Linting with auto-fix
- ``ruff-format`` - Code formatting
- ``codespell`` - Spell checking

Testing
-------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    spin test

    # Run tests matching a pattern
    spin test -m memmap

    # Run with pytest directly
    pytest trx/tests

    # Run with coverage
    pytest trx/tests --cov=trx --cov-report=term-missing

Test Data
~~~~~~~~~

Test data is automatically downloaded from Figshare on first run.
Data is cached in ``~/.tee_ar_ex/``.

You can manually fetch test data:

.. code-block:: python

    from trx.fetcher import fetch_data, get_testing_files_dict
    fetch_data(get_testing_files_dict())

Writing Tests
~~~~~~~~~~~~~

- Tests go in ``trx/tests/``
- Use pytest fixtures for setup/teardown
- Use ``pytest.mark.skipif`` for conditional tests

Example:

.. code-block:: python

    import pytest
    import numpy as np
    from numpy.testing import assert_array_equal

    def test_my_function():
        result = my_function(input_data)
        expected = np.array([1, 2, 3])
        assert_array_equal(result, expected)

    @pytest.mark.skipif(not dipy_available, reason="Dipy required")
    def test_with_dipy():
        # Test that requires dipy
        pass

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Build docs
    spin docs

    # Clean build
    spin docs --clean

    # Build and open in browser
    spin docs --open

Documentation is built with Sphinx and uses:

- ``pydata-sphinx-theme`` for styling
- ``sphinx-autoapi`` for API documentation
- ``numpydoc`` for NumPy-style docstrings

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

- Source files are in ``docs/source/``
- Use reStructuredText format
- API docs are auto-generated from docstrings

NumPy Docstring Format
~~~~~~~~~~~~~~~~~~~~~~

All functions and classes should be documented using NumPy-style docstrings:

.. code-block:: python

    def load(filename, reference=None):
        """Load a tractogram file.

        Parameters
        ----------
        filename : str
            Path to the tractogram file.
        reference : str, optional
            Path to reference anatomy for formats that require it.

        Returns
        -------
        tractogram : TrxFile or StatefulTractogram
            The loaded tractogram.

        Raises
        ------
        ValueError
            If the file format is not supported.

        See Also
        --------
        save : Save a tractogram to file.

        Examples
        --------
        >>> from trx.io import load
        >>> trx = load("tractogram.trx")
        """
        pass

Project Structure
-----------------

.. code-block:: text

    trx-python/
    ├── trx/                    # Main package
    │   ├── __init__.py
    │   ├── cli.py              # Command-line interface (Typer)
    │   ├── fetcher.py          # Test data fetching
    │   ├── io.py               # Unified I/O interface
    │   ├── streamlines_ops.py  # Streamline operations
    │   ├── trx_file_memmap.py  # Core TrxFile class
    │   ├── utils.py            # Utility functions
    │   ├── viz.py              # Visualization (optional)
    │   ├── workflows.py        # High-level workflows
    │   └── tests/              # Test suite
    ├── docs/                   # Documentation
    │   └── source/
    ├── .github/                # GitHub Actions workflows
    │   └── workflows/
    ├── .spin/                  # Spin configuration
    │   └── cmds.py
    ├── pyproject.toml          # Project configuration
    ├── ruff.toml               # Ruff configuration
    └── .pre-commit-config.yaml # Pre-commit hooks

Continuous Integration
----------------------

GitHub Actions runs on every push and pull request:

- **test.yml**: Runs tests on Python 3.11-3.13 across Linux, macOS, Windows
- **codeformat.yml**: Checks code formatting with pre-commit/ruff
- **coverage.yml**: Generates code coverage reports
- **docbuild.yml**: Builds and deploys documentation

Environment Variables
---------------------

TRX_TMPDIR
~~~~~~~~~~

Controls where temporary files are stored during memory-mapped operations.

.. code-block:: bash

    # Use a specific directory
    export TRX_TMPDIR=/path/to/tmp

    # Use current working directory
    export TRX_TMPDIR=use_working_dir

Default: System temp directory (``/tmp`` on Linux/macOS, ``C:\WINDOWS\Temp`` on Windows)

Release Process
---------------

Releases are managed via GitHub:

1. Update version in ``pyproject.toml`` if needed
2. Create a GitHub release with appropriate tag
3. CI automatically publishes to PyPI

Version Detection
~~~~~~~~~~~~~~~~~

We use ``setuptools_scm`` for automatic version detection from git tags.
This requires:

- Proper git tags from upstream
- Running ``spin setup`` after cloning a fork
