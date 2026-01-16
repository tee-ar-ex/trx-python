Contributing to TRX-Python
==========================

We welcome contributions from the community! This guide will help you get started
with contributing to the TRX-Python project.

Ways to Contribute
------------------

There are many ways to contribute to TRX-Python:

- **Report bugs**: If you find a bug, please open an issue on GitHub
- **Suggest features**: Have an idea? Open an issue to discuss it
- **Fix bugs**: Look for issues labeled "good first issue" or "help wanted"
- **Write documentation**: Help improve our docs or add examples
- **Write tests**: Increase test coverage
- **Code review**: Review pull requests from other contributors

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork**:

   .. code-block:: bash

       git clone https://github.com/YOUR_USERNAME/trx-python.git
       cd trx-python

3. **Set up development environment**:

   .. code-block:: bash

       pip install -e ".[all]"
       spin setup

   The ``spin setup`` command fetches version tags from upstream, which is
   required for correct version detection.

4. **Create a branch** for your changes:

   .. code-block:: bash

       git checkout -b my-feature-branch

Making Changes
--------------

Development Workflow
~~~~~~~~~~~~~~~~~~~~

We use `spin <https://github.com/scientific-python/spin>`_ for development workflow:

.. code-block:: bash

    spin install    # Install in editable mode
    spin test       # Run all tests
    spin lint       # Run linting (ruff)
    spin docs       # Build documentation

Before Submitting
~~~~~~~~~~~~~~~~~

1. **Run tests** to ensure your changes don't break existing functionality:

   .. code-block:: bash

       spin test

2. **Run linting** to ensure code style compliance:

   .. code-block:: bash

       spin lint

   You can auto-fix many issues with:

   .. code-block:: bash

       spin lint --fix

3. **Format your code** using ruff:

   .. code-block:: bash

       ruff format .

4. **Write tests** for any new functionality

5. **Update documentation** if needed

Submitting a Pull Request
-------------------------

1. **Push your changes** to your fork:

   .. code-block:: bash

       git push origin my-feature-branch

2. **Open a Pull Request** on GitHub against the ``master`` branch

3. **Describe your changes** in the PR description:

   - What does this PR do?
   - Why is this change needed?
   - How was it tested?

4. **Wait for CI checks** to pass

5. **Address review feedback** if requested

Code Style
----------

We follow these conventions:

- **PEP 8** style guide
- **Line length**: 88 characters maximum
- **Docstrings**: NumPy style format
- **Type hints**: Encouraged but not required

Example docstring:

.. code-block:: python

    def my_function(param1, param2):
        """Short description of the function.

        Parameters
        ----------
        param1 : int
            Description of param1.
        param2 : str
            Description of param2.

        Returns
        -------
        result : bool
            Description of return value.

        Examples
        --------
        >>> my_function(1, "test")
        True
        """
        pass

We use `ruff <https://docs.astral.sh/ruff/>`_ for linting and formatting.
Configuration is in ``ruff.toml``.

Testing
-------

Tests are located in ``trx/tests/``. We use pytest for testing.

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

    # Run all tests
    spin test

    # Run tests matching a pattern
    spin test -m memmap

    # Run with verbose output
    spin test -v

    # Run a specific test file
    pytest trx/tests/test_memmap.py

Writing Tests
~~~~~~~~~~~~~

- Place tests in ``trx/tests/``
- Name test files ``test_*.py``
- Name test functions ``test_*``
- Use pytest fixtures for common setup

Documentation
-------------

Documentation is built with Sphinx and hosted on GitHub Pages.

Building Docs
~~~~~~~~~~~~~

.. code-block:: bash

    spin docs              # Build documentation
    spin docs --clean      # Clean build
    spin docs --open       # Build and open in browser

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

- Documentation source is in ``docs/source/``
- Use reStructuredText format
- API documentation is auto-generated from docstrings

Getting Help
------------

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions

Thank you for contributing to TRX-Python!
