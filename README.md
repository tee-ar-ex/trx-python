# trx-python

[![Tests](https://github.com/tee-ar-ex/trx-python/actions/workflows/test.yml/badge.svg)](https://github.com/tee-ar-ex/trx-python/actions/workflows/test.yml)
[![Code Format](https://github.com/tee-ar-ex/trx-python/actions/workflows/codeformat.yml/badge.svg)](https://github.com/tee-ar-ex/trx-python/actions/workflows/codeformat.yml)
[![codecov](https://codecov.io/gh/tee-ar-ex/trx-python/branch/master/graph/badge.svg)](https://codecov.io/gh/tee-ar-ex/trx-python)
[![PyPI version](https://badge.fury.io/py/trx-python.svg)](https://badge.fury.io/py/trx-python)

A Python implementation of the TRX file format for tractography data.

For details, please visit the [documentation](https://tee-ar-ex.github.io/trx-python/).

## Installation

### From PyPI

```bash
pip install trx-python
```

### From Source

```bash
git clone https://github.com/tee-ar-ex/trx-python.git
cd trx-python
pip install .
```

## Quick Start

### Loading and Saving Tractograms

```python
from trx.io import load, save

# Load a tractogram (supports .trx, .trk, .tck, .vtk, .fib, .dpy)
trx = load("tractogram.trx")

# Save to a different format
save(trx, "output.trk")
```

### Command-Line Interface

TRX-Python provides a unified CLI (`trx`) for common operations:

```bash
# Show all available commands
trx --help

# Display TRX file information (header, groups, data keys, archive contents)
trx info data.trx

# Convert between formats
trx convert input.trk output.trx

# Concatenate tractograms
trx concatenate tract1.trx tract2.trx merged.trx

# Validate a TRX file
trx validate data.trx
```

Individual commands are also available for backward compatibility:

```bash
trx_info data.trx
trx_convert_tractogram input.trk output.trx
trx_concatenate_tractograms tract1.trx tract2.trx merged.trx
trx_validate data.trx
```

## Development

We use [spin](https://github.com/scientific-python/spin) for development workflow.

### First-Time Setup

```bash
# Clone the repository (or your fork)
git clone https://github.com/tee-ar-ex/trx-python.git
cd trx-python

# Install with all dependencies
pip install -e ".[all]"

# Set up development environment (fetches upstream tags)
spin setup
```

### Common Commands

```bash
spin setup      # Set up development environment
spin install    # Install in editable mode
spin test       # Run all tests
spin test -m memmap  # Run tests matching pattern
spin lint       # Run linting (ruff)
spin lint --fix # Auto-fix linting issues
spin docs       # Build documentation
spin clean      # Clean temporary files
```

Run `spin` without arguments to see all available commands.

### Code Quality

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check for issues
spin lint

# Auto-fix issues
spin lint --fix

# Format code
ruff format .
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Temporary Directory

The TRX file format uses memory-mapped files to limit RAM usage. When dealing with large files, several gigabytes may be required on disk.

By default, temporary files are stored in:
- Linux/macOS: `/tmp`
- Windows: `C:\WINDOWS\Temp`

To change the directory:

```bash
# Use a specific directory (must exist)
export TRX_TMPDIR=/path/to/tmp

# Use current working directory
export TRX_TMPDIR=use_working_dir
```

Temporary folders are automatically cleaned, but if the code crashes unexpectedly, ensure folders are deleted manually.

## Troubleshooting

If the `trx` command is not working as expected, run `trx --debug` to print diagnostic information about the Python interpreter, package location, and whether all required and optional dependencies are installed.

## Documentation

Full documentation is available at https://tee-ar-ex.github.io/trx-python/

To build locally:

```bash
spin docs --open
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://tee-ar-ex.github.io/trx-python/contributing.html) for details.

## License

BSD License - see [LICENSE](LICENSE) for details.
