"""TRX file format for brain tractography data."""

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("trx-python")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
