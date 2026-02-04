"""TRX file format for brain tractography data."""

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    try:
        from importlib.metadata import version

        __version__ = version("trx-python")
    except Exception:
        __version__ = "unknown"
