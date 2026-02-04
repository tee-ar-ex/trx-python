"""TRX file format for brain tractography data."""

try:
    from ._version import __version__  # noqa: F401
except ImportError:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        __version__ = "0+unknown"
    else:
        try:
            __version__ = version("trx-python")
        except PackageNotFoundError:
            __version__ = "0+unknown"
