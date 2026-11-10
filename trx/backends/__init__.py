# -*- coding: utf-8 -*-
"""Storage backends for TRX data.

The TRX data model is container-independent. A backend maps that model onto
one on-disk container. :class:`MemmapBackend` implements the TRX
specification's zip-or-folder layout and is registered by default.

Importing this package registers the default backend.
"""

from trx.backends.base import (
    TrxBackend,
    backend_for_loading,
    backend_for_saving,
    register_backend,
    registered_backends,
)
from trx.backends.memmap import MemmapBackend

__all__ = [
    "MemmapBackend",
    "TrxBackend",
    "backend_for_loading",
    "backend_for_saving",
    "register_backend",
    "registered_backends",
]
