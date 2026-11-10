# -*- coding: utf-8 -*-
"""Storage backend protocol and registry.

A *backend* knows how to read a :class:`~trx.trx_file_memmap.TrxFile` from,
and write one to, a particular on-disk container. The TRX data model
(header, positions, offsets, dpv, dps, groups, dpg) is the same for every
backend; only the container differs.

Today there is one backend, :class:`~trx.backends.memmap.MemmapBackend`,
which implements the TRX specification's zip-or-folder layout with
memory-mapped arrays. The registry exists so that other containers can be
added without touching :mod:`trx.trx_file_memmap`.
"""

from typing import TYPE_CHECKING, Any, List, Protocol, Type, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trx.trx_file_memmap import TrxFile


@runtime_checkable
class TrxBackend(Protocol):
    """What a storage backend must provide.

    All methods are class-level: a backend is stateless and is selected per
    call from the path it is asked to handle.
    """

    #: Short identifier, e.g. ``"memmap"``.
    name: str

    @classmethod
    def can_load(cls, path: str) -> bool:
        """Return True if ``path`` is a container this backend can read."""
        ...

    @classmethod
    def can_save(cls, path: str) -> bool:
        """Return True if ``path`` names a container this backend can write."""
        ...

    @classmethod
    def load(cls, path: str) -> "TrxFile":
        """Read ``path`` and return a :class:`TrxFile`."""
        ...

    @classmethod
    def save(cls, tgm: "TrxFile", path: str, **options: Any) -> None:
        """Write ``tgm`` to ``path``."""
        ...


_REGISTRY: List[Type[TrxBackend]] = []


def register_backend(backend: Type[TrxBackend]) -> Type[TrxBackend]:
    """Register a backend class; usable as a class decorator.

    Backends are consulted in registration order, so the default memmap
    backend, registered first, keeps precedence for ``.trx`` files.
    """
    if backend not in _REGISTRY:
        _REGISTRY.append(backend)
    return backend


def registered_backends() -> List[Type[TrxBackend]]:
    """Return the registered backends in precedence order."""
    return list(_REGISTRY)


def backend_for_loading(path: str) -> Type[TrxBackend]:
    """Return the first registered backend whose :meth:`can_load` accepts ``path``.

    Raises
    ------
    ValueError
        If no backend recognises ``path``.
    """
    for backend in _REGISTRY:
        if backend.can_load(path):
            return backend
    raise ValueError(f"No TRX storage backend can read {path!r}.")


def backend_for_saving(path: str) -> Type[TrxBackend]:
    """Return the first registered backend whose :meth:`can_save` accepts ``path``.

    Raises
    ------
    ValueError
        If no backend recognises the destination.
    """
    for backend in _REGISTRY:
        if backend.can_save(path):
            return backend
    raise ValueError("Unsupported extension.")
