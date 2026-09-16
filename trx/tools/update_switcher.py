#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Update switcher.json for documentation version switching.

This script maintains the version switcher JSON file used by pydata-sphinx-theme
to enable users to switch between different documentation versions.
"""

import argparse
import json
from pathlib import Path
import re
import sys

BASE_URL = "https://tee-ar-ex.github.io/trx-python"
DEV_VERSION = "dev"
STABLE_SUFFIX = " (stable)"

_NUMERIC_PREFIX = re.compile(r"^(\d+(?:\.\d+)*)")


def parse_version(version):
    """Convert a version string into a comparable tuple of integers.

    Only the leading dotted numeric part is considered, so pre-release and local
    suffixes are ignored. Strings without such a prefix (``"dev"`` for instance)
    are not releases and therefore have no ordering.

    Parameters
    ----------
    version : str
        Version string to parse (e.g., ``"0.5.0"`` or ``"0.3"``).

    Returns
    -------
    tuple of int or None
        Tuple of integers suitable for sorting, or None when the string does not
        start with a dotted numeric version.
    """
    match = _NUMERIC_PREFIX.match(version or "")
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def load_switcher(path):
    """Load existing switcher.json or return empty list.

    Parameters
    ----------
    path : str or Path
        Path to the switcher.json file.

    Returns
    -------
    list
        List of version entries from the switcher file.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_switcher(path, versions):
    """Save switcher.json with proper formatting.

    Parameters
    ----------
    path : str or Path
        Path to the switcher.json file.
    versions : list
        List of version entries to write.
    """
    with open(path, "w") as f:
        json.dump(versions, f, indent=4)
        f.write("\n")


def release_versions(versions):
    """Extract the sorted release versions held by a switcher list.

    Parameters
    ----------
    versions : list
        List of version entries.

    Returns
    -------
    list of str
        Release version strings, newest first. The dev entry is excluded.
    """
    releases = [
        v.get("version")
        for v in versions
        if parse_version(v.get("version", "")) is not None
    ]
    return sorted(set(releases), key=parse_version, reverse=True)


def is_latest(versions, version):
    """Tell whether a version is the newest release known to the switcher.

    Used to decide whether a tag should be promoted to ``stable``, so that a
    backport tag published after a newer release does not demote it.

    Parameters
    ----------
    versions : list
        List of version entries.
    version : str
        Version string to test (e.g., ``"0.5.0"``).

    Returns
    -------
    bool
        True when no known release sorts above ``version``.
    """
    key = parse_version(version)
    if key is None:
        return False
    return all(key >= parse_version(other) for other in release_versions(versions))


def build_switcher(releases):
    """Build a complete switcher list from a set of release versions.

    The newest release is the preferred entry and is served from the ``stable``
    alias; every other release keeps its own versioned URL. The dev entry comes
    first and is never preferred.

    Parameters
    ----------
    releases : iterable of str
        Release version strings, in any order. Entries that are not dotted
        numeric versions are ignored.

    Returns
    -------
    list
        Fully formed list of switcher entries.
    """
    ordered = sorted(
        {r for r in releases if parse_version(r) is not None},
        key=parse_version,
        reverse=True,
    )

    entries = [
        {
            "name": DEV_VERSION,
            "version": DEV_VERSION,
            "url": f"{BASE_URL}/{DEV_VERSION}/",
        }
    ]

    for index, release in enumerate(ordered):
        if index == 0:
            entries.append(
                {
                    "name": f"{release}{STABLE_SUFFIX}",
                    "version": release,
                    "url": f"{BASE_URL}/stable/",
                    "preferred": True,
                }
            )
        else:
            entries.append(
                {
                    "name": release,
                    "version": release,
                    "url": f"{BASE_URL}/{release}/",
                }
            )

    return entries


def add_versions(versions, new_versions):
    """Add releases to an existing switcher list and rebuild it.

    Parameters
    ----------
    versions : list
        List of existing version entries.
    new_versions : iterable of str
        Release versions to add (e.g., ``["0.5.0"]``).

    Returns
    -------
    list
        Rebuilt list of version entries.
    """
    return build_switcher([*release_versions(versions), *new_versions])


def main():
    """Run the switcher update workflow.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        description="Update switcher.json for documentation version switching"
    )
    parser.add_argument("switcher_path", type=Path, help="Path to switcher.json file")
    parser.add_argument(
        "--version",
        type=str,
        action="append",
        default=[],
        dest="versions",
        help="Release version to add (e.g., 0.5.0). Repeatable; the last "
        "one given is the one reported by --github-output.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore the existing file and rebuild it from --version values only",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        help="Path of a GitHub Actions output file to append is_latest to",
    )

    args = parser.parse_args()

    existing = [] if args.rebuild else load_switcher(args.switcher_path)
    versions = add_versions(existing, args.versions)

    latest = is_latest(versions, args.versions[-1]) if args.versions else False

    save_switcher(args.switcher_path, versions)

    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write(f"is_latest={str(latest).lower()}\n")

    print(f"Updated {args.switcher_path} (is_latest={str(latest).lower()}):")
    print(json.dumps(versions, indent=4))

    return 0


if __name__ == "__main__":
    sys.exit(main())
