#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Update switcher.json for documentation version switching.

This script maintains the version switcher JSON file used by pydata-sphinx-theme
to enable users to switch between different documentation versions.
"""
import argparse
import json
import sys
from pathlib import Path

BASE_URL = "https://tee-ar-ex.github.io/trx-python"


def load_switcher(path):
    """Load existing switcher.json or return empty list."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_switcher(path, versions):
    """Save switcher.json with proper formatting."""
    with open(path, 'w') as f:
        json.dump(versions, f, indent=4)
        f.write('\n')


def ensure_dev_entry(versions):
    """Ensure dev entry exists in versions list."""
    dev_exists = any(v.get('version') == 'dev' for v in versions)
    if not dev_exists:
        versions.insert(0, {
            "name": "dev",
            "version": "dev",
            "url": f"{BASE_URL}/dev/"
        })
    return versions


def ensure_stable_entry(versions):
    """Ensure stable entry exists with preferred flag."""
    stable_idx = next(
        (i for i, v in enumerate(versions) if v.get('version') == 'stable'),
        None
    )
    if stable_idx is not None:
        versions[stable_idx]['preferred'] = True
    else:
        versions.append({
            "name": "stable",
            "version": "stable",
            "url": f"{BASE_URL}/stable/",
            "preferred": True
        })
    return versions


def add_version(versions, version):
    """Add a new version entry to the versions list.

    Parameters
    ----------
    versions : list
        List of version entries.
    version : str
        Version string to add (e.g., "0.5.0").

    Returns
    -------
    list
        Updated versions list.
    """
    # Remove 'preferred' from all existing entries
    for v in versions:
        v.pop('preferred', None)

    # Check if this version already exists
    version_exists = any(v.get('version') == version for v in versions)

    if not version_exists:
        new_entry = {
            "name": version,
            "version": version,
            "url": f"{BASE_URL}/{version}/"
        }
        # Find dev entry index to insert after it
        dev_idx = next(
            (i for i, v in enumerate(versions) if v.get('version') == 'dev'),
            -1
        )
        if dev_idx >= 0:
            versions.insert(dev_idx + 1, new_entry)
        else:
            versions.insert(0, new_entry)

    return versions


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Update switcher.json for documentation version switching'
    )
    parser.add_argument(
        'switcher_path',
        type=Path,
        help='Path to switcher.json file'
    )
    parser.add_argument(
        '--version',
        type=str,
        help='New version to add (e.g., 0.5.0)'
    )

    args = parser.parse_args()

    # Load existing versions
    versions = load_switcher(args.switcher_path)

    # Add new version if specified
    if args.version:
        versions = add_version(versions, args.version)

    # Ensure required entries exist
    versions = ensure_dev_entry(versions)
    versions = ensure_stable_entry(versions)

    # Save updated switcher.json
    save_switcher(args.switcher_path, versions)

    # Print result for CI logs
    print(f"Updated {args.switcher_path}:")
    print(json.dumps(versions, indent=4))

    return 0


if __name__ == '__main__':
    sys.exit(main())
