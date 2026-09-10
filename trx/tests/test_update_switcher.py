# -*- coding: utf-8 -*-

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
SCRIPT = TOOLS_DIR / "update_switcher.py"


def _load_module():
    """Import update_switcher.py, which lives outside any importable package.

    Returns
    -------
    module
        The loaded ``update_switcher`` module.
    """
    spec = importlib.util.spec_from_file_location("update_switcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


switcher = _load_module()


def _preferred(entries):
    """Return the single preferred entry of a switcher list.

    Parameters
    ----------
    entries : list
        List of switcher entries.

    Returns
    -------
    dict
        The entry flagged as preferred.
    """
    preferred = [e for e in entries if e.get("preferred")]
    assert len(preferred) == 1
    return preferred[0]


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.3", (0, 3)),
        ("0.5.0", (0, 5, 0)),
        ("0.5.1.dev12+gabc123", (0, 5, 1)),
        ("dev", None),
        ("stable", None),
        ("", None),
    ],
)
def test_parse_version(version, expected):
    assert switcher.parse_version(version) == expected


def test_build_switcher_orders_newest_first():
    entries = switcher.build_switcher(["0.3", "0.5.0", "0.4.0"])

    assert [e["version"] for e in entries] == ["dev", "0.5.0", "0.4.0", "0.3"]


def test_build_switcher_promotes_newest_to_stable():
    entries = switcher.build_switcher(["0.3", "0.4.0", "0.5.0"])
    stable = _preferred(entries)

    assert stable["version"] == "0.5.0"
    assert stable["name"] == "0.5.0 (stable)"
    assert stable["url"] == f"{switcher.BASE_URL}/stable/"


def test_no_entry_uses_stable_as_a_version():
    # Regression test for gh-141: a preferred entry whose version is the
    # unparsable string "stable" makes the theme show the version warning
    # banner on every page, including the stable docs themselves.
    entries = switcher.build_switcher(["0.4.0", "0.5.0"])

    assert all(e["version"] != "stable" for e in entries)


def test_dev_entry_is_first_and_never_preferred():
    entries = switcher.build_switcher(["0.4.0", "0.5.0"])

    assert entries[0]["version"] == "dev"
    assert entries[0]["url"] == f"{switcher.BASE_URL}/dev/"
    assert "preferred" not in entries[0]


def test_older_releases_keep_their_versioned_url():
    entries = switcher.build_switcher(["0.4.0", "0.5.0"])
    old = next(e for e in entries if e["version"] == "0.4.0")

    assert old["name"] == "0.4.0"
    assert old["url"] == f"{switcher.BASE_URL}/0.4.0/"
    assert "preferred" not in old


def test_add_versions_demotes_previous_stable():
    entries = switcher.build_switcher(["0.4.0"])
    entries = switcher.add_versions(entries, ["0.5.0"])

    assert _preferred(entries)["version"] == "0.5.0"
    old = next(e for e in entries if e["version"] == "0.4.0")
    assert old["url"] == f"{switcher.BASE_URL}/0.4.0/"


def test_add_versions_keeps_stable_on_a_backport():
    entries = switcher.build_switcher(["0.4.0", "0.5.0"])
    entries = switcher.add_versions(entries, ["0.4.1"])

    assert _preferred(entries)["version"] == "0.5.0"
    assert any(e["version"] == "0.4.1" for e in entries)


def test_add_versions_is_idempotent():
    once = switcher.add_versions(switcher.build_switcher(["0.4.0"]), ["0.5.0"])
    twice = switcher.add_versions(once, ["0.5.0"])

    assert once == twice


def test_add_versions_ignores_the_legacy_stable_entry():
    legacy = [
        {"name": "dev", "version": "dev", "url": f"{switcher.BASE_URL}/dev/"},
        {"name": "0.4.0", "version": "0.4.0", "url": f"{switcher.BASE_URL}/0.4.0/"},
        {
            "name": "stable",
            "version": "stable",
            "url": f"{switcher.BASE_URL}/stable/",
            "preferred": True,
        },
    ]
    entries = switcher.add_versions(legacy, ["0.5.0"])

    assert all(e["version"] != "stable" for e in entries)
    assert _preferred(entries)["version"] == "0.5.0"


@pytest.mark.parametrize(
    "version, expected",
    [("0.6.0", True), ("0.5.0", True), ("0.4.1", False), ("dev", False)],
)
def test_is_latest(version, expected):
    entries = switcher.build_switcher(["0.3", "0.4.0", "0.5.0"])
    entries = switcher.add_versions(entries, [version])

    assert switcher.is_latest(entries, version) is expected


def test_rebuild_matches_incremental_adds(tmp_path):
    incremental = switcher.build_switcher([])
    for version in ["0.3", "0.4.0", "0.5.0"]:
        incremental = switcher.add_versions(incremental, [version])

    rebuilt = switcher.build_switcher(["0.5.0", "0.3", "0.4.0"])

    assert incremental == rebuilt


def test_cli_writes_switcher_and_is_latest_output(tmp_path):
    path = tmp_path / "switcher.json"
    output = tmp_path / "github_output"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(path),
            "--rebuild",
            "--version",
            "0.4.0",
            "--version",
            "0.5.0",
            "--github-output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )

    entries = json.loads(path.read_text())
    assert _preferred(entries)["version"] == "0.5.0"
    assert output.read_text() == "is_latest=true\n"


def test_cli_reports_a_backport_as_not_latest(tmp_path):
    path = tmp_path / "switcher.json"
    output = tmp_path / "github_output"
    switcher.save_switcher(path, switcher.build_switcher(["0.4.0", "0.5.0"]))

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(path),
            "--version",
            "0.4.1",
            "--github-output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )

    assert output.read_text() == "is_latest=false\n"
    assert _preferred(json.loads(path.read_text()))["version"] == "0.5.0"
