"""Custom spin commands for trx-python development."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import click

UPSTREAM_URL = "https://github.com/tee-ar-ex/trx-python.git"
UPSTREAM_NAME = "upstream"


def run(cmd, check=True, capture=True):
    """Run a shell command.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to execute.
    check : bool, optional
        If True, check the return code and report errors.
    capture : bool, optional
        If True, capture stdout and stderr.

    Returns
    -------
    str or int or None
        Captured stdout string, return code, or None on error.
    """
    result = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if check and result.returncode != 0:
        if capture:
            click.echo(f"Error: {result.stderr}", err=True)
        return None
    return result.stdout.strip() if capture else result.returncode


def get_remotes():
    """Get dict of remote names to URLs.

    Returns
    -------
    dict
        Mapping of remote names to their fetch URLs.
    """
    output = run(["git", "remote", "-v"])
    if not output:
        return {}
    remotes = {}
    for line in output.split("\n"):
        if "(fetch)" in line:
            parts = line.split()
            remotes[parts[0]] = parts[1]
    return remotes


@click.command()
def setup():
    """Set up development environment (fetch tags from upstream).

    This command configures your fork for development by:
    1. Adding the upstream remote if not present
    2. Fetching tags from upstream (required for correct version detection)

    Run this once after cloning your fork.
    """
    click.echo("Setting up trx-python development environment...\n")

    # Check if in git repo
    if run(["git", "rev-parse", "--git-dir"], check=False) is None:
        click.echo("Error: Not in a git repository", err=True)
        sys.exit(1)

    # Check/add upstream remote
    remotes = get_remotes()
    upstream_remote = None

    for name, url in remotes.items():
        if UPSTREAM_URL.rstrip(".git") in url.rstrip(".git"):
            upstream_remote = name
            click.echo(f"Found upstream remote: {name}")
            break

    if upstream_remote is None:
        click.echo(f"Adding upstream remote: {UPSTREAM_URL}")
        run(["git", "remote", "add", UPSTREAM_NAME, UPSTREAM_URL])
        upstream_remote = UPSTREAM_NAME

    # Fetch tags
    click.echo(f"\nFetching tags from {upstream_remote}...")
    run(["git", "fetch", upstream_remote, "--tags"], capture=False)

    # Verify version
    click.echo("\nVerifying version detection...")
    try:
        from setuptools_scm import get_version

        version = get_version()
        click.echo(f"Detected version: {version}")

        # Check for suspicious version patterns
        if version.startswith("0.0"):
            click.echo(
                "\nWarning: Version starts with 0.0 - tags may not be fetched.",
                err=True,
            )
            sys.exit(1)
    except ImportError:
        click.echo("Note: Install setuptools_scm to verify version detection")

    click.echo("\nSetup complete! You can now run:")
    click.echo("  spin install    # Install in development mode")
    click.echo("  spin test       # Run tests")


@click.command()
@click.option(
    "-m",
    "--match",
    "pattern",
    default=None,
    help="Only run tests matching this pattern (passed to pytest -k)",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Verbose output")
@click.argument("pytest_args", nargs=-1)
def test(pattern, verbose, pytest_args):
    """Run tests using pytest.

    Additional arguments are passed directly to pytest.

    Parameters
    ----------
    pattern : str or None
        Only run tests matching this pattern (passed to pytest -k).
    verbose : bool
        If True, enable verbose output.
    pytest_args : tuple
        Additional arguments passed directly to pytest.
    """
    cmd = ["pytest", "trx/tests"]

    if pattern:
        cmd.extend(["-k", pattern])

    if verbose:
        cmd.append("-v")

    if pytest_args:
        cmd.extend(pytest_args)

    click.echo(f"Running: {' '.join(cmd)}\n")
    sys.exit(run(cmd, capture=False, check=False))


@click.command()
@click.option(
    "--fix", is_flag=True, default=False, help="Automatically fix issues where possible"
)
def lint(fix):
    """Run linting checks using ruff and codespell.

    Parameters
    ----------
    fix : bool
        If True, automatically fix issues where possible.
    """
    click.echo("Running ruff linter...")
    cmd = ["ruff", "check", "."]

    if fix:
        cmd.append("--fix")

    result = run(cmd, capture=False, check=False)
    if result != 0:
        click.echo("\nLinting issues found!", err=True)
        sys.exit(1)

    click.echo("\nRunning ruff formatter check...")
    cmd_format = ["ruff", "format", "--check", "."]
    result = run(cmd_format, capture=False, check=False)
    if result != 0:
        click.echo("\nFormatting issues found!", err=True)
        sys.exit(1)

    click.echo("\nRunning codespell...")
    cmd_spell = [
        "codespell",
        "--skip",
        "*.pyc,.git,pyproject.toml,./docs/_build/*,*.egg-info,./build/*,./dist/*,./tmp/*",
        "trx",
        "docs/source",
        ".spin",
    ]
    result = run(cmd_spell, capture=False, check=False)
    if result != 0:
        click.echo("\nSpelling issues found!", err=True)
        sys.exit(1)

    click.echo("\nAll checks passed!")


@click.command()
@click.option(
    "--clean", is_flag=True, default=False, help="Clean build directory before building"
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    default=False,
    help="Open documentation in browser after building",
)
def docs(_clean, open_browser):
    """Build documentation using Sphinx.

    Parameters
    ----------
    _clean : bool
        If True, clean build directory before building.
    open_browser : bool
        If True, open documentation in browser after building.
    """

    docs_dir = Path("docs")

    if _clean:
        click.echo("Cleaning build directory...")
        build_dir = docs_dir / "_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)

        # Clean sphinx-gallery generated files
        gallery_dir = docs_dir / "source" / "auto_examples"
        if gallery_dir.exists():
            click.echo("Cleaning sphinx-gallery generated files...")
            shutil.rmtree(gallery_dir)

        # Clean sphinx-gallery execution times file
        sg_times = docs_dir / "source" / "sg_execution_times.rst"
        if sg_times.exists():
            os.remove(sg_times)

    click.echo("Building documentation...")
    cmd = ["make", "-C", str(docs_dir), "html"]
    result = run(cmd, capture=False, check=False)

    if result == 0:
        index_path = (docs_dir / "_build" / "html" / "index.html").resolve()
        click.echo("\nDocs built successfully!")
        click.echo(f"Open: {index_path}")

        if open_browser:
            import webbrowser

            webbrowser.open(f"file://{index_path}")

    sys.exit(result)


@click.command()
def clean():  # noqa: C901
    """Clean up temporary files and build artifacts."""
    click.echo("Cleaning up temporary files...")

    # Clean TRX temp directory
    trx_tmp_dir = os.getenv("TRX_TMPDIR", tempfile.gettempdir())
    if trx_tmp_dir.exists():
        for temp_name in trx_tmp_dir.glob("trx_*"):
            if temp_name.is_dir():
                click.echo(f"Removing temporary directory: {temp_name}")
                shutil.rmtree(temp_name)

    # Clean build artifacts
    for build_pattern in ["build", "dist", "*.egg-info"]:
        for path in Path(".").glob(build_pattern):
            if path.is_dir():
                click.echo(f"Removing build directory: {path}")
                shutil.rmtree(path)
            elif path.is_file():
                click.echo(f"Removing build file: {path}")
                path.unlink()

    # Clean Python cache
    for cache_dir in ["**/__pycache__", "**/.pytest_cache"]:
        for path in Path(".").glob(cache_dir):
            if path.is_dir():
                click.echo(f"Removing cache directory: {path}")
                shutil.rmtree(path)

    click.echo("Cleanup complete!")
