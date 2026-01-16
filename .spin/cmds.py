"""Custom spin commands for trx-python development."""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import click

UPSTREAM_URL = "https://github.com/tee-ar-ex/trx-python.git"
UPSTREAM_NAME = "upstream"


def run(cmd, check=True, capture=True):
    """Run a shell command."""
    result = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if check and result.returncode != 0:
        if capture:
            click.echo(f"Error: {result.stderr}", err=True)
        return None
    return result.stdout.strip() if capture else result.returncode


def get_remotes():
    """Get dict of remote names to URLs."""
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

    Examples:
        spin test                    # Run all tests
        spin test -m memmap          # Run tests matching 'memmap'
        spin test -v                 # Verbose output
        spin test -- -x --tb=short   # Pass args to pytest
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

    Examples:
        spin lint        # Run ruff and codespell checks
        spin lint --fix  # Run ruff and auto-fix issues
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
def docs(clean, open_browser):
    """Build documentation using Sphinx.

    Examples:
        spin docs          # Build docs
        spin docs --clean  # Clean and rebuild
        spin docs --open   # Build and open in browser
    """
    import os

    docs_dir = "docs"

    if clean:
        click.echo("Cleaning build directory...")
        build_dir = os.path.join(docs_dir, "_build")
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)

        # Clean sphinx-gallery generated files
        gallery_dir = os.path.join(docs_dir, "source", "auto_examples")
        if os.path.exists(gallery_dir):
            click.echo("Cleaning sphinx-gallery generated files...")
            shutil.rmtree(gallery_dir)

        # Clean sphinx-gallery execution times file
        sg_times = os.path.join(docs_dir, "source", "sg_execution_times.rst")
        if os.path.exists(sg_times):
            os.remove(sg_times)

    click.echo("Building documentation...")
    cmd = ["make", "-C", docs_dir, "html"]
    result = run(cmd, capture=False, check=False)

    if result == 0:
        index_path = os.path.abspath(
            os.path.join(docs_dir, "_build", "html", "index.html")
        )
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
    if os.path.exists(trx_tmp_dir):
        temp_files = glob.glob(os.path.join(trx_tmp_dir, "trx_*"))
        for temp_dir in temp_files:
            if os.path.isdir(temp_dir):
                click.echo(f"Removing temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir)

    # Clean build artifacts
    for build_pattern in ["build", "dist", "*.egg-info"]:
        for path in glob.glob(build_pattern):
            if os.path.isdir(path):
                click.echo(f"Removing build directory: {path}")
                shutil.rmtree(path)
            elif os.path.isfile(path):
                click.echo(f"Removing build file: {path}")
                os.remove(path)

    # Clean Python cache
    for cache_dir in ["**/__pycache__", "**/.pytest_cache"]:
        for path in glob.glob(cache_dir, recursive=True):
            if os.path.isdir(path):
                click.echo(f"Removing cache directory: {path}")
                shutil.rmtree(path)

    click.echo("Cleanup complete!")
