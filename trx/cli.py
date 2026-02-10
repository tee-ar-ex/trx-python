# -*- coding: utf-8 -*-
"""
TRX Command Line Interface.

This module provides a unified CLI for all TRX file format operations using Typer.
"""

from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from trx.io import load, save
from trx.trx_file_memmap import TrxFile, concatenate, load as load_trx
from trx.workflows import (
    convert_dsi_studio,
    convert_tractogram,
    generate_trx_from_scratch,
    manipulate_trx_datatype,
    tractogram_simple_compare,
    tractogram_visualize_overlap,
    validate_tractogram,
    verify_header_compatibility,
)


def _debug_callback(value: bool) -> None:
    """Print environment and dependency diagnostics, then exit.

    Parameters
    ----------
    value : bool
        Whether the ``--debug`` flag was passed.
    """
    if not value:
        return

    import importlib.metadata
    import importlib.util
    import sys

    from trx import __version__

    typer.echo("Environment diagnostics:")
    typer.echo(f"  Python executable : {sys.executable}")
    typer.echo(f"  sys.prefix        : {sys.prefix}")
    typer.echo(f"  trx-python version: {__version__}")

    trx_spec = importlib.util.find_spec("trx")
    trx_location = (
        trx_spec.submodule_search_locations[0]
        if trx_spec and trx_spec.submodule_search_locations
        else "unknown"
    )
    typer.echo(f"  trx package       : {trx_location}")

    # Read required dependencies from package metadata
    required_deps = []
    try:
        import re

        for req in importlib.metadata.requires("trx-python") or []:
            # Skip optional / extra deps (they contain "; extra ==")
            if "extra ==" in req:
                continue
            # Extract the package name (strip version specifiers like >=, <=, ~=)
            dep_name = re.split(r"[>=<!~;\s]", req)[0]
            required_deps.append(dep_name)
    except importlib.metadata.PackageNotFoundError:
        required_deps = ["deepdiff", "nibabel", "numpy", "typer"]

    optional_deps = ["dipy", "fury", "vtk"]

    typer.echo("\nRequired dependencies:")
    for dep in required_deps:
        spec = importlib.util.find_spec(dep)
        if spec is None:
            typer.echo(f"  {dep:12s} NOT FOUND")
        else:
            typer.echo(f"  {dep:12s} found")

    typer.echo("\nOptional dependencies:")
    for dep in optional_deps:
        spec = importlib.util.find_spec(dep)
        if spec is None:
            typer.echo(f"  {dep:12s} not found")
        else:
            typer.echo(f"  {dep:12s} found")

    raise typer.Exit()


def _main_callback(
    _debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Print environment and dependency diagnostics.",
            callback=_debug_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """TRX File Format Tools - CLI for brain tractography data manipulation.

    Parameters
    ----------
    _debug : bool, optional
        If True, print environment and dependency diagnostics and exit.
    """


app = typer.Typer(
    name="trx",
    help="TRX File Format Tools - CLI for brain tractography data manipulation.",
    add_completion=False,
    rich_markup_mode="rich",
    callback=_main_callback,
)


def _check_overwrite(filepath: Path, overwrite: bool) -> None:
    """Check if file exists and raise error if overwrite is not enabled.

    Parameters
    ----------
    filepath : Path
        Path to the output file.
    overwrite : bool
        If True, allow overwriting existing files.

    Raises
    ------
    typer.Exit
        If file exists and overwrite is False.
    """
    if filepath.is_file() and not overwrite:
        typer.echo(
            typer.style(
                f"Error: {filepath} already exists. Use --force to overwrite.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)


@app.command("concatenate")
def concatenate_tractograms(
    in_tractograms: Annotated[
        List[Path],
        typer.Argument(
            help="Input tractogram files. Format: trk, tck, vtk, fib, dpy, trx.",
        ),
    ],
    out_tractogram: Annotated[
        Path,
        typer.Argument(help="Output filename for the concatenated tractogram."),
    ],
    delete_dpv: Annotated[
        bool,
        typer.Option(
            "--delete-dpv",
            help="Delete data_per_vertex if not all inputs have the same metadata.",
        ),
    ] = False,
    delete_dps: Annotated[
        bool,
        typer.Option(
            "--delete-dps",
            help="Delete data_per_streamline if not all inputs have the same metadata.",
        ),
    ] = False,
    delete_groups: Annotated[
        bool,
        typer.Option(
            "--delete-groups",
            help="Delete groups if not all inputs have the same metadata.",
        ),
    ] = False,
    reference: Annotated[
        Optional[Path],
        typer.Option(
            "--reference",
            "-r",
            help="Reference anatomy for tck/vtk/fib/dpy files (.nii or .nii.gz).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force overwriting of output files."),
    ] = False,
) -> None:
    """Concatenate multiple tractograms into one.

    If the data_per_point or data_per_streamline is not the same for all
    tractograms, the data must be deleted first using the appropriate flags.

    Parameters
    ----------
    in_tractograms : list of Path
        Input tractogram files (.trk, .tck, .vtk, .fib, .dpy, .trx).
    out_tractogram : Path
        Output filename for the concatenated tractogram.
    delete_dpv : bool, optional
        Delete ``data_per_vertex`` if metadata differ across inputs.
    delete_dps : bool, optional
        Delete ``data_per_streamline`` if metadata differ across inputs.
    delete_groups : bool, optional
        Delete groups when metadata differ across inputs.
    reference : Path or None, optional
        Reference anatomy for tck/vtk/fib/dpy inputs.
    force : bool, optional
        Overwrite output if it already exists.

    Returns
    -------
    None
        Writes the concatenated tractogram to ``out_tractogram``.
    """
    _check_overwrite(out_tractogram, force)

    ref = str(reference) if reference else None

    trx_list = []
    has_group = False
    for filename in in_tractograms:
        tractogram_obj = load(str(filename), ref)

        if not isinstance(tractogram_obj, TrxFile):
            tractogram_obj = TrxFile.from_sft(tractogram_obj)
        elif len(tractogram_obj.groups):
            has_group = True
        trx_list.append(tractogram_obj)

    trx = concatenate(
        trx_list,
        delete_dpv=delete_dpv,
        delete_dps=delete_dps,
        delete_groups=delete_groups or not has_group,
        check_space_attributes=True,
        preallocation=False,
    )
    save(trx, str(out_tractogram))

    typer.echo(
        typer.style(
            f"Successfully concatenated {len(in_tractograms)} tractograms "
            f"to {out_tractogram}",
            fg=typer.colors.GREEN,
        )
    )


@app.command("convert")
def convert(
    in_tractogram: Annotated[
        Path,
        typer.Argument(help="Input tractogram. Format: trk, tck, vtk, fib, dpy, trx."),
    ],
    out_tractogram: Annotated[
        Path,
        typer.Argument(help="Output tractogram. Format: trk, tck, vtk, fib, dpy, trx."),
    ],
    reference: Annotated[
        Optional[Path],
        typer.Option(
            "--reference",
            "-r",
            help="Reference anatomy for tck/vtk/fib/dpy files (.nii or .nii.gz).",
        ),
    ] = None,
    positions_dtype: Annotated[
        str,
        typer.Option(
            "--positions-dtype",
            help="Datatype for positions in TRX output.",
        ),
    ] = "float32",
    offsets_dtype: Annotated[
        str,
        typer.Option(
            "--offsets-dtype",
            help="Datatype for offsets in TRX output.",
        ),
    ] = "uint64",
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force overwriting of output files."),
    ] = False,
) -> None:
    """Convert tractograms between formats.

    Supports conversion of .tck, .trk, .fib, .vtk, .trx and .dpy files.
    TCK files always need a reference NIFTI file for conversion.

    Parameters
    ----------
    in_tractogram : Path
        Input tractogram file.
    out_tractogram : Path
        Output tractogram path.
    reference : Path or None, optional
        Reference anatomy required for some input formats.
    positions_dtype : str, optional
        Datatype for positions in TRX output.
    offsets_dtype : str, optional
        Datatype for offsets in TRX output.
    force : bool, optional
        Overwrite output if it already exists.

    Returns
    -------
    None
        Writes the converted tractogram to disk.
    """
    _check_overwrite(out_tractogram, force)

    ref = str(reference) if reference else None
    convert_tractogram(
        str(in_tractogram),
        str(out_tractogram),
        ref,
        pos_dtype=positions_dtype,
        offsets_dtype=offsets_dtype,
    )

    typer.echo(
        typer.style(
            f"Successfully converted {in_tractogram} to {out_tractogram}",
            fg=typer.colors.GREEN,
        )
    )


@app.command("convert-dsi")
def convert_dsi(
    in_dsi_tractogram: Annotated[
        Path,
        typer.Argument(help="Input tractogram from DSI Studio (.trk)."),
    ],
    in_dsi_fa: Annotated[
        Path,
        typer.Argument(help="Input FA from DSI Studio (.nii.gz)."),
    ],
    out_tractogram: Annotated[
        Path,
        typer.Argument(help="Output tractogram file."),
    ],
    remove_invalid: Annotated[
        bool,
        typer.Option(
            "--remove-invalid",
            help="Remove streamlines landing out of the bounding box.",
        ),
    ] = False,
    keep_invalid: Annotated[
        bool,
        typer.Option(
            "--keep-invalid",
            help="Keep streamlines landing out of the bounding box.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force overwriting of output files."),
    ] = False,
) -> None:
    """Convert a DSI-Studio TRK file to TRX or TRK and fix space metadata.

    Parameters
    ----------
    in_dsi_tractogram : Path
        Input DSI-Studio tractogram (.trk or .trk.gz).
    in_dsi_fa : Path
        FA volume used as reference (.nii.gz).
    out_tractogram : Path
        Output tractogram path (.trx or .trk).
    remove_invalid : bool, optional
        Remove streamlines outside the bounding box. Defaults to False.
    keep_invalid : bool, optional
        Keep streamlines outside the bounding box. Defaults to False.
    force : bool, optional
        Overwrite output if it already exists.

    Returns
    -------
    None
        Writes the converted tractogram to disk.
    """
    _check_overwrite(out_tractogram, force)

    if remove_invalid and keep_invalid:
        typer.echo(
            typer.style(
                "Error: Cannot use both --remove-invalid and --keep-invalid.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    convert_dsi_studio(
        str(in_dsi_tractogram),
        str(in_dsi_fa),
        str(out_tractogram),
        remove_invalid=remove_invalid,
        keep_invalid=keep_invalid,
    )

    typer.echo(
        typer.style(
            f"Successfully converted DSI-Studio tractogram to {out_tractogram}",
            fg=typer.colors.GREEN,
        )
    )


@app.command("generate")
def generate(
    reference: Annotated[
        Path,
        typer.Argument(help="Reference anatomy (.nii or .nii.gz)."),
    ],
    out_tractogram: Annotated[
        Path,
        typer.Argument(help="Output tractogram. Format: trk, tck, vtk, fib, dpy, trx."),
    ],
    positions: Annotated[
        Optional[Path],
        typer.Option(
            "--positions",
            help="Binary file with streamline coordinates (Nx3 .npy).",
        ),
    ] = None,
    offsets: Annotated[
        Optional[Path],
        typer.Option(
            "--offsets",
            help="Binary file with streamline offsets (.npy).",
        ),
    ] = None,
    positions_csv: Annotated[
        Optional[Path],
        typer.Option(
            "--positions-csv",
            help="CSV file with streamline coordinates (x1,y1,z1,x2,y2,z2,...).",
        ),
    ] = None,
    space: Annotated[
        str,
        typer.Option(
            "--space",
            help="Coordinate space. Non-default requires Dipy.",
        ),
    ] = "RASMM",
    origin: Annotated[
        str,
        typer.Option(
            "--origin",
            help="Coordinate origin. Non-default requires Dipy.",
        ),
    ] = "NIFTI",
    positions_dtype: Annotated[
        str,
        typer.Option("--positions-dtype", help="Datatype for positions."),
    ] = "float32",
    offsets_dtype: Annotated[
        str,
        typer.Option("--offsets-dtype", help="Datatype for offsets."),
    ] = "uint64",
    dpv: Annotated[
        Optional[List[str]],
        typer.Option(
            "--dpv",
            help="Data per vertex: FILE,DTYPE (e.g., color.npy,uint8).",
        ),
    ] = None,
    dps: Annotated[
        Optional[List[str]],
        typer.Option(
            "--dps",
            help="Data per streamline: FILE,DTYPE (e.g., algo.npy,uint8).",
        ),
    ] = None,
    groups: Annotated[
        Optional[List[str]],
        typer.Option(
            "--groups",
            help="Groups: FILE,DTYPE (e.g., AF_L.npy,int32).",
        ),
    ] = None,
    dpg: Annotated[
        Optional[List[str]],
        typer.Option(
            "--dpg",
            help="Data per group: GROUP,FILE,DTYPE (e.g., AF_L,mean_fa.npy,float32).",
        ),
    ] = None,
    verify_invalid: Annotated[
        bool,
        typer.Option(
            "--verify-invalid",
            help="Verify positions are valid (within bounding box). Requires Dipy.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force overwriting of output files."),
    ] = False,
) -> None:
    """Generate a TRX file from raw data files.

    Create a TRX file from CSV, TXT, or NPY files by specifying positions,
    offsets, data_per_vertex, data_per_streamlines, groups, and data_per_group.

    Parameters
    ----------
    reference : Path
        Reference anatomy (.nii or .nii.gz).
    out_tractogram : Path
        Output tractogram (.trk, .tck, .vtk, .fib, .dpy, .trx).
    positions : Path or None, optional
        Binary file with streamline coordinates (Nx3 .npy).
    offsets : Path or None, optional
        Binary file with streamline offsets (.npy).
    positions_csv : Path or None, optional
        CSV file with flattened streamline coordinates.
    space : str, optional
        Coordinate space. Non-default requires Dipy.
    origin : str, optional
        Coordinate origin. Non-default requires Dipy.
    positions_dtype : str, optional
        Datatype for positions.
    offsets_dtype : str, optional
        Datatype for offsets.
    dpv : list of str or None, optional
        Data per vertex entries as FILE,DTYPE pairs.
    dps : list of str or None, optional
        Data per streamline entries as FILE,DTYPE pairs.
    groups : list of str or None, optional
        Group entries as FILE,DTYPE pairs.
    dpg : list of str or None, optional
        Data per group entries as GROUP,FILE,DTYPE triplets.
    verify_invalid : bool, optional
        Verify positions are inside bounding box (requires Dipy).
    force : bool, optional
        Overwrite output if it already exists.

    Returns
    -------
    None
        Writes the generated tractogram to disk.
    """
    _check_overwrite(out_tractogram, force)

    # Validate input combinations
    if not positions and not positions_csv:
        typer.echo(
            typer.style(
                "Error: At least one positions option must be provided "
                "(--positions or --positions-csv).",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if positions_csv and positions:
        typer.echo(
            typer.style(
                "Error: Cannot use both --positions and --positions-csv.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if positions and offsets is None:
        typer.echo(
            typer.style(
                "Error: --offsets must be provided if --positions is used.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)
    if offsets and positions is None:
        typer.echo(
            typer.style(
                "Error: --positions must be provided if --offsets is used.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    # Parse comma-separated arguments to tuples
    dpv_list = None
    if dpv:
        dpv_list = [tuple(item.split(",")) for item in dpv]

    dps_list = None
    if dps:
        dps_list = [tuple(item.split(",")) for item in dps]

    groups_list = None
    if groups:
        groups_list = [tuple(item.split(",")) for item in groups]

    dpg_list = None
    if dpg:
        dpg_list = [tuple(item.split(",")) for item in dpg]

    generate_trx_from_scratch(
        str(reference),
        str(out_tractogram),
        positions_csv=str(positions_csv) if positions_csv else None,
        positions=str(positions) if positions else None,
        offsets=str(offsets) if offsets else None,
        positions_dtype=positions_dtype,
        offsets_dtype=offsets_dtype,
        space_str=space,
        origin_str=origin,
        verify_invalid=verify_invalid,
        dpv=dpv_list,
        dps=dps_list,
        groups=groups_list,
        dpg=dpg_list,
    )

    typer.echo(
        typer.style(
            f"Successfully generated {out_tractogram}",
            fg=typer.colors.GREEN,
        )
    )


@app.command("manipulate-dtype")
def manipulate_dtype(
    in_tractogram: Annotated[
        Path,
        typer.Argument(help="Input TRX file."),
    ],
    out_tractogram: Annotated[
        Path,
        typer.Argument(help="Output tractogram file."),
    ],
    positions_dtype: Annotated[
        Optional[str],
        typer.Option(
            "--positions-dtype",
            help="Datatype for positions (float16, float32, float64).",
        ),
    ] = None,
    offsets_dtype: Annotated[
        Optional[str],
        typer.Option(
            "--offsets-dtype",
            help="Datatype for offsets (uint32, uint64).",
        ),
    ] = None,
    dpv: Annotated[
        Optional[List[str]],
        typer.Option(
            "--dpv",
            help="Data per vertex dtype: NAME,DTYPE (e.g., color_x,uint8).",
        ),
    ] = None,
    dps: Annotated[
        Optional[List[str]],
        typer.Option(
            "--dps",
            help="Data per streamline dtype: NAME,DTYPE (e.g., algo,uint8).",
        ),
    ] = None,
    groups: Annotated[
        Optional[List[str]],
        typer.Option(
            "--groups",
            help="Groups dtype: NAME,DTYPE (e.g., CC,uint64).",
        ),
    ] = None,
    dpg: Annotated[
        Optional[List[str]],
        typer.Option(
            "--dpg",
            help="Data per group dtype: GROUP,NAME,DTYPE (e.g., CC,mean_fa,float64).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force overwriting of output files."),
    ] = False,
) -> None:  # noqa: C901
    """Manipulate TRX file internal array data types.

    Change the data types of positions, offsets, data_per_vertex,
    data_per_streamline, groups, and data_per_group arrays.

    Parameters
    ----------
    in_tractogram : Path
        Input TRX file.
    out_tractogram : Path
        Output TRX file.
    positions_dtype : str or None, optional
        Target dtype for positions (float16, float32, float64).
    offsets_dtype : str or None, optional
        Target dtype for offsets (uint32, uint64).
    dpv : list of str or None, optional
        Data per vertex dtype overrides as NAME,DTYPE pairs.
    dps : list of str or None, optional
        Data per streamline dtype overrides as NAME,DTYPE pairs.
    groups : list of str or None, optional
        Group dtype overrides as NAME,DTYPE pairs.
    dpg : list of str or None, optional
        Data per group dtype overrides as GROUP,NAME,DTYPE triplets.
    force : bool, optional
        Overwrite output if it already exists.

    Returns
    -------
    None
        Writes the dtype-converted TRX file.
    """
    _check_overwrite(out_tractogram, force)

    dtype_dict = {}
    if positions_dtype:
        dtype_dict["positions"] = np.dtype(positions_dtype)
    if offsets_dtype:
        dtype_dict["offsets"] = np.dtype(offsets_dtype)
    if dpv:
        dtype_dict["dpv"] = {}
        for item in dpv:
            name, dtype = item.split(",")
            dtype_dict["dpv"][name] = np.dtype(dtype)
    if dps:
        dtype_dict["dps"] = {}
        for item in dps:
            name, dtype = item.split(",")
            dtype_dict["dps"][name] = np.dtype(dtype)
    if groups:
        dtype_dict["groups"] = {}
        for item in groups:
            name, dtype = item.split(",")
            dtype_dict["groups"][name] = np.dtype(dtype)
    if dpg:
        dtype_dict["dpg"] = {}
        for item in dpg:
            parts = item.split(",")
            group, name, dtype = parts[0], parts[1], parts[2]
            if group not in dtype_dict["dpg"]:
                dtype_dict["dpg"][group] = {}
            dtype_dict["dpg"][group][name] = np.dtype(dtype)

    manipulate_trx_datatype(str(in_tractogram), str(out_tractogram), dtype_dict)

    typer.echo(
        typer.style(
            f"Successfully manipulated datatypes and saved to {out_tractogram}",
            fg=typer.colors.GREEN,
        )
    )


@app.command("compare")
def compare(
    in_tractogram1: Annotated[
        Path,
        typer.Argument(help="First tractogram file."),
    ],
    in_tractogram2: Annotated[
        Path,
        typer.Argument(help="Second tractogram file."),
    ],
    reference: Annotated[
        Optional[Path],
        typer.Option(
            "--reference",
            "-r",
            help="Reference anatomy for tck/vtk/fib/dpy files (.nii or .nii.gz).",
        ),
    ] = None,
) -> None:
    """Compare two tractograms and report basic differences.

    Parameters
    ----------
    in_tractogram1 : Path
        First tractogram file.
    in_tractogram2 : Path
        Second tractogram file.
    reference : Path or None, optional
        Reference anatomy for formats requiring it.

    Returns
    -------
    None
        Prints comparison summary to stdout.
    """
    ref = str(reference) if reference else None
    tractogram_simple_compare([str(in_tractogram1), str(in_tractogram2)], ref)


@app.command("validate")
def validate(
    in_tractogram: Annotated[
        Path,
        typer.Argument(help="Input tractogram. Format: trk, tck, vtk, fib, dpy, trx."),
    ],
    out_tractogram: Annotated[
        Optional[Path],
        typer.Option(
            "--out",
            "-o",
            help="Output tractogram after removing invalid streamlines.",
        ),
    ] = None,
    remove_identical: Annotated[
        bool,
        typer.Option(
            "--remove-identical",
            help="Remove identical streamlines from the set.",
        ),
    ] = False,
    precision: Annotated[
        int,
        typer.Option(
            "--precision",
            "-p",
            help="Number of decimals when hashing streamline points.",
        ),
    ] = 1,
    reference: Annotated[
        Optional[Path],
        typer.Option(
            "--reference",
            "-r",
            help="Reference anatomy for tck/vtk/fib/dpy files (.nii or .nii.gz).",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Force overwriting of output files."),
    ] = False,
) -> None:
    """Validate a tractogram and optionally clean invalid/duplicate streamlines.

    Parameters
    ----------
    in_tractogram : Path
        Input tractogram (.trk, .tck, .vtk, .fib, .dpy, .trx).
    out_tractogram : Path or None, optional
        Optional output tractogram with invalid streamlines removed.
    remove_identical : bool, optional
        Remove duplicate streamlines based on hashing precision.
    precision : int, optional
        Number of decimals when hashing streamline points.
    reference : Path or None, optional
        Reference anatomy for formats requiring it.
    force : bool, optional
        Overwrite output if it already exists.

    Returns
    -------
    None
        Prints validation summary and optionally writes cleaned output.
    """
    if out_tractogram:
        _check_overwrite(out_tractogram, force)

    ref = str(reference) if reference else None
    out = str(out_tractogram) if out_tractogram else None

    validate_tractogram(
        str(in_tractogram),
        reference=ref,
        out_tractogram=out,
        remove_identical_streamlines=remove_identical,
        precision=precision,
    )

    if out_tractogram:
        typer.echo(
            typer.style(
                f"Validation complete. Output saved to {out_tractogram}",
                fg=typer.colors.GREEN,
            )
        )
    else:
        typer.echo(
            typer.style(
                "Validation complete.",
                fg=typer.colors.GREEN,
            )
        )


@app.command("verify-header")
def verify_header(
    in_files: Annotated[
        List[Path],
        typer.Argument(help="Files to compare (trk, trx, and nii)."),
    ],
) -> None:
    """Compare spatial attributes of input files.

    Parameters
    ----------
    in_files : list of Path
        Files to compare (.trk, .trx, .nii, .nii.gz).

    Returns
    -------
    None
        Prints compatibility results to stdout.
    """
    verify_header_compatibility([str(f) for f in in_files])


@app.command("visualize")
def visualize(
    in_tractogram: Annotated[
        Path,
        typer.Argument(help="Input tractogram. Format: trk, tck, vtk, fib, dpy, trx."),
    ],
    reference: Annotated[
        Path,
        typer.Argument(help="Reference anatomy (.nii or .nii.gz)."),
    ],
    remove_invalid: Annotated[
        bool,
        typer.Option(
            "--remove-invalid",
            help="Remove invalid streamlines to avoid density_map crash.",
        ),
    ] = False,
) -> None:
    """Display tractogram and density map with bounding box.

    Parameters
    ----------
    in_tractogram : Path
        Input tractogram (.trk, .tck, .vtk, .fib, .dpy, .trx).
    reference : Path
        Reference anatomy (.nii or .nii.gz).
    remove_invalid : bool, optional
        Remove invalid streamlines to avoid density map crashes.

    Returns
    -------
    None
        Opens visualization windows when fury is available.
    """
    tractogram_visualize_overlap(
        str(in_tractogram),
        str(reference),
        remove_invalid,
    )


def _format_size(size_bytes: int) -> str:
    """Format byte size to human readable string.

    Parameters
    ----------
    size_bytes : int
        Size in bytes.

    Returns
    -------
    str
        Human readable size string (e.g., "1.5 MB").
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@app.command("info")
def info(
    in_tractogram: Annotated[
        Path,
        typer.Argument(help="Input TRX file."),
    ],
) -> None:
    """Display detailed information about a TRX file.

    Shows file size, compression status, header metadata (affine, dimensions,
    voxel sizes), streamline/vertex counts, data keys (dpv, dps, dpg), groups,
    and archive contents listing similar to ``unzip -l``.

    Parameters
    ----------
    in_tractogram : Path
        Input TRX file (.trx extension required).

    Returns
    -------
    None
        Prints TRX file information to stdout.

    Examples
    --------
    $ trx info tractogram.trx
    $ trx_info tractogram.trx
    """
    import zipfile

    if not in_tractogram.exists():
        typer.echo(
            typer.style(f"Error: {in_tractogram} does not exist.", fg=typer.colors.RED),
            err=True,
        )
        raise typer.Exit(code=1)

    if in_tractogram.suffix.lower() != ".trx":
        typer.echo(
            typer.style(
                f"Error: {in_tractogram.name} is not a TRX file. "
                "Only .trx files are supported.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    # Show archive info
    file_size = in_tractogram.stat().st_size
    typer.echo(f"File: {in_tractogram}")
    typer.echo(f"Size: {_format_size(file_size)}")

    with zipfile.ZipFile(str(in_tractogram), "r") as zf:
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        is_compressed = any(info.compress_type != 0 for info in zf.infolist())
        typer.echo(f"Entries: {len(zf.infolist())}")
        typer.echo(f"Compressed: {'Yes' if is_compressed else 'No'}")
        typer.echo(f"Uncompressed size: {_format_size(total_uncompressed)}")

    typer.echo("")

    # Show TRX content info
    trx = load_trx(str(in_tractogram))
    typer.echo(trx)

    # Show file listing (unzip -l style)
    typer.echo("\nArchive contents:")
    typer.echo("  Length      Date    Time    Name")
    typer.echo("---------  ---------- -----   ----")
    with zipfile.ZipFile(str(in_tractogram), "r") as zf:
        for zinfo in zf.infolist():
            dt = zinfo.date_time
            date_str = f"{dt[1]:02d}-{dt[2]:02d}-{dt[0]}"
            time_str = f"{dt[3]:02d}:{dt[4]:02d}"
            typer.echo(
                f"{zinfo.file_size:>9}  {date_str} {time_str}   {zinfo.filename}"
            )
        num_files = len(zf.infolist())
        typer.echo("---------                     -------")
        typer.echo(f"{total_uncompressed:>9}                     {num_files} files")

    trx.close()


def main():
    """Entry point for the TRX CLI."""
    app()


# Standalone entry points for backward compatibility
# These create individual Typer apps for each command


def _create_standalone_app(command_func, name: str, help_text: str):
    """Create a standalone Typer app for a single command.

    Parameters
    ----------
    command_func : callable
        The command function to wrap.
    name : str
        Name of the command.
    help_text : str
        Help text for the command.

    Returns
    -------
    callable
        Entry point function.
    """
    standalone = typer.Typer(
        name=name,
        help=help_text,
        add_completion=False,
        rich_markup_mode="rich",
    )
    standalone.command()(command_func)
    return lambda: standalone()


concatenate_tractograms_cmd = _create_standalone_app(
    concatenate_tractograms,
    "trx_concatenate_tractograms",
    "Concatenate multiple tractograms into one.",
)

convert_dsi_cmd = _create_standalone_app(
    convert_dsi,
    "trx_convert_dsi_studio",
    "Fix DSI-Studio TRK files for compatibility.",
)

convert_cmd = _create_standalone_app(
    convert,
    "trx_convert_tractogram",
    "Convert tractograms between formats.",
)

generate_cmd = _create_standalone_app(
    generate,
    "trx_generate_from_scratch",
    "Generate TRX file from raw data files.",
)

manipulate_dtype_cmd = _create_standalone_app(
    manipulate_dtype,
    "trx_manipulate_datatype",
    "Manipulate TRX file internal array data types.",
)

compare_cmd = _create_standalone_app(
    compare,
    "trx_simple_compare",
    "Simple comparison of tractograms by subtracting coordinates.",
)

validate_cmd = _create_standalone_app(
    validate,
    "trx_validate",
    "Validate TRX file and remove invalid streamlines.",
)

verify_header_cmd = _create_standalone_app(
    verify_header,
    "trx_verify_header_compatibility",
    "Compare spatial attributes of input files.",
)

visualize_cmd = _create_standalone_app(
    visualize,
    "trx_visualize_overlap",
    "Display tractogram and density map with bounding box.",
)

info_cmd = _create_standalone_app(
    info,
    "trx_info",
    "Display information about a TRX file.",
)


if __name__ == "__main__":
    main()
