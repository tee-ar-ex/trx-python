# -*- coding: utf-8 -*-
"""
TRX Command Line Interface.

This module provides a unified CLI for all TRX file format operations using Typer.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer
from typing_extensions import Annotated

from trx.io import load, save
from trx.trx_file_memmap import TrxFile, concatenate
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

app = typer.Typer(
    name="trx",
    help="TRX File Format Tools - CLI for brain tractography data manipulation.",
    add_completion=False,
    rich_markup_mode="rich",
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
    """Fix DSI-Studio TRK files for compatibility.

    This script fixes DSI-Studio TRK files (unknown space/convention) to make
    them compatible with TrackVis, MI-Brain, and Dipy Horizon.

    [bold yellow]WARNING:[/bold yellow] This script is experimental. DSI-Studio evolves
    quickly and results may vary depending on the data and DSI-Studio version.
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
    """Generate TRX file from raw data files.

    Create a TRX file from CSV, TXT, or NPY files by specifying positions,
    offsets, data_per_vertex, data_per_streamlines, groups, and data_per_group.

    Each --dpv, --dps, --groups option requires FILE,DTYPE format.
    Each --dpg option requires GROUP,FILE,DTYPE format.

    Valid DTYPEs: (u)int8, (u)int16, (u)int32, (u)int64, float16, float32, float64, bool
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

    Valid DTYPEs: (u)int8, (u)int16, (u)int32, (u)int64, float16, float32, float64, bool
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
    """Simple comparison of tractograms by subtracting coordinates.

    Does not account for shuffling of streamlines. Simple A-B operations.

    Differences below 1e-3 are expected for affines with large rotation/scaling.
    Differences below 1e-6 are expected for isotropic data with small rotation.
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
    """Validate TRX file and remove invalid streamlines.

    Removes streamlines that are out of the volume bounding box (in voxel space,
    no negative coordinates or coordinates above volume dimensions).

    Also removes streamlines with single or no points.
    Use --remove-identical to remove duplicate streamlines based on precision.
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

    Compares all input files against the first one for compatibility of
    spatial attributes: affine, dimensions, voxel sizes, and voxel order.
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

    Shows the tractogram and its density map (computed from Dipy) in
    rasmm, voxmm, and vox space with its bounding box.
    """
    tractogram_visualize_overlap(
        str(in_tractogram),
        str(reference),
        remove_invalid,
    )


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


if __name__ == "__main__":
    main()
