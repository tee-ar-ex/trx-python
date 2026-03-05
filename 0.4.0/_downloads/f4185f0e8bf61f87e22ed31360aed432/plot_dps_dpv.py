# -*- coding: utf-8 -*-
# sphinx_gallery_thumbnail_path = '../docs/_static/trx_logo.png'
"""
Data Per Vertex and Data Per Streamline
========================================

This tutorial demonstrates how to work with metadata in TRX files.
TRX supports two types of metadata:

- **Data Per Vertex (dpv)**: Information attached to each point along streamlines
- **Data Per Streamline (dps)**: Information attached to entire streamlines

By the end of this tutorial, you will know how to:

- Access dpv and dps data in a TRX file
- Understand the data shapes and organization
- Use metadata for filtering and analysis
"""

# %%
# Understanding DPV and DPS
# -------------------------
#
# **Data Per Vertex (dpv):**
#
# - Attached to each individual point (vertex) in all streamlines
# - Shape: (NB_VERTICES, 1) for scalar data or (NB_VERTICES, N) for vector data
# - Common uses: FA values at each point, RGB colors, local orientations
#
# **Data Per Streamline (dps):**
#
# - Attached to entire streamlines (one value per streamline)
# - Shape: (NB_STREAMLINES, 1) for scalar data or (NB_STREAMLINES, N) for vector data
# - Common uses: bundle ID, mean FA, streamline length, tracking algorithm ID

# %%
# Loading a TRX file with metadata
# --------------------------------
#
# Let's load a TRX file and explore its metadata.

import os

import numpy as np

from trx.fetcher import fetch_data, get_home, get_testing_files_dict
from trx.trx_file_memmap import load

# Download test data
fetch_data(get_testing_files_dict(), keys="gold_standard.zip")
trx_home = get_home()
trx_path = os.path.join(trx_home, "gold_standard", "gs.trx")

# Load the TRX file
trx = load(trx_path)

print(f"Loaded TRX with {len(trx)} streamlines")
print(f"Total vertices: {trx.header['NB_VERTICES']}")

# %%
# Exploring Data Per Vertex (dpv)
# -------------------------------
#
# Let's see what dpv data is available.

print("Data Per Vertex keys:", list(trx.data_per_vertex.keys()))

# Examine each dpv field
for key in trx.data_per_vertex:
    data = trx.data_per_vertex[key]
    print(f"\n  {key}:")
    print(f"    Shape: {data._data.shape}")
    print(f"    Dtype: {data._data.dtype}")
    print(f"    Sample values: {data._data[:3].flatten()}")

# %%
# Accessing dpv for a specific streamline
# ---------------------------------------
#
# The dpv data is organized to match the streamlines. You can access
# the dpv values for a specific streamline using the same indices.

if len(trx.data_per_vertex) > 0:
    first_dpv_key = list(trx.data_per_vertex.keys())[0]
    dpv_data = trx.data_per_vertex[first_dpv_key]

    # Get dpv values for the first streamline
    first_streamline_dpv = dpv_data[0]
    print(f"DPV '{first_dpv_key}' for first streamline:")
    print(f"  Shape: {first_streamline_dpv.shape}")
    print(f"  Values: {first_streamline_dpv.flatten()}")

# %%
# Exploring Data Per Streamline (dps)
# -----------------------------------
#
# Now let's examine the dps data.

print("Data Per Streamline keys:", list(trx.data_per_streamline.keys()))

# Examine each dps field
for key in trx.data_per_streamline:
    data = trx.data_per_streamline[key]
    print(f"\n  {key}:")
    print(f"    Shape: {data.shape}")
    print(f"    Dtype: {data.dtype}")
    print(f"    First 5 values: {data[:5].flatten()}")

# %%
# DPS for filtering streamlines
# -----------------------------
#
# A common use case is filtering streamlines based on dps values.
# For example, selecting streamlines with high FA values.

if len(trx.data_per_streamline) > 0:
    # Use the first dps key for demonstration
    first_dps_key = list(trx.data_per_streamline.keys())[0]
    dps_data = trx.data_per_streamline[first_dps_key]

    # Calculate some statistics
    print(f"\nStatistics for '{first_dps_key}':")
    print(f"  Min: {np.min(dps_data):.4f}")
    print(f"  Max: {np.max(dps_data):.4f}")
    print(f"  Mean: {np.mean(dps_data):.4f}")
    print(f"  Std: {np.std(dps_data):.4f}")

# %%
# File structure for dpv and dps
# ------------------------------
#
# In the TRX format, dpv and dps are stored in separate directories:
#
# .. code-block:: text
#
#     my_tractogram.trx/
#     |-- dpv/
#     |   |-- fa.float16              # FA values per vertex
#     |   |-- colors.3.uint8          # RGB colors (3 values per vertex)
#     |   +-- curvature.float32       # Curvature per vertex
#     |-- dps/
#     |   |-- bundle_id.uint8         # Bundle assignment per streamline
#     |   |-- length.uint16           # Length per streamline
#     |   +-- mean_fa.float32         # Mean FA per streamline
#     +-- ...
#
# The filename format is: ``name.dtype`` or ``name.dimension.dtype``

# %%
# Working with multi-dimensional data
# -----------------------------------
#
# Both dpv and dps can have multiple dimensions. For example, RGB colors
# have 3 values per vertex.

print("\nDemonstrating multi-dimensional data:")

# Check for any multi-dimensional dpv
for key in trx.data_per_vertex:
    data = trx.data_per_vertex[key]
    if len(data._data.shape) > 1 and data._data.shape[1] > 1:
        print(f"  {key}: {data._data.shape[1]}D data per vertex")

# Check for any multi-dimensional dps
for key in trx.data_per_streamline:
    data = trx.data_per_streamline[key]
    if len(data.shape) > 1 and data.shape[1] > 1:
        print(f"  {key}: {data.shape[1]}D data per streamline")

# %%
# Relationship between dpv and streamlines
# ----------------------------------------
#
# It's important to understand how dpv data maps to individual streamlines.
# Each streamline's dpv values can be accessed using the streamline's
# vertex indices.

# Get vertex counts for first few streamlines
print("\nVertex distribution for first 5 streamlines:")
for i in range(min(5, len(trx))):
    streamline = trx.streamlines[i]
    print(f"  Streamline {i}: {len(streamline)} vertices")

# Total vertices should match
total_from_streamlines = sum(len(trx.streamlines[i]) for i in range(len(trx)))
print(f"\nTotal vertices from streamlines: {total_from_streamlines}")
print(f"Total vertices in header: {trx.header['NB_VERTICES']}")

# %%
# Summary
# -------
#
# In this tutorial, you learned how to:
#
# - Access dpv data using ``trx.data_per_vertex[key]``
# - Access dps data using ``trx.data_per_streamline[key]``
# - Understand the shape conventions for scalar and vector data
# - Use metadata for statistical analysis
# - Understand the file structure for dpv and dps
#
# The TRX format's metadata system is designed for flexibility, allowing
# you to attach any kind of information to vertices or streamlines.
