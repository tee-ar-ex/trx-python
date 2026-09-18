# -*- coding: utf-8 -*-
# sphinx_gallery_thumbnail_path = '../docs/_static/trx_logo.png'
"""
Reading and Writing TRX Files
==============================

This tutorial demonstrates how to read and write TRX files using trx-python.
TRX is a tractography file format designed for efficient storage and access
of brain fiber tract streamline data.

By the end of this tutorial, you will know how to:

- Load a TRX file from disk
- Inspect the contents of a TRX file
- Access streamlines and metadata
- Save a TRX file to disk
- Create a TRX file from scratch
"""

# %%
# Loading a TRX file
# ------------------
#
# Let's start by loading an existing TRX file. First, we need to download
# some test data.

import os
import tempfile

from trx.fetcher import fetch_data, get_home, get_testing_files_dict
from trx.trx_file_memmap import load, save

# Download test data
fetch_data(get_testing_files_dict(), keys="gold_standard.zip")
trx_home = get_home()
trx_path = os.path.join(trx_home, "gold_standard", "gs.trx")

# Load the TRX file
trx = load(trx_path)

print("TRX file loaded successfully!")

# %%
# Inspecting TRX file contents
# ----------------------------
#
# The TrxFile object has several key attributes that you can inspect.
# Let's look at what's inside our loaded file.

# Print a summary of the TRX file
print(trx)

# %%
# The header contains essential metadata about the tractogram:

print("Header information:")
print(f"  Number of streamlines: {trx.header['NB_STREAMLINES']}")
print(f"  Number of vertices: {trx.header['NB_VERTICES']}")
print(f"  Image dimensions: {trx.header['DIMENSIONS']}")
print(f"  Voxel to RASMM affine:\n{trx.header['VOXEL_TO_RASMM']}")

# %%
# Accessing streamlines
# ---------------------
#
# Streamlines are the core data in a TRX file. Each streamline is a sequence
# of 3D points representing a fiber tract in the brain.

print(f"Number of streamlines: {len(trx)}")
print(f"Total number of vertices: {len(trx.streamlines._data)}")

# Access the first streamline
first_streamline = trx.streamlines[0]
print(f"\nFirst streamline has {len(first_streamline)} points")
print(f"First 3 points of the first streamline:\n{first_streamline[:3]}")

# %%
# Accessing metadata
# ------------------
#
# TRX files can contain additional data per vertex (dpv) and per streamline (dps).

print("Data per vertex (dpv) keys:", list(trx.data_per_vertex.keys()))
print("Data per streamline (dps) keys:", list(trx.data_per_streamline.keys()))
print("Groups:", list(trx.groups.keys()))

# %%
# Selecting a subset of streamlines
# ---------------------------------
#
# You can easily select a subset of streamlines using indices or slicing.

# Select first 5 streamlines
subset = trx[:5]
print(f"Subset has {len(subset)} streamlines")

# Select specific streamlines by indices (ensure indices are valid)
max_idx = len(trx) - 1
indices = [0, min(2, max_idx), min(5, max_idx)]
selected = trx.select(indices)
print(f"Selected {len(selected)} streamlines")

# %%
# Saving a TRX file
# -----------------
#
# You can save a TRX file back to disk. The file can be saved as a compressed
# or uncompressed zip archive, or as a directory.

with tempfile.TemporaryDirectory() as tmpdir:
    # Save as TRX file (zip archive)
    output_path = os.path.join(tmpdir, "output.trx")
    save(trx, output_path)
    print(f"Saved TRX file to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")

    # Reload to verify
    reloaded = load(output_path)
    print(f"Reloaded TRX has {len(reloaded)} streamlines")

# %%
# Creating a TRX file from an existing one
# ----------------------------------------
#
# A common workflow is to create a new TRX file based on an existing one,
# preserving the spatial reference information.

# Create a deepcopy of the loaded TRX file
trx_copy = trx.deepcopy()

print(f"Created copy with {len(trx_copy)} streamlines")
print(f"Header preserved: DIMENSIONS = {trx_copy.header['DIMENSIONS']}")

# %%
# Summary
# -------
#
# In this tutorial, you learned how to:
#
# - Load TRX files using ``load()``
# - Inspect header information and streamline data
# - Access data per vertex (dpv) and data per streamline (dps)
# - Select subsets of streamlines
# - Save TRX files using ``save()``
# - Create copies of TRX files using ``deepcopy()``
#
# The TRX format is designed for memory efficiency through memory-mapping,
# making it suitable for large tractography datasets.
