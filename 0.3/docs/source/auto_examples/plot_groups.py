# -*- coding: utf-8 -*-
# sphinx_gallery_thumbnail_path = '../docs/_static/trx_logo.png'
"""
Working with Groups
====================

This tutorial demonstrates how to work with groups in TRX files.
Groups allow you to organize streamlines into meaningful subsets,
such as anatomical bundles or clusters.

By the end of this tutorial, you will know how to:

- Access groups in a TRX file
- Extract streamlines belonging to a specific group
- Understand the relationship between groups and data_per_group (dpg)
- Work with overlapping groups
"""

# %%
# What are Groups?
# ----------------
#
# Groups in TRX files are collections of streamline indices. They enable:
#
# - **Sparse representation**: Only store indices instead of copying data
# - **Overlapping membership**: A streamline can belong to multiple groups
# - **Efficient access**: Quickly extract predefined subsets of streamlines
#
# Common use cases include anatomical bundles (e.g., Arcuate Fasciculus,
# Corpus Callosum), clustering results, or connectivity-based groupings.

# %%
# Loading a TRX file with groups
# ------------------------------
#
# Let's load a TRX file that contains group information.

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

# %%
# Accessing groups
# ----------------
#
# Groups are stored as a dictionary where keys are group names and values
# are numpy arrays of streamline indices.

print(f"Available groups: {list(trx.groups.keys())}")

# Check the number of groups
print(f"Number of groups: {len(trx.groups)}")

# %%
# Let's examine the groups in more detail:

for group_name, indices in trx.groups.items():
    print(f"  {group_name}: {len(indices)} streamlines")

# %%
# Extracting a group
# ------------------
#
# You can extract all streamlines belonging to a specific group using
# the ``get_group()`` method.

if len(trx.groups) > 0:
    # Get the first group name
    first_group = list(trx.groups.keys())[0]

    # Extract the group as a new TrxFile
    group_trx = trx.get_group(first_group)
    print(f"Extracted group '{first_group}' with {len(group_trx)} streamlines")

    # You can also access the raw indices
    group_indices = trx.groups[first_group]
    print(f"Raw indices (first 10): {group_indices[:10]}")
else:
    print("No groups available in this file")

# %%
# Using group indices directly
# ----------------------------
#
# You can use group indices to select streamlines directly with the
# ``select()`` method.

if len(trx.groups) > 0:
    first_group = list(trx.groups.keys())[0]
    indices = trx.groups[first_group]

    # Select streamlines using indices
    selected = trx.select(indices[:5])  # Select first 5 from the group
    print(f"Selected {len(selected)} streamlines from group '{first_group}'")

# %%
# Data per group (dpg)
# --------------------
#
# Groups can have associated metadata stored in ``data_per_group`` (dpg).
# This is useful for storing group-level statistics like mean FA, volume,
# or color codes.

print(f"Data per group keys: {list(trx.data_per_group.keys())}")

# Check what metadata is available for each group
for group_name in trx.data_per_group:
    dpg_keys = list(trx.data_per_group[group_name].keys())
    print(f"  {group_name}: {dpg_keys}")

# %%
# Creating groups manually
# ------------------------
#
# You can create groups by assigning indices to the groups dictionary.
# Here's an example of how groups work conceptually.

# Example: Create conceptual groups for 10 streamlines
example_groups = {
    'bundle_A': np.array([0, 1, 2, 3], dtype=np.uint32),
    'bundle_B': np.array([4, 5, 6, 7, 8, 9], dtype=np.uint32),
    'overlapping': np.array([3, 4, 5], dtype=np.uint32),  # Overlaps with A and B
}

print("Example groups:")
for name, indices in example_groups.items():
    print(f"  {name}: streamlines {indices}")

# Note: Streamline 3 is in both bundle_A and overlapping
# Note: Streamlines 4, 5 are in both bundle_B and overlapping
print("\nOverlapping groups are allowed in TRX!")

# %%
# Group file structure
# --------------------
#
# In the TRX file format, groups are stored as binary files in a ``groups/``
# directory:
#
# .. code-block:: text
#
#     my_tractogram.trx/
#     |-- groups/
#     |   |-- AF_L.uint32      # Arcuate Fasciculus Left
#     |   |-- AF_R.uint32      # Arcuate Fasciculus Right
#     |   |-- CC.uint32        # Corpus Callosum
#     |   +-- CST_L.uint32     # Corticospinal Tract Left
#     +-- ...
#
# Each file contains a flat array of streamline indices as uint32 values.

# %%
# Filtering streamlines by group
# ------------------------------
#
# A common workflow is to filter streamlines based on group membership
# and then analyze or visualize specific bundles.

if len(trx.groups) > 0:
    # Get all group names
    group_names = list(trx.groups.keys())

    # Report statistics for each group
    print("Group statistics:")
    for group_name in group_names:
        group_trx = trx.get_group(group_name)
        total_points = len(group_trx.streamlines._data)
        avg_length = total_points / len(group_trx) if len(group_trx) > 0 else 0
        print(f"  {group_name}:")
        print(f"    - Streamlines: {len(group_trx)}")
        print(f"    - Total points: {total_points}")
        print(f"    - Avg points per streamline: {avg_length:.1f}")

# %%
# Summary
# -------
#
# In this tutorial, you learned how to:
#
# - Access groups using ``trx.groups``
# - Extract group streamlines using ``get_group()``
# - Work with ``data_per_group`` (dpg) metadata
# - Understand that groups can overlap
# - Filter and analyze streamlines by group membership
#
# Groups are a powerful feature of the TRX format that enable efficient
# organization and retrieval of streamline subsets without data duplication.
