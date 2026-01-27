:html_theme.sidebar_secondary.remove:

TRX File Format Specifications
===============================

This document contains the complete specifications for the TRX (Tractography File Format)
as defined by the TRX specification. TRX is a community-oriented tractography file format
designed to facilitate dataset exchange, interoperability, and state-of-the-art analyses.

General Properties
------------------

**File Structure**

- (Un)-Compressed Zip File or simple folder architecture
- File architecture describes the data
- Each file basename is the metadata's name
- Each file extension is the metadata's dtype
- Each file dimension is in the value between basename and metadata (1-dimension arrays do not have to follow this convention for readability)

**Data Organization**

- All arrays have a C-style memory layout (row-major)
- All arrays have a little-endian byte order
- Compression is optional:

  - Use ``ZIP_STORE`` for uncompressed storage
  - Use ``ZIP_DEFLATE`` if compression is desired
  - Compressed TRX files will have to be decompressed before being loaded

Header
------

The header contains metadata for readability, run-time checks, and broader compatibility.
It is stored as a dictionary in JSON format with the following fields:

**Required Fields:**

.. code-block:: text

    VOXEL_TO_RASMM : 4x4 transformation matrix (list of 4 lists, each containing 4 floats)
    DIMENSIONS     : Image dimensions (list of 3 uint16)
    NB_STREAMLINES : Number of streamlines (uint32)
    NB_VERTICES    : Total number of vertices (uint64)

Arrays
------

positions.{N}.float{16,32,64}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Written in world space (RASMM), similar to TCK files
- Should always be float16/32/64 (default recommended: float16)
- Stored as contiguous 3D array with shape (NB_VERTICES, 3)
- The {N} dimension specifier can be omitted for 1D arrays for readability

offsets.uint{32,64}
~~~~~~~~~~~~~~~~~~~

- Should always be uint32 or uint64
- Indicates the starting vertex index for each streamline (starts at 0)
- Streamline lengths can be calculated by:

  1. Checking the header for total vertices count
  2. Using positions array size: ``positions.shape[0] / 3``
  3. Calculating differences between consecutive elements: append total_vertices to offsets array and compute ediff1d

dpv (data_per_vertex)
~~~~~~~~~~~~~~~~~~~~~

- Always of size (NB_VERTICES, 1) or (NB_VERTICES, N)
- Contains data associated with each vertex/point along streamlines
- Common uses: FA values, colors, curvature, local coordinate systems

dps (data_per_streamline)
~~~~~~~~~~~~~~~~~~~~~~~~~

- Always of size (NB_STREAMLINES, 1) or (NB_STREAMLINES, N)
- Contains data associated with entire streamlines
- Common uses: bundle IDs, mean metrics, algorithm information

Groups
------

Groups are tables of indices that allow sparse & overlapping representation
(clusters, connectomics, bundles).

**Properties:**

- All indices must be ``0 <= id < NB_STREAMLINES``
- Datatype should be uint32
- Allows efficient retrieval of predefined streamline subsets from memmaps
- Variables can have different sizes

dpg (data_per_group)
~~~~~~~~~~~~~~~~~~~~

- Each folder corresponds to name of a group
- Not all metadata have to be present in all groups
- Always of size (1,) or (N,) per group
- Contains group-specific metadata like volumes, mean values, color codes

Supported Data Types
--------------------

The TRX format supports the following data types:

**Integer Types:**

- ``int8``, ``int16``, ``int32``, ``int64``
- ``uint8``, ``uint16``, ``uint32``, ``uint64``

**Floating Point Types:**

- ``float16``, ``float32``, ``float64``

**Boolean Type:**

- ``bit`` (for boolean data)

Example File Structure
----------------------

.. code-block:: text

    OHBM_demo.trx
    |-- dpg
    |   |-- AF_L
    |   |   |-- mean_fa.float16
    |   |   |-- shuffle_colors.3.uint8
    |   |   +-- volume.uint32
    |   |-- AF_R
    |   |   |-- mean_fa.float16
    |   |   |-- shuffle_colors.3.uint8
    |   |   +-- volume.uint32
    |   |-- CC
    |   |   |-- mean_fa.float16
    |   |   |-- shuffle_colors.3.uint8
    |   |   +-- volume.uint32
    |   |-- CST_L
    |   |   +-- shuffle_colors.3.uint8
    |   |-- CST_R
    |   |   +-- shuffle_colors.3.uint8
    |   |-- SLF_L
    |   |   |-- mean_fa.float16
    |   |   |-- shuffle_colors.3.uint8
    |   |   +-- volume.uint32
    |   +-- SLF_R
    |       |-- mean_fa.float16
    |       |-- shuffle_colors.3.uint8
    |       +-- volume.uint32
    |-- dpv
    |   |-- color_x.uint8
    |   |-- color_y.uint8
    |   |-- color_z.uint8
    |   +-- fa.float16
    |-- dps
    |   |-- algo.uint8
    |   |-- algo.json
    |   |-- clusters_QB.uint16
    |   |-- commit_colors.3.uint8
    |   +-- commit_weights.float32
    |-- groups
    |   |-- AF_L.uint32
    |   |-- AF_R.uint32
    |   |-- CC.uint32
    |   |-- CST_L.uint32
    |   |-- CST_R.uint32
    |   |-- SLF_L.uint32
    |   +-- SLF_R.uint32
    |-- header.json
    |-- offsets.uint64
    +-- positions.3.float16

Naming Conventions
------------------

**Files:**

- Basename = metadata name
- Extension = data type
- Dimension specifiers between basename and extension (optional for 1D)

**Examples:**

- ``positions.3.float16`` - 3D position data as float16
- ``fa.float16`` - 1D fractional anisotropy values as float16
- ``colors.3.uint8`` - RGB color values as 8-bit unsigned integers
- ``bundle_id.uint8`` - Bundle identifiers as 8-bit unsigned integers

Memory and Performance Considerations
-------------------------------------

**Memory Efficiency:**

- Use float16 for positional data when precision allows
- Choose appropriate integer sizes for indices (uint32 for streamline indices)
- Consider compression for disk storage but expect decompression overhead

**Performance:**

- C-style memory layout enables efficient numpy operations
- Little-endian byte order ensures consistency across platforms
- Memory-mapped access for large datasets without full loading

**Scalability:**

- Support for arbitrarily large numbers of streamlines and vertices
- Group-based organization enables efficient subset operations
- Flexible metadata structure accommodates various analysis workflows

Compatibility and Integration
-----------------------------

TRX is designed for integration with existing neuroimaging ecosystems:

**Current Support:**

- Native support in trx-python library
- Conversion tools for common tractography formats (TCK, TRK, etc.)
- Integration with DIPY for advanced processing

**Future Goals:**

- Integration with Brain Imaging Data Structure (BIDS) ecosystem
- Support in major neuroimaging software packages
- Standardization across the tractography community

For latest updates and community discussions, see:

- `TRX Specification Repository <https://github.com/tee-ar-ex/trx-spec>`_
- `TRX Python Implementation <https://github.com/tee-ar-ex/trx-python>`_
