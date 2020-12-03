# 1. Introduction

The goal is to describe a more useful and long term format for
tractography data. Each of the existing formats has limitations, leading
to different tools being biased toward one tool or another.

# 2. Limitations with current formats

Below the current limitations with the current tractography file formats
(please contribute given your experience):

-   Lack of community based and agreed development plan

-   Large file size that does not allow easy storage of tractograms over
    > 5-10 millions streamlines on standard desktop computers.

-   Lack of compression for disk storage.

-   Lack of partial loading of streamlines, streamlines groups.

-   Non standard spatial transformation. For example, the popular TRK
    > format does provide support for per-vertex and per-streamline
    > values, it uses a complex and non-standard spatial mapping.

-   lack of xyz \< more here please \>?

-

# 3. Wanted features for new file format

The desired features (though note that any format will require a
trade-off between these different metrics):

-   **Community-based development.** Community development instead of
    > toolbox dependence.

-   **High-level of cross development platform compatibility.** Format
    > should be compatible with Python, C++, JavaScript, Julia, and
    > MatLab coding environments.

-   **Compression with multi-thread support.** The new format should
    > allow support for data compression, just like NIfTI \"nii.gz"

-   **Simplicity.** Essential if it\'s to be accepted as a standard. It
    > should be relatively easy to code up import/export routines from
    > any language, without relying on external tooling. As also
    > mentioned by \@frheault, there\'s lots of ways of storing these
    > data wrong, no matter the format, so it\'s important to minimise
    > any unnecessary complexities in the format, and be explicit about
    > the conventions used. For example, the tck format used in MRtrix
    > stores vertices in world coordinates and has only 2 required
    > elements in its header: a datatype specifier (almost invariably
    > Float32LE), and an offset to the start of the binary data. I
    > can\'t think of anything else that would be classed as necessary
    > here (though of course in practice there\'s lots of additional
    > useful information that we want to store in the header).

-   **Support for seamless spatial transformation and stateful
    > representation. Intuitive metadata handling at any scale
    > (vertices, streamlines, bundles).**

-   **Space and load efficiency:** these are likely to contain very
    > large amounts of data, and these should take no more space than is
    > strictly necessary to store the information. The type of geometry
    > and/or data layout is known in advance, so I don\'t think it makes
    > sense to try to use more generic container formats like VTK or
    > HDF5 - these will likely require more space to describe the
    > geometry / layout. Text formats are also likely to be inefficient
    > from that point of view.

    -   The early mentions of more advanced data formats were based on
        > the "hierarchical" features I (Francois) read in the thread.
        > However, I came to see that simple grouping tables was enough
        > to achieve what was desired. So I agree that such generic
        > containers would be "overkill" for what we need. Let's forget
        > about ASDF/HDF5 (and of course, TXT).

-   **Load/store efficiency:** loading large amounts of data will take
    > even longer if the data need to be converted, especially to/from
    > text. Ideally it should be possible to read() / write() the data
    > into/out of memory in one go, and even better, memory-map it and
    > access it directly. This implies storing in IEEE floating-point
    > format, most likely little-endian since that\'s the native format
    > on all CPUs in common use. We could discuss whether to store in
    > single or double precision, but I don\'t expect there will be many
    > applications where we need to store vertex locations with 64 bits
    > of precision - in fact, I wouldn\'t be surprised if the discussion
    > goes the other way, with the possibility of using 16 or 24 bit
    > floats instead (though these would require conversion and could
    > potentially slow down the IO). Note that some formats can yield
    > better store performance during streamline creation, while others
    > can provide better read performance.

-   **Independence:** I think it\'s critical that the format is
    > standalone, and independent of any external reference. Having to
    > supply the coordinate system for the tractogram by way of a
    > user-supplied image would in my opinion massively expand the scope
    > for mistakes. I don\'t mind so much if the necessary information
    > is encoded in the header, as suggested by \@frheault- but I don\'t
    > see that it adds a great deal to simply storing the data in world
    > coordinates directly. I do appreciate that it probably matches the
    > way data are processed in many packages, where everything is
    > performed in voxel space. In MRtrix, everything is performed in
    > real space, and the fODFs / eigenvectors are stored relative to
    > world coordinates also, so there\'s no further conversion
    > necessary - I appreciate not all packages work that way. And for
    > full disclosure: we (MRtrix) would have a vested interest here
    > since storing in anything other than world coordinates would
    > probably mean more work for our applications.

-   **Extensibility**: we routinely add lots more information in our
    > tractogram headers as the need arises, and I expect there will be
    > many applications where the ability to store additional
    > information more loosely will be useful. A standard format should
    > allow for this, and also allow for additional entries in the
    > header to become part of the official standard if & when their use
    > becomes commonplace.

-   **On-the-fly writing:** This could be a nice feature IF the user
    > follows basic restrictions. It would be very hard to allow
    > efficient writing of point data and at the same time writing
    > metadata. When designing the file format in-depth it would be nice
    > to see if we could allow for that feature.

    -   However, it is possible that this 'feature' would actually be
        > left to anyone coding their own writer/reader. It could be as
        > simple as declaring an upper bound size and writing in-place
        > in a memmap and when the writing is finished a resizing
        > operation is performed and the header is updated.

-   Metadata handling to store properties of

    -   The entire tractogram

    -   Per-vertex scalar values, e.g. curvature along a track

    -   Per-streamline properties, e.g. sets of streamlines belong to
        > different bundles

-   Allows for grouping streamlines and per-group-properties

**Note.** Some developers prefer extending an existing format, to aid
adoption and conversion. On the other hand, there is a strong sentiment
that a new format should not be constrained by existing formats.

#

# 4. Streamline Storage

Any format requires a compromise of desired features. Here, competing
methods for storing streamlines are described. A basic issue is a single
tractography file typically stores many streamlines, some of which have
fewer vertices than others. This situation is often referred to as
\"jagged arrays\" or \"ragged arrays\".

1.  The [[VTK formats (legacy and
    > XML-based)]{.ul}](https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf)
    > describe streamlines by explicitly providing an index for each and
    > every vertex of a streamline. For example, with the legacy VTK
    > format there is a floating-point list of all the vertices (e.g.
    > V~0~V~1~1\...V~n~) and a separate integer array explicitly
    > associating each vertex with a line (numPoints0, i0, j0, k0, \...
    > numPoints1, i1, j1, k1, ..). This allows vertices to be re-used,
    > which is efficient when vertices are
    > [[shared]{.ul}](http://hacksoflife.blogspot.com/2010/01/to-strip-or-not-to-strip.html),
    > which is common for triangulated meshes. For the datasets
    > evaluated here, where each vertex is only used once, and where
    > points in a streamline are stored sequentially this approach leads
    > to much larger files than the methods listed below (as
    > demonstrated by the examples provided
    > [[here]{.ul}](https://github.com/rordenlab/TractographyFormat/tree/master/DATA)).

2.  Most current formats
    > ([[BFLOAT]{.ul}](http://camino.cs.ucl.ac.uk/index.php?n=Main.Fileformats),
    > [[DAT]{.ul}](https://www.mristudio.org/wiki/faq),
    > [[TCK]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#tracks-file-format-tck),
    > [[TRK]{.ul}](http://trackvis.org/docs/?subsect=fileformat),
    > [[PDB]{.ul}](http://graphics.stanford.edu/projects/dti/software/pdb_format.html),
    > NIML.TRACT) create a single array that includes both the position
    > of the vertices and streamline end/length information. OpenGL's
    > primitive restart index is an example of this method. This has
    > clear benefits for writing data during streamline creation, where
    > the final number of streamlines is not known.

3.  A more efficient layout in terms of reading and random access is to
    > store the vertex position data as a separate array from line
    > length information (e.g. [[netCDF
    > VLEN]{.ul}](https://www.unidata.ucar.edu/software/netcdf/docs/data_type.html)
    > or [[TRAKO]{.ul}](https://github.com/bostongfx/TRAKO)). These two
    > different datatypes (integer based line length/offset versus
    > float-based vertex position) can each be accessed with a single
    > block read. This way of storing the data is much more friendly to
    > the memmap, having a huge memmap for all data and one for each
    > metadata would be nice would be an ideal compromise for
    > interpretation and read/write/random access. This approach was
    > [[evaluated]{.ul}](https://github.com/rordenlab/TractographyFormat),
    > but there is not clear evidence for meaningful improvement in load
    > speed (though larger datasets and different languages may impact
    > this). This seems to support extending existing, proven formats
    > that are already supported. On the other hand, the benefit with
    > regards to random access is a unique feature, which may support
    > considering a new data format.

    -   The storage of the 3D vertex positions could be stored as an
        > [[Array of Structures (AoS) or as Structure of Arrays
        > (SoA)]{.ul}](https://en.wikipedia.org/wiki/AoS_and_SoA). The
        > 3D vertex positions would intuitively be saved as
        > x~0~,y~0~,z~0~,\...x~n-1~,y~n-1~,z~0-1~. Alternatively, the
        > data could be saved
        > x~0~\...x~n-1~,y~0~,\...y~n-1~,z~0~,\...z~n-1~. The former is
        > more intuitive, and matches how vertices are sent to the
        > graphics card (though note vertex position will be interleaved
        > with other vertex properties). On the other hand, the latter
        > approach can benefit CPU-based SIMD programming.

# 4. Proposed new formats

The subsequent sections allow members to concretely describe potential
formats to address this issue. Authors should expect candid discussion
of the strengths and limitations inherent to any alternative.

# 4.1 Extending TCK

Given that the TCK format is extensible, and fulfills many of the
desired properties, this section explores extending this format to add
support to include new features. This proposal attempts to leverage the
best features of two existing formats.

-   [[TRK]{.ul}](http://trackvis.org/docs/?subsect=fileformat) is
    > simple, fast, and describes up to 10 per-vertex and per-streamline
    > values. However, it is not extensible and it uses a strange method
    > for describing spatial position (using voxel corners rather than
    > world space like most streamline formats, or voxel centers like
    > popular voxel formats, see nibabel trk.py for details).

-   [[TCK]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#tracks-file-format-tck)
    > is simple, fast, extensible and describes vertex position simply,
    > but does not currently support per-vertex and per-streamline
    > values (relying on additional TSF files for this feature).

Lets begin with a brief description of this format, which will help
elucidate its inherent features. The format begins with a simple text
format header, that continues until the word END.

The
[[TCK]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#tracks-file-format-tck)
format format for a simple file might look like this:

> mrtrix tracks
>
> mrtrix_version: 3.0_RC3_latest-73-g8252c3b6
>
> timestamp: 1596710885.4959571362
>
> datatype: Float32LE
>
> file: . 180
>
> count: 305
>
> total_count: 305
>
> END

The
[[TSF]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#track-scalar-file-format-tsf)
format for a simple file (created by
[[tcksample]{.ul}](https://mrtrix.readthedocs.io/en/latest/reference/commands/tcksample.html))
might look like this:

> mrtrix track scalars
>
> mrtrix_version: 3.0_RC3_latest-73-g8252c3b6
>
> timestamp: 1596710885.4959571362
>
> datatype: Float32LE
>
> file: . 188
>
> count: 305
>
> total_count: 305
>
> END

The links to each format provide more details on the implementation.
Here are a couple points that must be considered if we wish to extend
this format as described earlier:

-   The TCK/TSF formats provide offsets to the start of their payload,
    > but do not report the overall size of their payload. For example,
    > the TSF file lists a count of 305 streamlines, but does not report
    > the total number of vertices encoded. In this example, one starts
    > reading a 32-bit float after seeking 188 bytes into the file, and
    > continues until a NaN is encountered. Likewise, for the TCK file,
    > we know that 305 streamlines are encoded, but not the number of
    > vertices nor the payload size, one reads triplets of 32-bit floats
    > until a NaN is encountered. For the current TCK and TSF files that
    > only encode a single array, one can use filesize to determine the
    > number of elements. For example, assuming the TSF filesize is
    > 178404 bytes, with a 188 byte header, one can determine there are
    > 44554 32-bit floats, with 304 Inf values (end of streamline) and
    > one NaN value (end of file), so one can pre-determine (and
    > allocate memory for) the 44249 vertices. However, note that in
    > practice some tools save a trailing NaN before the final Inf and
    > others do not (e.g. a stream may be saved as either 304 Inf values
    > or 305 Inf values).

-   The **file:** tag can denote that the raw data is stored in a
    > separate
    > [[detached]{.ul}](http://teem.sourceforge.net/nrrd/format.html#detached)
    > file or in these examples (where the file is .) the binary data is
    > attached, i.e. embedded in a single file. The use of a detached
    > file allows a user to create a header with a simple text editor.
    > When the embedded format is used, the contents should not be
    > edited with a text editor: the binary file offset (e.g. 188 for
    > the example TSF file) would change and text-editors would corrupt
    > the binary data. This is different from XML formats like
    > [[GIfTI]{.ul}](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjk1Mng4ovrAhVQS6wKHdKfCRIQFjAAegQIBBAB&url=https%3A%2F%2Fwww.nitrc.org%2Ffrs%2Fdownload.php%2F2871%2FGIFTI_Surface_Format.pdf&usg=AOvVaw00jwRf7g2K6lifPUzjVhc4)
    > where binary data is converted to
    > [[Base64]{.ul}](https://en.wikipedia.org/wiki/Base64), allowing
    > the file to be edited with a text editor. This cost is a trade-off
    > with the improved simplicity, load/store and disk space efficiency
    > of native binary data formats.

## Limitations of TCK

The current implementation of TCK has weaknesses. Some might be fixed by
an extension and improved implementation. Others are inherent and lend
support for selecting a different format.

-   As noted in Section 3, TCK (and other popular formats) is optimized
    > for file writing during streamline creation, and may not be
    > optimal for file reading, and memory-mapped random access.

-   As noted above, the TCK and TSF formats do not explicitly report the
    > number of vertices. While this can be roughly guessed, the
    > presence or absence of a trailing Inf in different implementations
    > exist.

-   As noted above, TCK embedded files combine a text header with a
    > binary stream. While binary formats offer excellent load/store and
    > size efficiency, they will be corrupted if they are opened and
    > then saved by a text editor or transferred by ftp using the
    > default txt mode. One clever feature of the
    > [[PNG]{.ul}](http://www.libpng.org/pub/png/spec/1.2/PNG-Rationale.html)
    > format is that the first few bytes of the file are designed to
    > detect this type of corruption, providing the user with clear
    > feedback if file corruption occurs. Like most binary formats, TCK
    > files do not have a method to detect this common form of
    > corruption.

-   While the text header is convenient, there is a (very small)
    > performance penalty as lines must be parsed character by
    > character. In contrast, fixed length headers like NIfTI and TRK
    > can be read into a structure in a single block read. This
    > criticism is typical of extensible formats, and the proportional
    > influence of this penalty is extremely small for larger files.

-   This format does not explicitly support file compression. One could
    > always compress the whole file (.tck.gz) in the same manner as
    > NIfTI (.nii.gz). However, this is not explicitly defined and may
    > not be supported by other tools. The format could also be extended
    > to support compression of entire detached binary files (like
    > [[NRRD]{.ul}](http://teem.sourceforge.net/nrrd/format.html)).
    > Another trait of NRRD that could be cloned is compressing only the
    > binary portion of attached files (leaving the text header
    > uncompressed). This allows tools to inspect the header without
    > extracting the whole file, but preclude the use of
    > high-performance whole-file compression tools like
    > [[pigz]{.ul}](https://github.com/ebiggers/libdeflate) and
    > [[libdeflate]{.ul}](https://github.com/ebiggers/libdeflate). Given
    > the popularity in our field (e.g. .nii.gz), simplicity and
    > ubiquity, full file .gz seems like a good option. Another option
    > would be the modern zstd which offers [[superior
    > performance]{.ul}](https://github.com/neurolabusc/zlib-bench-python)
    > but would require all tools to rely on an external library
    > (whereas many languages have native zlib support for .gz).

-   As noted, the TCK format describes attached (binary data in the same
    > file as text header) or detached (binary data in different file
    > from header) formats. Formats that depend on multiple files have
    > some
    > [[limitations]{.ul}](https://github.com/bids-standard/bids-2-devel/issues/25),
    > they can get separated due to files moving or not being renamed.
    > Further, operating systems that [[sandbox
    > applications]{.ul}](https://developer.apple.com/library/archive/documentation/Security/Conceptual/AppSandboxDesignGuide/AppSandboxInDepth/AppSandboxInDepth.html)
    > may prevent access to one file when the user interactively selects
    > the other file via a file-open dialog or drag and drop. This is a
    > minor concern, as this feature is optional for TCK and so many
    > formats in our field use detached files (e.g. BIDS sidecars,
    > .head/.brik, .nhdr/.raw, .hdr/.img, etc). Therefore, as these
    > operating systems more strictly enforce entitlements, they may be
    > increasing ill-suited to our domain of research.

## Outline for an extension of TCK

Given that the TCK format is extensible, and fulfills many of the
desired properties, here we suggest extending this format to add support
to include new features. One challenge with complete backwards
compatibility is the re-use of tags like "count" - if used with legacy
tools, some may look at the first instance, and other tools may give
precedence to the final instance. For complete backward compatibility,
these could be incremented or pre-fixed. This is certainly something
that should be discussed. For simplicity, the example below assumes a
new format that would get a new file extension and therefore would
require developers to make some changes to support.

Here is a simple proposed header for an extended TCK format:

> tcx format
>
> creator_version: 3.0_RC3_latest-73-g8252c3b6
>
> timestamp: 1596710885.4959571362
>
> kind: tracks
>
> name: myTracks
>
> datatype: Float32LE
>
> file: . 512
>
> count: 305
>
> total_count: 305
>
> bytes: 534660
>
> kind: scalars
>
> name: FractionalAnisotropy
>
> datatype: Float32LE
>
> file: . 535172
>
> count: 305
>
> total_count: 305
>
> bytes: 178216
>
> kind: property
>
> name: myBundles
>
> datatype: UInt16LE
>
> file: . 713388
>
> count: 305
>
> bytes: 710
>
> kind: property_name
>
> name: myBundles
>
> datatype: UTF8null
>
> file: . 714098
>
> count: 12
>
> bytes: 124
>
> kind: source
>
> matrix: \[2 0 0 0; 0 2 0 0; 0 0 2 0; 0 0 0 1\]
>
> directions: 30
>
> Samples0: 5
>
> Samples1000: 60
>
> Samples2000: 60
>
> END

Each class of data (tracks, scalars, etc.) begins with the 'kind' field,
subsequent fields refer to this until the next 'kind' (or END). Each
Class that contains binary data must describe the datatype, the file
name (detached or embedded), byte offset and the bytes stored.

## Proposed new fields

-   kind: defines class of subsequent fields, as described below

-   bytes: size of binary data stored for this field (if zero, class is
    > completely described in the header). This defines the size of the
    > payload, allowing a rapid block-read limited to the data desired.

-   name: since a single file might contain multiple scalars or
    > properties, this provides a mechanism to distinguish them, e.g.
    > 'FA', 'MD', 'ADC'. If multiple scalars/properties are provided,
    > each name is encouraged to be unique, e.g. 'TRACEb1000',
    > 'TRACEb2000'. These also provide a mechanism to pair property and
    > property_name classes.

-   A file can have at most one entry of kind tracks. All other kinds
    > are optional (a file may not have any), but multiple instances are
    > allowed.

## Proposed new datatypes

In general, the datatype field defines standard MRtrix
[[datatypes]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#data-types).
However, two new fields are proposed:

-   Float16/Float16LE/Float16BE : see description of the tracks kind
    > below.

-   UTF8null see description of property_name kind below.:

## Kind: tracks

This class corresponds to the
[[TCK]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#tracks-file-format-tck)
format.

-   A file can have at most one 'tracks' class. A file could have none
    > (e.g. a file that just stores scalars, analogous to the current
    > TSF format).

-   Some have suggested a new datatype: Float16 (half-precision). This
    > is not natively supported by modern CPUs, but is common for GPUs
    > (aka mediump). This would lead to better space efficiency, however
    > it is likely to have negative impacts on load/store efficiency and
    > simplicity.

## Kind: scalars

This class corresponds to the
[[TSF]{.ul}](https://mrtrix.readthedocs.io/en/latest/getting_started/image_data.html#track-scalar-file-format-tsf)
format. It provides per-vertex values.

-   Note that the TSF and TRK and formats require that scalars MUST be
    > floating point type. Tools that support TRK may opt to only
    > support 10 distinct scalars in a single file (the limit of that
    > format), but the number is not limited in this proposed format.

## Kind: property

This class mimics the behavior of the TRK format property. It provides a
per-streamline value.

-   Note that TRK format enforces that properties are 32-bit float.
    > Perhaps we should have some discussion if integers are often
    > appropriate. For example, tracks that are segmented into the 30
    > discrete bundles described by
    > [[NatBrainLab]{.ul}](https://www.natbrainlab.co.uk/atlas-maps) are
    > naturally described as integers (e.g. there are regions 13 and 14,
    > but not 13.2). On the other hand, for simplicity we could retain
    > these features as floats, and just have tools round them.

-   Tools that support TRK may opt to only support 10 distinct scalars
    > in a single file (the limit of that format), but the number is not
    > limited in this proposed format.

-   (Not clear if there is demand for this feature) The new format could
    > allow a property to use the compact datatype UInt8 as a mechanism
    > to describe groups. Consider a group "ConnectedToBrodmannArea17"
    > where each streamline is either considered to be connected to this
    > region (1) or not (0). Likewise, another group might be
    > "ConnectedToPutamen". One could modulate visibility with groups
    > (e.g. only show fibers connected to one of these regions, either
    > of these regions or both of these regions) This would provide an
    > alternative method for groups, and would compress well with
    > popular methods like GZip. By this method, each grouping requires
    > the same number of bytes as the number of streamlines (it
    > completely specifies each streamline that belongs to a group. See
    > kind:group for an alternative, sparse implementation.

## Kind: property_name

(Not clear if there is demand for this feature) This provides a
mechanism to name different bundles in a property. A file can include a
property class without a property_name class. On the other hand, a
property_name requires that the file include a property class with an
identical name field. This class has no equivalent in either TCK/TSF or
TRK, and therefore is the most contentious and least fleshed out. The
notion is to allow a mechanism to define different bundes described by a
class, e.g. 'Anterior_Commissure',
'Anterior_Segment_Left','Arcuate_Left'. This class is a series of
null-terminated UTF-8 strings, indexed from zero. Consider a property
that labelled only regions 1,2,4. Even though there are only 3 distinct
bundles, we must define 5 strings \[0,1,2,3,4\]. The list of strings
might be '\\nArea1\\nArea2\\n\\n\\Area4\\n', and the 'count' would be 5.
Inclusion of this class suggests that the paired property should be
treated as integers, even if stored as floats.

-   Like other classes, support for reading and writing this field is
    > totally optional. Tools may choose to ignore this.

## Kind: source

(Not clear if there is demand for this feature) This optional class can
be used for scalars derived from a single-subject's diffusion scan. For
example, the signal to noise of a MeanDiffusity map might be impacted by
the source image voxel size, and the sample averaging. This class would
be omitted for group averages or statistics. Again, this has no origin
in TCK or TRK, so may prove contentious. Based on suggestions by
\@frheault, and hoping to maintain parity with his format.

-   This is an example of a self-contained tag, it does not have any
    > binary data.

-   The fields associated with this can be open to discussion. For
    > example, what factors might we want to include: source voxel
    > volume in mm^3^, magnet field strength, number of b=0 samples,
    > number of samples at specific b\>0 values, etc.

## Kind: groups

(Not clear if there is demand for this feature) This optional class
would provide a way to sparsely label the streamlines that belong to a
specific group. Unlike scalars, this is useful when there is a binary
classification: a streamline either does or does not belong to this
group.This is an alternative approach to usage of datatype for UInt8 in
kind:properties. Whereas that method would require each streamline to be
explicitly coded as either being part of a group or not, here we suggest
that one could sparsely describe only the streamlines that fulfill the
criteria. The datatype would be UInt32, and provide the index number of
the streamlines that survive. Consider a group that lists \[0, 2, 120\]
- in this case only three streamlines qualify as being part of the named
group. These groups could be used to show/hide specific fiber bundles
(e.g. a group might be 'Anterior_Segment_Left'). Another use would be to
identify bundes that connect to some gray matter regions, for example
one group might be "FrontalEyeFields" and another "CalcerineFissure",
and the user could select to identify fibers that belong to either one
of these groups, either of these groups or both groups. The use of the
"groups" class would require less disk space than using "properties"
with a uint8 datatype as long as fewer than 25% of fibers meet the
criteria for inclusion.

4.2 hypothetical format \#1 (.tgy)

-   **Agreement on the coordinate system**

    -   Written on disk in world coordinate (rasmm)

-   **Random-access support with minimal upfront IO**

    -   Reading the header and metadata description is minimal, then
        > blocks of raw data that can be memmap (one for the points and
        > one for each data-per-something)

    -   The approach of data/offset is very efficient for non-complete
        > reading, when ALL streamlines must be read this is only
        > slightly more efficient (block reading, then reconstruction in
        > memory)

    -   This is also very useful for visualisation when only a
        > percentage of the streamlines need to be loaded, shuffling
        > streamlines from a big file, displaying a random portion of
        > the file, etc.

-   **Minimalistic header for \"backward\" compatibility**

    -   Dipy\'s StatefulTractogram friendly, which allows for standalone
        > conversion to tck/trk/vtk and vice-versa

    -   Conversion to tck should be straightforward, trk would required
        > one affine transformation

    -   More importantly this would be a standalone file that would
        > allow all tools to achieve the same computation as they are
        > used to with tck/trk/vtk (without having to provide additional
        > file or external information)

    -   As mentioned on github, the data is written in rasmm so the use
        > of the *vox2rasmm* affine would be optional, but the tag would
        > be mandatory.

        -   Which is also useful for many tools to perform sanity check,
            > this way the superiority of rasmm (of tck) remains, BUT
            > allows for a broader compatibility in the community.

-   **Ability to store streamline groups** (bundles, streamlines
    > connecting pairs of regions, clusters as tables of indices)

    -   This should be declared as tables at the beginning, combined
        > with the memmap approach for data and metadata it would allow
        > to read at very low cost a specific set of streamlines and
        > their metadata.

    -   This is, in my opinion, more extensible and general than storing
        > such information in the data_per_streamline, since it allows
        > indices to be in more than one group and is more compatible
        > with the memmap and random access approach (i.e reading all
        > entries of a metadata tags to obtains a single specific group.

    -   For example, creating an indices table for each region of a
        > parcellation would allow to read only what is necessary to
        > reconstruct the streamlines connecting a specific region

        -   Doing an 'intersection' of two tables would give the indices
            > of the streamlines connecting the pairs, this is
            > irrelevant to the file format, just an example use-case I
            > know is very important for multiple people.

-   **Ability to store None, one or more additional per-streamline
    > value**

    -   Feature from trk, everyone agrees it is useful

-   **Ability to store None, one or more additional per-vertex value**

    -   Feature from trk, everyone agrees it is useful

-   **Ability to store None, one or more properties by streamline
    > group**

    -   This is a feature with no equivalent in trk/tck/vtk, sometimes
        > storing a single value per group (average metrics, volume,
        > score of some sort) is needed.

        -   e.g Coloring of groups would require one RGB per group
            > rather than writing RGB for each streamline like in the
            > trk format.

-   **GPU compatibility**

    -   Array of data and offset is a common approach for polyline in
        > VTK (ITK/MITK too I think) & OpenGL

-   **Compression**

    -   Could be compatible with random access if we choose wisely the
        > compression standard, to discuss

    -   The raw data / offsets approach likely makes the compression
        > much more easier/efficients. *For example, whole file
        > compression using popular format (like .nii.gz) are easy to
        > implement, and allow use of high-performance whole-file
        > compression tools (pigz/libdeflate). On the other hand, they
        > require the decompression of the entire file at once, rather
        > than sequentially decompressing just the portion needed.
        > Further, popular compression methods like gz and zstd are not
        > tuned for the datatypes used in streamline formats. Consider
        > the index offset meta data, this UINT32 reports ordered
        > numbers (the value is always ascending) so neighboring values
        > tend to have similar/identical most significant byte, while
        > the least significant byte almost always changes. The raw byte
        > storage A~1~B~1~C~1~D~1~A~2~B~2~C~2~D~2~\...A~n~B~n~C~n~D~n~
        > will be compressed much more efficiently if saved
        > A~1~A~2~\...A~n~B~1~B~2~\...B~n~... To leverage this method,
        > data is swizzled prior to standard (e.g. gz) compression for
        > saving files. When reading files the data is first
        > decompressed and then un-swizzled. This method is referred to
        > as 'byte interleaving', 'swizzling' or 'byte shuffling' and is
        > described/used by other formats such as
        > [[OpenCTM]{.ul}](http://openctm.sourceforge.net/media/FormatSpecification.pdf)
        > and [[BLOSC]{.ul}](https://blosc.org/pages/blosc-in-depth/).
        > This method can improve compression ratio (storage
        > efficiency), but it does incur costs in terms of read/write
        > efficiency, memory requirements, and simplicity.*

-   **Support for data streaming for very large data / constrained
    > memory operation**

    -   *IF the user does not have groupings and
        > data-per-streamline/point /group*

    -   *OR unless a library/tool decides to have a fancy memory
        > allocation routine with resizing at close time (At the choice
        > of the project implementing their reader/writer)*

-   **Enforce strictly the coherence between everything in the
    > header/metadata/data (length-wise)**. The user should not be able
    > to load/save a file that says it has grouping when in fact does
    > not or if it has the wrong number of data-per-point or if there is
    > a missing point in the raw data, can\'t have metadata per group if
    > there is no grouping, etc. This should be strictly verified at
    > loading AND saving.

    -   Saving only metadata and no streamlines *could* be allowed, but
        > I would like a way to know that. Like a flag in the header or
        > something like *streamline_count: -1*

    -   This is of course at the implementation level, not the standard
        > itself. But I think we should consider "invalid" files that
        > have such inconsistencies.

**MOCKUP ORGANISATION**\
\*\*extensible header\*\*

\- first 8 bytes are file format signature + ASCII [[corruption
detection]{.ul}](http://www.libpng.org/pub/png/spec/1.2/PNG-Rationale.html)

(ASCII C notation) \\211 T G Y \\r \\n \\032 \\n

\- number of streamlines

\- number of 3D points

\- software/version

\- timestamp

\- number of grouping

\- number of data-per-streamline metadata

\- number of data-per-point metadata

\- number of data-per-group metadata

\- datatype of 3D points (if we want to support float16/24/32/64 for
data)

\- raw arrays compression standard? (if we agree on something, maybe
just a boolean)

\- vox2 rasmm affine (same as nifti it was generated from)

*(Just to re-iterate, this would be a mandatory BUT optional to use,
since it would be written on disk in rasmm already)*

\- dimensions (same as nifti it was generated from)

\*\*\[end of header tag\]\*\*

\*\*Datasize information (int64 for offsets)\*\*

\[streamline offsets NBR_STREAMLINE integer\]

\*\*\[end of datasize tag\]\*\*

\*\*Grouping indices (uint32)\*\*

\[group name (STRING) \* NBR_GROUP\]

\[\[group offsets (array of variable size)\] \* NBR_GROUP INTEGER\]

\*\*\[end of grouping tag\]\*\*

\*\*metadata information\*\*

\[datatype for data-per-streamline \* NBR_DPS\]

\[datatype for data-per-point \* NBR_DPP\]

\[datatype for data-per-group \* NBR_DPG\]

\[data-per-streamline name (string) \* NBR_DPS\]

\[data-per-point name (string) \* NBR_DPP\]

\[data-per-group name (string) \* NBR_DPG\]

\*\*\[end of metadata tag\]\*\*

\*\*Raw data\*\*

\[vertex position NBR_POINTx3 float16/32/64 \]

\[per-streamline property \#0 \* NBR_STREAMLINE\] (optional)

\[per-streamline property \#1 \* NBR_STREAMLINE\] (optional)

\...

\[per-point scalar \#0 \* NBR_POINE\] (optional)

\[per-point scalar \#1 \* NBR_POINT\] (optional)

\...

\[per-group scalar \#0 \* NBR_GROUP\] (optional)

\[per-group scalar \#1 \* NBR_GROUP\] (optional)

PS : I (Francois) am not as familiar with the very specific way of
designing a file format. I coded in c++ a long time ago a TRK reader in
MITK and a PNG reader from scratch (only experience related to file
format in c++), but never dealt with memmap outside of python. So if
anything I wrote is impossible or very inefficient due to a CPU
limitation in reading/writing, do tell me/us your concerns since I am
probably not aware of it. Based on other file formats I dealt with, I
think all of it is possible, but maybe it requires some magic!
