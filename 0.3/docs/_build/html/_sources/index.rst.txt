.. trx-python documentation master file, created by
   sphinx-quickstart on Fri Jun 24 23:14:56 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TRX: A community-oriented tractography file format
===================================================

We propose **TRX**, a tractography file format designed to facilitate dataset
exchange, interoperability, and state-of-the-art analyses, acting as a
community-driven replacement for the myriad existing file formats.


Getting Started
~~~~~~~~~~~~~~~

New to TRX? Start here:

1. **Understand the format**: Read the :doc:`trx_specifications` to understand the TRX file structure
2. **Learn by example**: Follow our :doc:`auto_examples/index` to learn how to read, write, and manipulate TRX files
3. **Use the CLI tools**: Check out the :doc:`scripts` documentation for command-line operations

.. grid:: 2

   .. grid-item-card:: Tutorials
      :link: auto_examples/index
      :link-type: doc

      Learn how to work with TRX files through hands-on tutorials covering
      reading/writing files, working with groups, and using metadata.

   .. grid-item-card:: TRX Specifications
      :link: trx_specifications
      :link-type: doc

      Complete technical specifications of the TRX file format including
      header fields, array structures, and naming conventions.


Why TRX?
~~~~~~~~
File formats that store the results of computational tractography were typically
developed within specific software packages. This approach has facilitated a
myriad of applications, but this development approach has also generated
insularity within software packages, and has limited standardization. Moreover,
because tractography file formats were developed to solve immediate challenges,
only a limited breadth of applications within a single software package was
envisioned, sometimes also neglecting computational performance. Given the
growing interest in tractography methods and applications, and the increasing
size and complexity of datasets, a community-driven standardization of
tractography have become a priority. To address these challenges, our community
initiated a discussion to design a new file format and agreed to participate in
its conception, development, and, if successful, its adoption.

The goal of TRX is to become the first, community-driven, standard amongst
tractography file formats. As with other file formats like NiFTI, we believe
that TRX will serve the community well and the growing computational needs of
our field. We encourage community members to consider early contributions to our
proposal so as to ensure the new standard will cover the needs of the wider
audience of software developers, toolboxes, and scientists. Our long-term plan
is to integrate TRX within the `Brain Imaging Data Structure (BIDS) <https://bids.neuroimaging.io/>`_ ecosystem.


Acknowledgments
~~~~~~~~~~~~~~~~
Development of TRX is supported by `NIMH grant 1R01MH126699 <https://reporter.nih.gov/search/D-c1NJBQFE-BkwP1Cxc2Lg/project-details/10253558>`_.


.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   trx_specifications
   scripts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: Development:

   dev

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   autoapi/index
