# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
from datetime import datetime as dt

# -- Version information -----------------------------------------------------
# Get version from environment variable (set by CI) or package
version = os.environ.get('TRX_VERSION', None)
if version is None:
    try:
        from trx import __version__
        version = __version__
    except ImportError:
        version = "dev"

# Normalize version for switcher matching
# Remove .devX suffix for matching against switcher.json
version_match = version.split('.dev')[0] if '.dev' in version else version
if version_match == version and 'dev' not in version:
    # This is a release version
    pass
else:
    # Development version - match against "dev"
    version_match = "dev"

# -- Project information -----------------------------------------------------

project = 'trx-python'
copyright = copyright = f'2021-{dt.now().year}, The TRX developers'
author = 'The TRX developers'
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'autoapi.extension',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../_static']
html_logo = "../_static/trx_logo.png"


html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/tee-ar-ex",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    # Version switcher configuration
    "switcher": {
        "json_url": "https://tee-ar-ex.github.io/trx-python/switcher.json",
        "version_match": version_match,
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "show_version_warning_banner": True,
}


autoapi_type = 'python'
autoapi_dirs = ['../../trx']
autoapi_ignore = ['*test*', '*version*']
