# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'jaxkan'
copyright = '2024-2025, Spyros Rigas, Michalis Papachristou'
author = 'Spyros Rigas, Michalis Papachristou'

# The full version, including alpha/beta/rc tags
release = '0.1.9'

# -- General configuration ------------------------------------------------

exclude_patterns = []

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx.ext.mathjax", 
    "sphinx.ext.coverage",
    "nbsphinx",
    "nbsphinx_link"
]

language = 'en'

templates_path = ['_templates']

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']