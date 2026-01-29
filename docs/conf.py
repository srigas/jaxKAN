# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'jaxkan'
copyright = '2025, Spyros Rigas, Michalis Papachristou'
author = 'Spyros Rigas, Michalis Papachristou'

release = '0.3.5'

# -- General configuration ------------------------------------------------

root_doc = "index"

language = 'en'

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

autosummary_generate = True
autodoc_member_order = "bysource"
autosectionlabel_prefix_document = True

# -- Options for HTML output ----------------------------------------------

html_theme = 'sphinx_rtd_theme'