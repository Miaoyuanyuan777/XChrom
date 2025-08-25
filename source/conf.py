# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XChrom'
copyright = '2025, Yuanyuan Miao'
author = 'Yuanyuan Miao'
release = 'v1.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',    # autodoc
    'sphinx.ext.viewcode',   # add source code link
    'sphinx.ext.napoleon',    # support Google style documentation
    'sphinx.ext.autosummary',
    'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = []
# enable autosummary
autosummary_generate = True
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


import sys
import os
sys.path.insert(0, os.path.abspath('../../XChrom'))

html_last_updated_fmt = '%Y-%m-%d'
html_domain_indices = True
html_logo = '_static/XChrom_logo.png'
html_favicon = '_static/XChrom_icon.svg'
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_depth": 1,
    "titles_only": True,
    'logo_only': True,
    'style_nav_header_background': '#E6E6FA',
}

autodoc_mock_imports = [
    "rpy2"
]