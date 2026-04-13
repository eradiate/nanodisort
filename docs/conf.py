"""Sphinx configuration for nanodisort documentation."""

import datetime

import nanodisort

# Project information
project = "nanodisort"
copyright = f"2025-{datetime.datetime.now().year}, Rayference"
author = "Vincent Leroy"
version = nanodisort.__version__
release = version

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "myst_nb",
]

# Napoleon settings (for Numpydoc)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# BibTeX configuration
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# Templates and static files
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "shibuya"
html_static_path = ["_static"]
html_title = "nanodisort"

html_theme_options = {
    "github_url": "https://github.com/rayference/nanodisort",
    "nav_links": [
        {"title": "Home", "url": "index"},
        {"title": "API Reference", "url": "api"},
    ],
    "accent_color": "amber",
}

# Autodoc options
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_typehints_format = "short"

# myst-nb options
nb_execution_mode = "off"

# MyST options
myst_enable_extensions = ["dollarmath"]
