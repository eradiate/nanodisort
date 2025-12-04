"""Sphinx configuration for nanodisort documentation."""

import os
import sys

# Add source to path
sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "nanodisort"
copyright = "2024, Vincent Leroy"
author = "Vincent Leroy"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
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
}

# Autodoc options
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
