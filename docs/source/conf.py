"""Configuration file for the Sphinx documentation builder."""
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

release = importlib.metadata.version("abinslib")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "abinslib"
copyright = "2026, STFC"  # noqa: A001
author = "Adam J. Jackson / Science and Technology Facilities Council"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["tutorials/GALLERY_HEADER.rst"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

_PALE = "#dae9e0"
_DESATURATED = "#97b1ab"
_BRIGHT = "#c4fcf0"
_LIGHT_GREEN = "#4eae9e"
_GREEN = "#00796b"
_TURQUOISE = "#007e8c"
_DARK_GREEN = "#344b47"
_DARKEST = "#00392f"

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": _GREEN,
        "color-brand-content": _GREEN,
        "color-brand-visited": _DESATURATED,
        "color-link": _GREEN,
        "color-link--visited": _TURQUOISE,
        "color-link--hover": _DESATURATED,
        "color-admonition-title--note": _LIGHT_GREEN,
        "color-admonition-title-background--note": _BRIGHT,
        "color-api-keyword": _DARK_GREEN,
        "color-api-pre-name": _GREEN,
        "color-api-name": _DARK_GREEN,
        # The next three tags are not built-ins, but referred to as variables
        # in _static/custom.css; this gets the auto dark/light mode working.
        "color-download-button-bg": _PALE,
        "color-download-gradient-top": _PALE,
        "color-download-gradient-bottom": _LIGHT_GREEN,
    },
    "dark_css_variables": {
        "color-brand-primary": _LIGHT_GREEN,
        "color-brand-content": _LIGHT_GREEN,
        "color-brand-visited": _TURQUOISE,
        "color-link": _LIGHT_GREEN,
        "color-link--visited": _GREEN,
        "color-link--hover": _TURQUOISE,
        "color-admonition-title--note": _LIGHT_GREEN,
        "color-admonition-title-background--note": _DARK_GREEN,
        "color-admonition-background": _DARKEST,
        "color-download-button-bg": _DESATURATED,
        "color-download-gradient-top": _LIGHT_GREEN,
        "color-download-gradient-bottom": _GREEN,
    },
}

# -- AutoAPI configuration ---------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../src"]
autoapi_keep_files = False  # Set true for debugging
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

autodoc_typehints = "signature"

# -- Sphinx gallery setup ----------------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "tutorials",
    "gallery_dirs": "auto_examples",
    "remove_config_comments": True,
}

# -- Napoleon options --------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

napoleon_use_param = True
napoleon_use_rtype = True
