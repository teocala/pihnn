# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PIHNN'
copyright = '2024, Matteo Calafà'
author = 'Matteo Calafà'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os, sys
sys.path.insert(0, os.path.abspath(".."))

extensions = ["sphinx.ext.mathjax",'sphinx.ext.viewcode','sphinxcontrib.youtube','autoapi.extension']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

templates_path = ['_templates']
autoapi_dirs = ['../pihnn']
autoapi_python_class_content = 'both'
autoapi_own_page_level = 'function'
autoapi_template_dir = '_templates/autoapi'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'pydata_sphinx_theme'
html_show_sourcelink = False
html_logo = "https://matteocalafa.com/images/pihnn-logo.svg"
html_favicon = "https://matteocalafa.com/images/pihnn-logo-small.svg"
html_theme_options = {
  "external_links": [
      {"name": "Paper", "url": "https://doi.org/10.1016/j.cma.2024.117406"},
  ],
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/teocala/pihnn",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
   ],
   # Note we have omitted `theme-switcher` below
   "navbar_end": ["navbar-icon-links"],
   "navbar_persistent": ["search-button"]
}
html_context = {
   "default_mode": "light"
}

# To remove class attributes from docs
def skip_util_classes(app, what, name, obj, skip, options):
    if what == "attribute":
       skip = True
    return skip

def setup(sphinx):
   sphinx.connect("autoapi-skip-member", skip_util_classes)