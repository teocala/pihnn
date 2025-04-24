Documentation
================

This page contains auto-generated API reference documentation.
We recommend to read the relevant literature in :doc:`../../publications`.


Modules
--------

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}

