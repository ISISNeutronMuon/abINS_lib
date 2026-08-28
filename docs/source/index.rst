abinslib documentation
======================

Documentation for the abinslib library.

This library collects implementations of inelastic neutron-scattering
intensity calculations from harmonic phonon data. It relies on
`Euphonic <https://euphonic.readthedocs.io/en/stable/>`_ for phonon
data structures, while instrument resolution functions are delegated
to `ResINS <https://pace-neutrons.github.io/resins/>`_.

.. important::

  The library is currently in alpha state: we expect that functions will
  be renamed and moved around. In order to use this package in prototype
  code, please pin to the minor version number e.g. ``abinslib~=0.1.0``.

  When the API is stable, the major version number will be increased to
  1.0. This follows `Semantic Versioning <https://semver.org/>`_ as also
  used by Euphonic.

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   auto_examples/index
   development/index
   changelog
   API Reference <autoapi/abinslib/index>
