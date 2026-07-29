[![Ruff][ruff-badge]][ruff-link]
[![Run tests][ci-badge]][ci-link]
[![Documentation][docs-badge]][docs-link]
[![License][license-badge]][license-link]

# abinslib
A library for dynamical structure factor calculations from phonon data in the harmonic approximation.

**This is still in alpha: if using for production, pin a minor version number (e.g. `abinslib~=0.1.0`)**

Abinslib implements a variety of simulation methods for inelastic neutron scattering (INS) from phonons,
in the general CLIMAX/abINS lineage.  Most of these approximations have not been rigorously compared,
so this is intended to provide both
- reference implementations that can be called by user-facing simulation software; and
- a playground for method development and validation in this area.

## Testing and linting

Install with testing dependency group, e.g. from this directory

```
pip install . --group test
```

and run with pytest

```
pytest
```

You may need to update Pip to access the `--group` feature.

New code should be accompanied with appropriate tests; use

`pytest --cov --cov-report term-missing`

to get a breakdown of which lines are not covered by tests. (This can
also be found in the Github Actions output when making a Pull Request.)
As well as unit testing, code contributions must pass linting with
`ruff check` and `ruff format`.

The *pytest-regressions* package is used to create/test reference
outputs for functions given particular input. This gives confidence
that results do not change unexpectedly between versions, but such
["regression tests"](https://en.wikipedia.org/wiki/Regression_testing)
are not necessarily validated against an ideal or external reference.

## Validation status

So far we have implemented some of the calculation methods from
Mantid-Abins: 1D fundamental spectra in the fully-isotropic and
almost-isotropic incoherent approximation, and 1D second-order spectra
in the almost-isotropic incoherent approximation. This is benchmarked
against the equivalent calculation in Mantid: a Snakemake workflow in
*dev/validation* is used to produce reference data, and this is
plotted against abinslib calculations in the documentation gallery.

![](https://github.com/ISISNeutronMuon/abINS_lib/blob/fed5dc022e887998ec09f9e90ec6e8c9e6b1c5db/dev/validation/results/second-order-plot.png)

To reproduce the result so closely it was necessary to apply similar
implementation details: the `q_scaling_isotropic_incoherent_spectra`
function computes mode intensities at Q = 1Å and bins them to a dense
spectrum before applying Q^n2/n! scaling and Debye—Waller factor for
the bin-centre Q values. In practice this gives an acceptable
discretisation error of ~0.1%.

[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-link]: https://github.com/astral-sh/ruff
[ci-badge]: https://github.com/isisneutronmuon/abINS_lib/actions/workflows/test.yml/badge.svg
[ci-link]: https://github.com/isisneutronmuon/abINS_lib/actions/workflows/test.yml
[docs-badge]: https://img.shields.io/badge/docs-online-00796B
[docs-link]: https://isisneutronmuon.github.io/abINS_lib/
[license-badge]: https://img.shields.io/badge/License-GPLv3-blue.svg
[license-link]: https://www.gnu.org/licenses/gpl-3.0
