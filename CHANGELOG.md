# Changelog

See [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### API and Data Model Changes
- **BREAKING**: `mantid_like_combination_spectra` now returns a flattened collection with one spectrum per atom instead of one per (q-point, atom). This is the behavior the discarded `group_by("atom_index")` call always intended.
  - Return shape changed from `(n_qpts * n_atoms, n_bins)` to `(n_atoms, n_bins)`.
  - Per-row `metadata["qpt"]` is no longer present in the result. It only survived due to the broken grouping and is meaningless on a spectrum combining multiple q-points.
  - Downstream consumers that reduce over rows (e.g., `.sum()`) are numerically unaffected.

### Performance Improvements
- **Linear scaling with q-point count**: `mantid_like_combination_spectra` now accumulates per-q-point contributions as arrays and constructs the result collection once, making the function linear in q-point count (O(n)) instead of quadratic (O(n²)).
  - Peak memory usage reduced from ~24 MB to ~0.7 MB at 64 q-points for a test system.
  - Execution time at 128 q-points reduced from ~44 s to ~0.45 s (measured on synthetic ethanol data with 27 bands, 9 atoms, 1000 bins).

### Bug Fixes
- Fixed discarded `group_by("atom_index")` call in `mantid_like_combination_spectra` that caused incorrect return shape.

## [0.2.0] - 2026-08-28

This release includes improvements to tutorial and validation workflows, and adds JSON serialization for `Displacements`. The main breaking change is the decoupling of cross-section weighting into an explicit `apply_weights` step.

### API and Data Model Changes
- **Decoupled cross-section weighting from intensity calculations**: Intensity calculation functions now **always compute unweighted spectra** (dimensionless / cross-section-free intensity units).
  - Removed `apply_cross_sections` boolean parameters from intensity calculation routines.
  - Added explicit `abinslib.util.apply_weights` function. Callers are now responsible for applying neutron cross-sections to unweighted spectra when desired.
  - Spectrum metadata (`symbols`, `masses`) is used with Euphonic's `IsotopeData` protocol to resolve isotope cross sections, avoiding the need to pass `Crystal` objects into intensity routines.
  - Corrected units handling to ensure raw spectra are cross-section-free.

### New Features
- **JSON Serialization**: Added `abinslib.io` module with `JsonMixin`: this provides  `to_json`, `from_json`, `to_json_file`, and `from_json_file` methods to classes that implement `to_dict` and `from_dict`.
  - Includes and validates API version and class metadata on deserialization.
  - This functionality is added to the `Displacements` class.  
- **Sample Data Management**: Added `abinslib.data` module using `pooch` to fetch sample and validation data on demand from machine-readable registries in `abinslib.registries`. This keeps the source repository light while making realistic data conveniently available for tutorials and validation.

### Documentation & Tutorials
- Converted Mantid validation benchmarks into executable Sphinx Gallery examples (`plot_validation.py`) rendered directly in the online documentation.
- Added tutorials illustrating TOSCA INS simulation workflows.
- Configured Sphinx documentation with `furo` theme, `sphinx-autoapi`, `myst-parser`, and custom styling.

### Maintenance & Workflows
- Overhauled `dev/validation` workflows using `pixi` and Snakemake for hashing and generating reference validation datasets.
- Updated repository URLs to `https://github.com/isisneutronmuon/abINS_lib` and documentation to `https://isisneutronmuon.github.io/abINS_lib`.
- Configured Ruff linting and docstring style enforcement across the repository.

---

## [0.1.0] - 2026-07-16

- Initial release of `abinslib`.
- Support for 1-D and 2-D almost-isotropic and isotropic incoherent inelastic neutron-scattering intensities from harmonic phonon modes.
- Integration with Euphonic 2.0 and ResINS 0.1.

[Unreleased]: https://github.com/isisneutronmuon/abINS_lib/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/isisneutronmuon/abINS_lib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/isisneutronmuon/abINS_lib/releases/tag/v0.1.0
