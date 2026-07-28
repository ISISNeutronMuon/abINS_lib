This directory contains scripts and notes used in development. It
should not be included in the distributed package.

## Validation

A workflow to generate benchmark data from Mantid-Abins. This requires
Mantid, which we prefer not to deal with in the main test suite or
documentation builds. Here we use pixi to define a Conda environment
and some useful commands.

The files generated here are then uploaded (by hand) to somewhere they
can be found by `abinslib.data.get_validation_data()`, which will
check that the file hashes match the expected values. This allows docs
builds to validate fresh abinslib results against archived Mantid
results.

To generate the data, make sure pixi is available and you are on a
Mantid-supported platform, then from the *validation/* directory:

```
pixi run generate
```

You can then get the results file hashes with

```
pixi run hashes
```

and create a .zip archive with

```
pixi run archive
```

Note that your hashes might differ from the reference due to small
numerical changes, so unexpected hashes don't _necessarily_ mean
something has gone badly wrong.
Local builds of the docs will prefer files in dev/validation/results
to their online equivalents, so you can check the generated plots to
see if a genuine mismatch has appeared.