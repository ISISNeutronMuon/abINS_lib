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

## Convolution benchmark

`validation/scripts/` also times the higher-order convolution in
`abinslib.convolution` against Mantid-Abins autoconvolution. It runs in two
steps because Mantid and abinslib need different Euphonic versions. Run it
on demand; it is not part of CI. Inputs and results are written to
`$TMPDIR/abinslib-convolution-benchmark` (or a directory given as an
argument), not to the repository.

From `validation/`, step 1 times Mantid's `add_autoconvolution_spectra`
(single thread, convolution only) and stores the inputs and results:

```
pixi run benchmark-convolution-mantid
```

From the abinslib root, step 2 times the array-level routine on the same
inputs, checks agreement with Mantid's new orders, reports the per-atom
overhead of the public `convolve_orders` call, and checks the regression
gate at the largest mesh with 9 atoms:

```
uv run python dev/validation/scripts/convolution_bench_abinslib.py
```

Both steps report the minimum wall time over five repeats, and peak memory
from a separate traced call.
