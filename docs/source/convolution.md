# Higher quantum orders by convolution

{py:func}`~abinslib.convolution.convolve_orders` derives quantum-order `n`
spectra from a binned order-`n-1` spectrum and a binned order-1 spectrum (the
fundamentals) by discrete convolution. It applies only the convolution and the
bookkeeping around it: grid checks, placement, density units and metadata. Scale
factors and cross-section weights are applied afterwards, to the returned
collection. The output units and the metadata carried onto each row are
specified in the function's API documentation.

```python
from abinslib.convolution import convolve_orders
from abinslib.util import apply_weights

orders = convolve_orders(order1, order1, max_order=4)  # orders 2, 3 and 4
# (order1 has a bin centre at zero; see "Sample positions" below)
weighted = apply_weights(orders)
```

## Approximations

**Isotropic approximation.** Convolving scalar spectra assumes that the
orientation-averaged scattering factorises into a product of per-order
spectra. This is exact for isotropic tensors and an approximation otherwise.

**Negative-energy terms.** Mantid-Abins neglects the negative-energy (gain)
terms. When the fundamentals grid starts at a negative bin centre,
`convolve_orders` keeps contributions that leave the output grid and return
through a negative-energy step. This costs more memory and time, and is
selected automatically from the grid.

## Sample positions

The convolution process adds lower-order frequencies to get the
positions of higher-order modes (in the harmonic approximation).
In `convolve_orders` the frequencies are meshes of input sample positions.
The fundamental spectrum is used as a discrete convolution kernel and
its lattice must include zero; this is not required for the output grid.

A grid with edges at zero, such as `numpy.linspace(0, e_max, n + 1)`, has zero
on a bin edge instead. To use one, pass the lower edges as N sampling points,
as described next.

### Reproducing Mantid-Abins

Mantid-Abins locates each bin at its lower edge, so that bin indices
add: output bin `m` holds the sum over pairs `i + j = m`. To reproduce
this from a `Spectrum1D` whose bin edges include zero (as above), pass
the lower edges as N sampling points for both spectra, and put the
results back on the original axis. If a bin centre is already at zero,
pass the spectrum to `convolve_orders` directly.

```python
from euphonic.spectra import Spectrum1D, Spectrum1DCollection

# The lower edge of every bin, as N sampling points.
# restrict_range=False allows expansion to maintain regular spacing
points = Spectrum1D(
    spectrum.get_bin_edges(restrict_range=False)[:-1],
    spectrum.y_data,
    metadata=spectrum.metadata,
)
# initial_order can be omitted if the metadata has a "quantum_order"
rows = convolve_orders(points, points, max_order=4, initial_order=1)

# The same bins, on the original axis
orders = Spectrum1DCollection(spectrum.x_data, rows.y_data, metadata=rows.metadata)
```

```{warning}
Reproducing Mantid-Abins by passing the lower edges as sampling points
introduces a systematic offset: intensity of order `n` truly sits
`(n - 1) / 2` bins above the centre of the bin it is reported in, so the
reported spectrum is displaced low by that amount.
```

Splitting intensity between output bins, or resampling onto a grid with edges
at zero, is not part of this function. A resampling wrapper is planned for a
later change.
