# Higher quantum orders by convolution

`abinslib.convolution.convolve_orders` derives quantum-order `n` spectra from
a binned order-`n-1` spectrum and a binned order-1 spectrum (the fundamentals)
by discrete convolution. It applies only the convolution and the bookkeeping
around it: grid checks, bin placement, density units and metadata. Scale
factors and cross-section weights are applied afterwards, to the returned
collection.

```python
from abinslib.convolution import convolve_orders
from abinslib.util import apply_weights

orders = convolve_orders(order1, order1, max_order=4)  # orders 2, 3 and 4
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

## Bin positions

Convolution adds bin positions. Energy zero must therefore lie on a bin centre
or on a bin edge of the fundamentals grid, so that convolved bins land on the
output grid. The grid does not need to include zero within its range.

| Input axis                              | Zero falls on | Reading used | Warning                         |
| --------------------------------------- | ------------- | ------------ | ------------------------------- |
| Bin edges, one edge at zero             | edge          | lower edge   | yes, unless `bin_position` given |
| Bin edges at half-spacing offsets       | centre        | centre       | no                              |
| N sampling points, one at zero          | centre        | centre       | no                              |
| N sampling points, one at half a spacing | edge         | lower edge   | yes, unless `bin_position` given |

With the **centre** reading, each bin is located at its centre, and an order-`n`
bin sits where the sum of the centres of its contributing bins sits. This is
the reading to use when accuracy in energy matters.

With the **lower-edge** reading, each bin is located at its lower edge, as in
Mantid-Abins. Bin indices then add, so output bin `m` holds the sum over pairs
`i + j = m`. This introduces a systematic offset: intensity of order `n` sits
`(n - 1) / 2` bins above the centre of the bin it is reported in.

Synthetic example: a fundamentals grid with edges at `0, 1, 2, 3` meV and a
single occupied bin (index 0) with intensity 1 per meV. The lower-edge reading
reports the order-2 intensity in bin 0 (centre 0.5 meV). The true energy of
that intensity is the sum of the two bin centres, 0.5 + 0.5 = 1.0 meV, which is
half a bin above the reported centre. Order 3 is offset by one bin, as
`(3 - 1) / 2` predicts. This example is constructed for illustration and is not
a measurement of Mantid-Abins output.

Passing `bin_position="centre"` or `bin_position="lower_edge"` states the
reading explicitly. A value that does not match the grid raises an error, and
an explicit `"lower_edge"` silences the warning. Fractional placement, such as
splitting bins between two output bins, is not implemented.

## Units and metadata

Inputs and outputs are intensity densities per unit energy. Each convolution
multiplies by the fundamentals' bin width once, so that the result approximates
the continuous convolution. The output units are the product of the initial
spectrum's intensity unit, the fundamentals' intensity unit and the energy
unit.

Every returned row uses the initial spectrum's energy axis, in the form it was
given (bin edges or centres). Other metadata of the initial spectrum, such as
`atom_symbol` and `mass`, is carried onto each row, so that `apply_weights`
works on the result. The quantum order of row `k` is the initial order plus
`k`.
