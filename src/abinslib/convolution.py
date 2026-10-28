"""Derive higher quantum-order spectra by discrete convolution.

Repeated convolution of a binned order-n spectrum with a binned order-1
(fundamentals) spectrum gives the order-(n + k) spectrum. The operation here
applies only the convolution and the bookkeeping that goes with it: grid
validation, bin placement, density units and metadata. Scale factors and
cross-section weights are applied afterwards by the caller.

Approximations and conventions:

- Convolving scalar spectra assumes the isotropic approximation, in which the
  orientation-averaged scattering factorises into a scalar product of
  per-order spectra.
- Mantid-Abins neglects the negative-energy (gain) terms. The padded
  two-sided path in this module includes them when the fundamentals grid
  starts at a negative bin centre.
- Energy zero must lie on a bin centre or a bin edge of the fundamentals
  grid. Bins are located at their centres when zero is on a centre, and at
  their lower edges when zero is on an edge (the Mantid-Abins convention).
  With the lower-edge reading, intensity of order n sits (n - 1) / 2 bins
  above the centre of the bin it is reported in.

Grid readings, shown for a bin of width ``d`` centred on zero:

.. list-table::
   :header-rows: 1

   * - Input axis
     - Zero falls on
     - Reading used
     - Warning issued
   * - Bin edges, one edge at zero
     - edge
     - lower edge
     - yes, unless ``bin_position`` is given
   * - Bin edges at half-spacing offsets
     - centre
     - centre
     - no
   * - N sampling points, one at zero
     - centre
     - centre
     - no
   * - N sampling points, one at half a spacing
     - edge
     - lower edge
     - yes, unless ``bin_position`` is given

Scale factors and weights are not part of this module; they are applied to
the returned collection.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from euphonic import Quantity
from euphonic.spectra import Spectrum1D, Spectrum1DCollection
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

_REL_TOL = 1e-6
_READINGS = ("centre", "lower_edge")


@dataclass(frozen=True)
class _GridInfo:
    """Regular energy grid of a spectrum.

    Attributes:
        spacing: Bin width, in ``unit``.
        first_edge: Lower edge of the first bin, in ``unit``.
        unit: Energy unit string of the axis.
        length: Number of intensity values (bins).
        is_edges: True if the axis was given as bin edges, False if as centres.
        edge_offset: Distance from energy zero to the nearest bin edge at or
            below it, as a fraction of the spacing, in [0, 1).
    """

    spacing: float
    first_edge: float
    unit: str
    length: int
    is_edges: bool
    edge_offset: float


@dataclass(frozen=True)
class _Placement:
    """Bin reading and integer shift of a zero-aligned grid.

    Attributes:
        reading: ``"centre"`` or ``"lower_edge"``.
        shift: Position of the first bin, in units of the spacing.
    """

    reading: str
    shift: int


def _grid_from_axis(x_data: Quantity, n_values: int) -> _GridInfo:
    """Read an energy axis once and check that it is regular.

    Args:
        x_data: Energy values, as bin edges (``n_values + 1``) or bin centres
            (``n_values``).
        n_values: Number of intensity values on the axis.

    Returns:
        Grid description in the axis' own energy unit.

    Raises:
        ValueError: If the axis lengths are inconsistent, the axis has fewer
            than two bins, it is not increasing, or the bins are irregular.
    """
    x = np.asarray(x_data.magnitude, dtype=float)
    unit = str(x_data.units)

    # Spectrum1D validates that x has N + 1 (edges) or N (centres) values.
    is_edges = x.size == n_values + 1

    diffs = np.diff(x)
    if diffs.size == 0:
        raise ValueError("At least two energy values are needed to define a bin width.")

    spacing = float(diffs.mean())
    if spacing <= 0:
        raise ValueError("Energy axis must be increasing.")

    spread = float(np.ptp(diffs))
    if spread > _REL_TOL * spacing:
        raise ValueError(
            "Energy grid is irregular: bin widths range from "
            f"{diffs.min():.6g} to {diffs.max():.6g} {unit}."
        )

    first_edge = float(x[0]) if is_edges else float(x[0]) - spacing / 2
    edge_offset = float((-first_edge / spacing) % 1.0)
    return _GridInfo(
        spacing=spacing,
        first_edge=first_edge,
        unit=unit,
        length=n_values,
        is_edges=is_edges,
        edge_offset=edge_offset,
    )


def _same_axis(a: Quantity, n_a: int, b: Quantity, n_b: int) -> bool:
    """True if two energy axes and intensity lengths are identical."""
    if n_a != n_b:
        return False
    if a is b:
        return True
    return bool(
        np.array_equal(a.magnitude, b.magnitude) and str(a.units) == str(b.units)
    )


def _zero_placement(grid: _GridInfo) -> _Placement:
    """Find the bin reading and integer shift from where energy zero falls.

    Args:
        grid: Regular grid of the fundamentals.

    Returns:
        The natural reading for the grid and the integer shift of its first
        bin.

    Raises:
        ValueError: If zero is neither on a bin centre nor on a bin edge.
    """
    offset = grid.edge_offset
    distance_to_edge = min(offset, 1.0 - offset)
    if distance_to_edge <= _REL_TOL:
        reading = "lower_edge"
        first_bin = grid.first_edge
    elif abs(offset - 0.5) <= _REL_TOL:
        reading = "centre"
        first_bin = grid.first_edge + grid.spacing / 2
    else:
        raise ValueError(
            "Energy zero lies "
            f"{distance_to_edge:.4g} bin spacings "
            f"({distance_to_edge * grid.spacing:.4g} {grid.unit}) from the nearest "
            "bin edge of the fundamentals grid. Convolution requires energy zero "
            "on a bin centre (0.5 spacings from the nearest edge) or on a bin "
            "edge (0 spacings)."
        )
    shift = round(first_bin / grid.spacing)
    return _Placement(reading=reading, shift=shift)


def _shift_crop(values: np.ndarray, shift: int, length: int) -> np.ndarray:
    """Move values by an integer number of bins and crop to a fixed length.

    Output index m takes input index m - shift; entries that fall outside
    ``[0, length)`` are dropped and uncovered entries are zero.

    Args:
        values: Input array indexed from the start of the window.
        shift: Integer displacement of the values.
        length: Length of the output window.

    Returns:
        Array of ``length`` values.
    """
    out = np.zeros(length)
    if shift >= 0:
        n = min(length - shift, values.size)
        if n > 0:
            out[shift : shift + n] = values[:n]
    else:
        n = min(length, values.size + shift)
        if n > 0:
            out[:n] = values[-shift : -shift + n]
    return out


def _convolve_orders_one_sided(
    initial_values: np.ndarray,
    fundamentals_values: np.ndarray,
    n_new: int,
    *,
    shift: int,
    spacing: float,
) -> np.ndarray:
    """Convolve repeatedly with cropping at every order, for ``shift >= 0``.

    With a non-negative shift no contribution can return to the grid after
    leaving it, so each order can be cropped to the initial grid. The
    fundamentals' transform is computed once; each order uses one forward and
    one inverse transform, with cropping in real space.

    Args:
        initial_values: Intensity densities on the initial grid.
        fundamentals_values: Intensity densities of the fundamentals.
        n_new: Number of higher orders to return.
        shift: Integer shift of the fundamentals' first bin, in bins.
        spacing: Bin width of the fundamentals, in the initial grid's energy
            unit.

    Returns:
        Array of shape ``(n_new, N)`` with one row per order.
    """
    n = initial_values.size
    out = np.zeros((n_new, n))
    kernel_length = min(fundamentals_values.size, n - shift)
    if kernel_length <= 0:
        return out

    kernel = fundamentals_values[:kernel_length] * spacing
    nfft = next_fast_len(n + kernel_length - 1, real=True)
    kernel_hat = rfft(kernel, nfft)

    row = initial_values
    for k in range(n_new):
        conv = irfft(rfft(row, nfft) * kernel_hat, nfft)[:n]
        row = _shift_crop(conv, shift, n)
        out[k] = row
    return out


def _convolve_orders_padded(
    initial_values: np.ndarray,
    fundamentals_values: np.ndarray,
    n_new: int,
    *,
    shift: int,
    spacing: float,
) -> np.ndarray:
    """Convolve repeatedly keeping a padded window, valid for any shift.

    Intermediate orders are kept on a window wide enough that no value outside
    it can return to the initial grid within the remaining orders. The window
    is the widest one needed (after ``n_new - 1`` further orders), so every
    order uses the same padded length. Works for ``shift < 0`` (contributions
    that leave the grid and return), and gives the same result as the
    one-sided path for ``shift >= 0``.

    Args:
        initial_values: Intensity densities on the initial grid.
        fundamentals_values: Intensity densities of the fundamentals.
        n_new: Number of higher orders to return.
        shift: Integer shift of the fundamentals' first bin, in bins.
        spacing: Bin width of the fundamentals, in the initial grid's energy
            unit.

    Returns:
        Array of shape ``(n_new, N)`` with one row per order.
    """
    n = initial_values.size
    length = fundamentals_values.size
    kernel = fundamentals_values * spacing

    # Window bounds (grid indices) for an intermediate with r further orders
    # remaining: [-r * (shift + length - 1), N - 1 - r * shift]. The widest
    # case is r = n_new - 1; the initial grid is always inside it.
    r_max = n_new - 1
    lo = min(0, -r_max * (shift + length - 1))
    hi = max(n - 1, n - 1 - r_max * shift)
    width = hi - lo + 1
    grid_slice = slice(-lo, -lo + n)

    nfft = next_fast_len(width + length - 1, real=True)
    kernel_hat = rfft(kernel, nfft)

    row = np.zeros(width)
    row[grid_slice] = initial_values

    out = np.zeros((n_new, n))
    for k in range(n_new):
        conv = irfft(rfft(row, nfft) * kernel_hat, nfft)[: width + length - 1]
        row = _shift_crop(conv, shift, width)
        out[k] = row[grid_slice]
    return out


def _convolve_orders_arrays(
    initial_values: np.ndarray,
    fundamentals_values: np.ndarray,
    n_new: int,
    *,
    shift: int,
    spacing: float,
) -> np.ndarray:
    """Convolve repeatedly on plain arrays, choosing the path from ``shift``.

    Args:
        initial_values: Intensity densities on the initial grid.
        fundamentals_values: Intensity densities of the fundamentals.
        n_new: Number of higher orders to return.
        shift: Integer shift of the fundamentals' first bin, in bins. Negative
            when the fundamentals' first bin centre is negative.
        spacing: Bin width of the fundamentals.

    Returns:
        Array of shape ``(n_new, N)`` with one row per derived order.
    """
    initial_values = np.asarray(initial_values, dtype=float)
    fundamentals_values = np.asarray(fundamentals_values, dtype=float)
    if shift < 0:
        return _convolve_orders_padded(
            initial_values, fundamentals_values, n_new, shift=shift, spacing=spacing
        )
    return _convolve_orders_one_sided(
        initial_values, fundamentals_values, n_new, shift=shift, spacing=spacing
    )


def _resolve_initial_order(initial: Spectrum1D, initial_order: int | None) -> int:
    """Choose the initial quantum order from the argument or metadata.

    Args:
        initial: Initial spectrum.
        initial_order: Explicit order argument, or None.

    Returns:
        The initial quantum order.

    Raises:
        ValueError: If neither the argument nor the metadata gives an order.
    """
    meta_order = (initial.metadata or {}).get("quantum_order")
    if initial_order is None:
        if meta_order is None:
            raise ValueError(
                "Initial quantum order is unknown: pass initial_order or set "
                "'quantum_order' in the initial spectrum's metadata."
            )
        return int(meta_order)
    return int(initial_order)


def _units_per_fund_unit(initial_unit: str, fundamentals_unit: str) -> float:
    """Number of initial energy units in one fundamentals energy unit.

    Args:
        initial_unit: Energy unit of the initial spectrum.
        fundamentals_unit: Energy unit of the fundamentals.

    Returns:
        Conversion factor, so that ``x_f * factor`` is in the initial unit.

    Raises:
        ValueError: If the units cannot be converted into each other, or the
            conversion is not linear (for example wavelength to energy), since
            bin spacings and densities could not be rescaled.
    """
    if initial_unit == fundamentals_unit:
        return 1.0
    message = (
        "Energy units are incompatible: initial spectrum is in "
        f"{initial_unit}, fundamentals in {fundamentals_unit}."
    )
    try:
        factor = Quantity(1, fundamentals_unit).to(initial_unit).magnitude
        doubled = Quantity(2, fundamentals_unit).to(initial_unit).magnitude
    except TypeError as exc:
        raise ValueError(message) from exc
    if not np.isclose(doubled, 2 * factor, rtol=1e-12, atol=0.0):
        raise ValueError(
            f"{message} The conversion is not linear, so bin widths and densities "
            "cannot be rescaled between them."
        )
    return factor


def convolve_orders(
    initial: Spectrum1D,
    fundamentals: Spectrum1D,
    max_order: int = 10,
    *,
    initial_order: int | None = None,
    bin_position: str | None = None,
) -> Spectrum1DCollection:
    """Derive higher quantum-order spectra by repeated discrete convolution.

    Row k of the result is the initial spectrum convolved k times with the
    fundamentals (quantum order 1). The rows are labelled with quantum order
    ``initial_order + k``, share the initial spectrum's energy axis, and carry
    its other metadata. No scale factors or cross-section weights are applied;
    apply them to the returned collection.

    Convolving scalar spectra assumes the isotropic approximation. Mantid-Abins
    neglects negative-energy terms; this function includes them when the
    fundamentals grid starts at a negative bin centre. Energy zero must lie on
    a bin centre or bin edge of the fundamentals grid:

    - Zero on a bin centre: bins are located at their centres. No warning.
    - Zero on a bin edge: bins are located at their lower edges. Intensity
      of order n then sits (n - 1) / 2 bins above the centre of the bin it
      is reported in. A warning is issued unless ``bin_position`` is given.

    Example:
        Autoconvolution of an order-1 spectrum up to order 4, with zero at a
        bin centre::

            from abinslib.convolution import convolve_orders
            spectra = convolve_orders(order1, order1, max_order=4)
            weighted = apply_weights(spectra)

    Args:
        initial: Spectrum of quantum order ``initial_order``.
        fundamentals: Spectrum of quantum order 1 on the same bin spacing.
        max_order: Highest quantum order to return.
        initial_order: Quantum order of ``initial``. Defaults to its
            ``quantum_order`` metadata.
        bin_position: ``"centre"`` or ``"lower_edge"``, to state the bin
            reading explicitly. It must match the reading implied by where
            zero falls on the fundamentals grid.

    Returns:
        Collection of the derived orders, in increasing order.

    Raises:
        ValueError: If the initial order is unknown, ``max_order`` is not above
            it, the grids are irregular or differ in spacing, energy zero is
            not on a bin centre or edge, or ``bin_position`` does not match the
            grid.
        TypeError: If an input is not a Spectrum1D.
    """
    if not isinstance(initial, Spectrum1D) or not isinstance(fundamentals, Spectrum1D):
        raise TypeError("initial and fundamentals must both be Spectrum1D objects.")
    if bin_position is not None and bin_position not in _READINGS:
        raise ValueError(
            f"bin_position must be one of {_READINGS} or None, not {bin_position!r}."
        )

    order = _resolve_initial_order(initial, initial_order)
    initial_meta = dict(initial.metadata or {})
    meta_order = initial_meta.get("quantum_order")
    if initial_order is not None and meta_order is not None and meta_order != order:
        warnings.warn(
            f"initial_order={order} differs from the initial spectrum's "
            f"quantum_order metadata ({meta_order}); using initial_order.",
            UserWarning,
            stacklevel=2,
        )

    fund_order = (fundamentals.metadata or {}).get("quantum_order")
    if fund_order is not None and fund_order != 1:
        warnings.warn(
            f"Fundamentals carry quantum_order={fund_order}; they are treated "
            "as quantum order 1.",
            UserWarning,
            stacklevel=2,
        )

    if max_order <= order:
        raise ValueError(
            f"max_order ({max_order}) must be greater than the initial order ({order})."
        )
    n_new = max_order - order

    # Read each spectrum's arrays once; the same object is only read once.
    init_x, init_y = initial.x_data, initial.y_data
    n_init = init_y.magnitude.shape[-1]
    if fundamentals is initial:
        fund_x, fund_y = init_x, init_y
    else:
        fund_x, fund_y = fundamentals.x_data, fundamentals.y_data
    n_fund = fund_y.magnitude.shape[-1]

    init_grid = _grid_from_axis(init_x, n_init)
    if _same_axis(init_x, n_init, fund_x, n_fund):
        fund_grid = init_grid
    else:
        fund_grid = _grid_from_axis(fund_x, n_fund)

    # Express the fundamentals in the initial grid's energy unit. A density
    # per fundamentals unit becomes a density per initial unit by dividing by
    # the number of initial units in one fundamentals unit.
    units_per_fund_unit = _units_per_fund_unit(init_grid.unit, fund_grid.unit)
    fund_spacing = fund_grid.spacing * units_per_fund_unit
    if abs(fund_spacing - init_grid.spacing) > _REL_TOL * init_grid.spacing:
        raise ValueError(
            "Bin spacings differ: initial spectrum has "
            f"{init_grid.spacing:.6g} {init_grid.unit}, fundamentals have "
            f"{fund_spacing:.6g} {init_grid.unit}."
        )

    placement = _zero_placement(fund_grid)
    if bin_position is not None and bin_position != placement.reading:
        raise ValueError(
            f"bin_position={bin_position!r} does not match the fundamentals grid, "
            f"where energy zero implies the {placement.reading!r} reading."
        )
    if placement.reading == "lower_edge" and bin_position is None:
        warnings.warn(
            "Energy zero lies on a bin edge of the fundamentals grid, so bins are "
            "located at their lower edges (Mantid-Abins convention). Each order is "
            "quantised accordingly. Pass bin_position='lower_edge' to silence this "
            "warning, or see the convolution documentation.",
            UserWarning,
            stacklevel=2,
        )

    fund_units_in_initial_energy = fund_y.units * fund_x.units / init_x.units
    out_units = init_y.units * fund_units_in_initial_energy * init_x.units

    values = _convolve_orders_arrays(
        init_y.magnitude,
        fund_y.magnitude / units_per_fund_unit,
        n_new,
        shift=placement.shift,
        spacing=init_grid.spacing,
    )

    line_data = [
        {key: value for key, value in initial_meta.items() if key != "quantum_order"}
        | {"quantum_order": order + k}
        for k in range(1, n_new + 1)
    ]
    return Spectrum1DCollection(
        x_data=init_x,
        y_data=Quantity(values, out_units),
        metadata={"line_data": line_data},
    )
