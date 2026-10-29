"""Derive higher quantum-order spectra by discrete convolution.

Repeated convolution of a binned order-n spectrum with a binned order-1
(fundamentals) spectrum gives the order-(n + k) spectrum. The operation here
applies only the convolution and the bookkeeping that goes with it: grid
validation, placement, density units and metadata. Scale factors and
cross-section weights are applied afterwards by the caller.

Approximations and conventions:

- Convolving scalar spectra assumes the isotropic approximation, in which the
  orientation-averaged scattering factorises into a scalar product of
  per-order spectra.
- Mantid-Abins neglects the negative-energy (gain) terms. The padded
  two-sided path in this module includes them when the fundamentals grid
  starts at a negative bin centre.
- Sample positions are added as given. Energy zero must be a bin centre
  (sample position) of the fundamentals grid, so that convolved positions land
  on the output grid, and an error is raised otherwise. A grid built with
  *edges* at zero, such as ``numpy.linspace(0, e_max, n + 1)`` is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from euphonic import Quantity, ureg
from euphonic.spectra import Spectrum1D, Spectrum1DCollection
import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

_REL_TOL = 1e-6


@dataclass(frozen=True)
class _GridInfo:
    """Regular energy grid of a spectrum.

    Attributes:
        spacing: Bin width, in ``unit``.
        first_centre: Position of the first bin centre (sample position), in
            ``unit``.
        unit: Energy unit of the axis.
    """

    spacing: float
    first_centre: float
    unit: ureg.Unit


def _grid_from_axis(x_data: Quantity, n_values: int) -> _GridInfo:
    """Read an energy axis once and check that it is regular.

    Args:
        x_data: Energy values, as bin edges (``n_values + 1``) or bin centres
            (``n_values``).
        n_values: Number of intensity values on the axis.

    Returns:
        Grid description in the axis' own energy unit.

    Raises:
        ValueError: If the axis has fewer than two values, is not increasing,
            or has irregular bins.
    """
    x = np.asarray(x_data.magnitude, dtype=float)
    unit = x_data.units

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

    first_centre = float(x[0]) + spacing / 2 if is_edges else float(x[0])
    return _GridInfo(spacing=spacing, first_centre=first_centre, unit=unit)


def _same_axis(a: Quantity, n_a: int, b: Quantity, n_b: int) -> bool:
    """True if two energy axes and intensity lengths are identical."""
    if n_a != n_b:
        return False
    if a is b:
        return True
    return bool(np.array_equal(a.magnitude, b.magnitude) and a.units == b.units)


def _zero_placement(grid: _GridInfo) -> int:
    """Find the integer shift of the grid's first sample position.

    Args:
        grid: Regular grid of the fundamentals.

    Returns:
        The position of the first sample, in units of the spacing.

    Raises:
        ValueError: If energy zero is not on a bin centre. The message names
            the remedy when zero is on a bin edge.
    """
    ratio = grid.first_centre / grid.spacing
    shift = round(ratio)
    offset = abs(ratio - shift)
    if offset <= _REL_TOL:
        return shift
    if abs(offset - 0.5) <= _REL_TOL:
        raise ValueError(
            "Energy zero lies on a bin edge of the fundamentals grid, but "
            "convolution requires it on a bin centre (sample position). If the "
            "x values are bin edges starting at zero, as in Mantid-Abins, pass "
            "the lower edges as N sampling points instead."
        )
    raise ValueError(
        f"Energy zero lies {offset:.4g} bin spacings ({offset * grid.spacing:.4g} "
        f"{grid.unit}) from the nearest bin centre of the fundamentals grid. "
        "Convolution requires energy zero on a bin centre (sample position)."
    )


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
    start, stop = max(0, shift), min(length, shift + values.size)
    if stop > start:
        out[start:stop] = values[start - shift : stop - shift]
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
    assert shift >= 0, "One-sided convolution requires a non-negative shift."
    n = initial_values.size
    out = np.zeros((n_new, n))
    kernel_length = min(fundamentals_values.size, n - shift)
    if kernel_length <= 0:
        return out

    kernel = fundamentals_values[:kernel_length] * spacing
    n_fft = next_fast_len(n + kernel_length - 1, real=True)
    kernel_fft = rfft(kernel, n_fft)

    row = initial_values
    for k in range(n_new):
        conv = irfft(rfft(row, n_fft) * kernel_fft, n_fft)
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

    n_fft = next_fast_len(width + length - 1, real=True)
    kernel_fft = rfft(kernel, n_fft)

    row = np.zeros(width)
    row[grid_slice] = initial_values

    out = np.zeros((n_new, n))
    for k in range(n_new):
        conv = irfft(rfft(row, n_fft) * kernel_fft, n_fft)[: width + length - 1]
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


def _units_per_fund_unit(
    initial_unit: ureg.Unit, fundamentals_unit: ureg.Unit
) -> float:
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
) -> Spectrum1DCollection:
    """Derive higher quantum-order spectra by repeated discrete convolution.

    Row k of the result is the initial spectrum convolved k times with the
    fundamentals (quantum order 1), labelled with quantum order
    ``initial_order + k``. No scale factors or cross-section weights are
    applied; apply them to the returned collection.

    To avoid drift, the energy zero must be a bin centre on the
    sampling lattice of fundamentals; otherwise an error is raised.

    If negative energies are included in ``fundamentals``, this
    implementation will use additional padding/memory to accommodate
    the resulting combination spectra; either way, the results are
    cropped to the range of ``initial``.

    Example:
        Autoconvolution of an order-1 spectrum up to order 4, with a bin
        centre at zero::

            from abinslib.convolution import convolve_orders
            spectra = convolve_orders(order1, order1, max_order=4)
            weighted = apply_weights(spectra)

        For spectra on bin edges starting at zero, see :doc:`/convolution`.

    Args:
        initial: Spectrum of quantum order ``initial_order``.
        fundamentals: Spectrum of quantum order 1 on the same bin spacing, as
            an unweighted intensity density per unit energy (for example in
            1 / meV), so that the output keeps the units of ``initial``.
        max_order: Highest quantum order to return.
        initial_order: Quantum order of ``initial``. Defaults to its
            ``quantum_order`` metadata.

    Returns:
        Collection of the derived orders, in increasing order, with the
        ``initial`` energy bins and units. ``quantum_order`` metadata
        is set appropriately for each line; other metadata is copied.

    Raises:
        ValueError: If the initial order is unknown, ``max_order`` is not above
            it, the grids are irregular or differ in spacing, or energy zero is
            not on a bin centre of the fundamentals lattice.
        TypeError: If an input is not a Spectrum1D.
    """
    if not isinstance(initial, Spectrum1D) or not isinstance(fundamentals, Spectrum1D):
        raise TypeError("initial and fundamentals must both be Spectrum1D objects.")

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

    shift = _zero_placement(fund_grid)

    fund_units_in_initial_energy = fund_y.units * fund_x.units / init_x.units
    out_units = init_y.units * fund_units_in_initial_energy * init_x.units

    values = _convolve_orders_arrays(
        init_y.magnitude,
        fund_y.magnitude / units_per_fund_unit,
        n_new,
        shift=shift,
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
