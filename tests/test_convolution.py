"""Unit tests for abinslib.convolution"""

import itertools
import re
import warnings

from euphonic import Quantity
from euphonic.spectra import Spectrum1D, Spectrum1DCollection
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.signal import convolve

from abinslib.convolution import (
    _convolve_orders_arrays,
    _convolve_orders_one_sided,
    _convolve_orders_padded,
    _grid_from_axis,
    _zero_placement,
    convolve_orders,
)
from abinslib.util import apply_weights

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spectrum(x, y, unit="meV", metadata=None, y_unit="1/meV"):
    return Spectrum1D(
        Quantity(np.asarray(x, dtype=float), unit),
        Quantity(np.asarray(y, dtype=float), y_unit),
        metadata=metadata,
    )


def _build_grid(spectrum):
    """Grid of a spectrum, as read by convolve_orders."""
    return _grid_from_axis(spectrum.x_data, spectrum.y_data.magnitude.shape[-1])


def _centres(first_index, n, d):
    """Bin centres at (first_index + i) * d."""
    return (first_index + np.arange(n)) * d


def _edges(first_index, n, d):
    """Bin edges at (first_index + i) * d, for n bins."""
    return (first_index + np.arange(n + 1)) * d


def _order1(n=12, d=0.5, first=0):
    rng = np.random.default_rng(99)
    return _spectrum(
        _centres(first, n, d), rng.random(n), metadata={"quantum_order": 1}
    )


def _crop(full, origin, n):
    """Values of ``full`` at output bins 0..n-1 when full[0] sits at bin ``origin``."""
    out = np.zeros(n)
    for m in range(n):
        c = m - origin
        if 0 <= c < full.size:
            out[m] = full[c]
    return out


def _scipy_reference(initial, fundamentals, n_new, shift, spacing, crop_each=False):
    """Repeated full scipy convolution of the initial values with the fundamentals.

    By default there is no intermediate crop: each order moves its result by
    the integer shift, tracked as an origin, and the array is cropped once at
    the end. ``crop_each`` crops to the grid after every order instead, as
    Mantid does, which is only exact when no intensity can return from outside
    the grid (non-negative shift).
    """
    n = len(initial)
    kernel = np.asarray(fundamentals, dtype=float) * spacing
    current = np.asarray(initial, dtype=float)
    if crop_each:
        for _ in range(n_new):
            current = _crop(convolve(current, kernel, mode="full"), shift, n)
        return current
    origin = 0
    for _ in range(n_new):
        current = convolve(current, kernel, mode="full")
        origin += shift
    return _crop(current, origin, n)


# ---------------------------------------------------------------------------
# Grid reading and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "n_values", "is_edges", "first_edge", "edge_offset"),
    [
        pytest.param(_edges(0, 5, 0.5), 5, True, 0.0, 0.0, id="edges"),
        # Centres at 0, d, 2d... put the first edge at -d/2: zero is on a centre
        pytest.param(_centres(0, 5, 0.5), 5, False, -0.25, 0.5, id="centres"),
    ],
)
def test_build_grid(x, n_values, is_edges, first_edge, edge_offset):
    grid = _build_grid(_spectrum(x, np.ones(n_values)))
    assert bool(grid.is_edges) is is_edges
    assert grid.length == n_values
    assert grid.unit == "millielectron_volt"
    assert_allclose(grid.spacing, 0.5)
    assert_allclose(grid.first_edge, first_edge, atol=1e-12)
    assert_allclose(grid.edge_offset, edge_offset, atol=1e-12)
    # A second spectrum on an equal axis gives the same grid information
    assert _build_grid(_spectrum(x.copy(), np.ones(n_values))) == grid


# Scenario -> (axis, number of values, expected reading, expected shift).
# Edges at (k + 0.5) d have a bin centre at zero; centres at (k + 0.5) d have an
# edge at zero.
_HALF = (np.arange(7) - 3 + 0.5) * 0.5
_PLACEMENT_CASES = {
    "Bin edges include zero": (_edges(-2, 6, 1.0), 6, "lower_edge", -2),
    "Bin centres at zero given as edges": (_HALF, 6, "centre", -2),
    "Sampling points include zero": (_centres(-3, 7, 0.5), 7, "centre", -3),
    "Sampling points offset by half a spacing": (_HALF, 7, "lower_edge", -3),
    "Grid range excludes zero": (_edges(10, 4, 0.5), 4, "lower_edge", 10),
}


@pytest.mark.parametrize(
    ("x", "n_values", "reading", "shift"),
    _PLACEMENT_CASES.values(),
    ids=_PLACEMENT_CASES.keys(),
)
def test_zero_placement(x, n_values, reading, shift):
    placement = _zero_placement(_build_grid(_spectrum(x, np.ones(n_values))))
    assert placement.reading == reading
    assert placement.shift == shift


def _pair(init_x, fund_x, init_unit="meV", fund_unit="meV", n=6):
    return (
        _spectrum(init_x, np.ones(n), unit=init_unit),
        _spectrum(fund_x, np.ones(n), unit=fund_unit),
    )


_EDGE_PAIR = (_edges(0, 6, 1.0), _edges(0, 6, 1.0))
_CENTRE_PAIR = (_centres(-2, 6, 1.0), _centres(-2, 6, 1.0))


@pytest.mark.parametrize(
    ("call", "exception", "match"),
    [
        pytest.param(
            lambda: _build_grid(_spectrum([0.0, 1.0, 2.5, 3.0], np.ones(3))),
            ValueError,
            "irregular",
            id="Irregular grid",
        ),
        pytest.param(
            lambda: _zero_placement(
                _build_grid(_spectrum(_edges(0, 6, 1.0) + 0.25, np.ones(6)))
            ),
            ValueError,
            "0.25 bin spacings",
            id="Zero not on a centre or edge",
        ),
        pytest.param(
            lambda: convolve_orders(
                *_pair(_centres(0, 6, 1.0), _centres(0, 6, 0.5)),
                max_order=3,
                initial_order=1,
            ),
            ValueError,
            r"Bin spacings differ.*1 millielectron_volt.*0\.5 millielectron_volt",
            id="Different spacing",
        ),
        pytest.param(
            lambda: convolve_orders(
                *_pair(
                    _centres(0, 4, 1.0), _centres(0, 4, 1.0), fund_unit="kelvin", n=4
                ),
                max_order=3,
                initial_order=1,
            ),
            ValueError,
            "incompatible",
            id="Incompatible energy units",
        ),
        pytest.param(
            # Wavelength to energy is not a linear rescaling
            lambda: convolve_orders(
                *_pair(
                    _centres(0, 4, 1.0), _centres(0, 4, 1.0), fund_unit="angstrom", n=4
                ),
                max_order=3,
                initial_order=1,
            ),
            ValueError,
            "not linear",
            id="Non-linear energy conversion",
        ),
        pytest.param(
            lambda: convolve_orders(
                _order1(), _order1(), max_order=1, bin_position="centre"
            ),
            ValueError,
            "greater than the initial order",
            id="Maximum order not above the initial order",
        ),
        pytest.param(
            lambda: convolve_orders(
                _spectrum(_centres(0, 12, 0.5), np.ones(12)),
                _order1(),
                max_order=3,
                bin_position="centre",
            ),
            ValueError,
            "Initial quantum order is unknown",
            id="No initial order available",
        ),
        pytest.param(
            lambda: convolve_orders(
                Spectrum1DCollection(
                    x_data=_order1().x_data, y_data=Quantity(np.ones((2, 12)), "1/meV")
                ),
                _order1(),
                max_order=3,
                bin_position="centre",
            ),
            TypeError,
            None,
            id="Collection instead of a spectrum",
        ),
        pytest.param(
            lambda: convolve_orders(
                *_pair(*_EDGE_PAIR),
                max_order=3,
                initial_order=1,
                bin_position="centre",
            ),
            ValueError,
            "does not match",
            id="Explicit centre reading on an edge grid",
        ),
        pytest.param(
            lambda: convolve_orders(
                *_pair(*_CENTRE_PAIR),
                max_order=3,
                initial_order=1,
                bin_position="lower_edge",
            ),
            ValueError,
            "does not match",
            id="Explicit lower-edge reading on a centre grid",
        ),
    ],
)
def test_invalid_inputs_raise(call, exception, match):
    with pytest.raises(exception, match=match):
        call()


@pytest.mark.parametrize(
    ("bin_position", "warns"),
    [
        pytest.param(None, True, id="Default reading warns"),
        pytest.param("lower_edge", False, id="Explicit reading matches"),
    ],
)
def test_lower_edge_reading_warning(bin_position, warns):
    init, fund = _pair(*_EDGE_PAIR)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        convolve_orders(
            init, fund, max_order=3, initial_order=1, bin_position=bin_position
        )
    messages = [w for w in record if "lower edges" in str(w.message)]
    assert bool(messages) is warns
    if warns:
        assert messages[0].category is UserWarning
        assert messages[0].filename == __file__


def test_convertible_units_with_matching_spacing_proceed():
    cm_per_mev = Quantity(1, "meV").to("cm^-1").magnitude
    rng = np.random.default_rng(8)
    y_mev = rng.random(7)
    # Same physical grid and intensities, initial spectrum written in cm^-1
    ref_init = _spectrum(_centres(-3, 7, 1.0), y_mev, metadata={"quantum_order": 1})
    ref_fund = _spectrum(_centres(-3, 7, 1.0), y_mev, metadata={"quantum_order": 1})
    cm_init = _spectrum(
        _centres(-3, 7, cm_per_mev),
        y_mev / cm_per_mev,
        unit="cm^-1",
        y_unit="1/cm^-1",
        metadata={"quantum_order": 1},
    )
    ref = convolve_orders(ref_init, ref_fund, max_order=3, bin_position="centre")
    out = convolve_orders(cm_init, ref_fund, max_order=3, bin_position="centre")
    # Densities per cm^-1 are densities per meV divided by cm^-1 per meV
    assert_allclose(
        out.y_data.magnitude,
        ref.y_data.magnitude / cm_per_mev,
        rtol=1e-12,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Array routine: agreement with repeated scipy convolution
# ---------------------------------------------------------------------------

_SHIFTS = [-4, -2, -1, 0, 1, 3, 7]
_ROUTINES = {
    "arrays": _convolve_orders_arrays,  # dispatches on the sign of the shift
    "one_sided": _convolve_orders_one_sided,
    "padded": _convolve_orders_padded,
}
_ROUTINE_CASES = [
    pytest.param(name, shift, id=f"{name}-shift{shift}")
    for name in _ROUTINES
    for shift in _SHIFTS
    if not (name == "one_sided" and shift < 0)
]


@pytest.mark.parametrize(("routine", "shift"), _ROUTINE_CASES)
def test_routine_matches_repeated_scipy_convolution(routine, shift):
    rng = np.random.default_rng(1000 + shift)
    init = rng.random(40)
    fund = rng.random(25)
    spacing = 0.37
    got = _ROUTINES[routine](init, fund, 4, shift=shift, spacing=spacing)

    orders = range(1, 5)
    expected = np.array(
        [_scipy_reference(init, fund, k, shift, spacing) for k in orders]
    )
    atol = 1e-12 * np.abs(expected).max()
    assert_allclose(got, expected, rtol=0, atol=atol)
    if shift >= 0:
        # Mantid's method, cropping after every order, gives the same result
        cropped = np.array(
            [_scipy_reference(init, fund, k, shift, spacing, True) for k in orders]
        )
        assert_allclose(got, cropped, rtol=0, atol=atol)


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------


def _exact_pairwise_orders(initial_by_pos, fund_by_pos, n_new):
    """Exact sums of positions for delta features (positions are integers)."""
    result = []
    current = dict(initial_by_pos)
    for _ in range(n_new):
        nxt = {}
        for p, a in current.items():
            for q, b in fund_by_pos.items():
                nxt[p + q] = nxt.get(p + q, 0.0) + a * b
        result.append(nxt)
        current = nxt
    return result


def test_features_on_bin_positions_match_exact_pairwise_sums():
    d = 0.5
    # Centre grid: zero on a bin centre; positions are integer multiples of d
    positions = {0: 1.0, 2: 0.6, 3: 0.3}
    fund_pos = {0: 0.9, 1: 0.4, 3: 0.2}
    index = np.arange(-10, 12)
    init_y = np.zeros(index.size)
    fund_y = np.zeros(index.size)
    for p, a in positions.items():
        init_y[np.searchsorted(index, p)] = a / d
    for p, a in fund_pos.items():
        fund_y[np.searchsorted(index, p)] = a / d
    fund_centres = _centres(index[0], index.size, d)
    init = _spectrum(fund_centres, init_y)
    fund = _spectrum(fund_centres, fund_y)
    out = convolve_orders(init, fund, max_order=5, initial_order=1)
    exact = _exact_pairwise_orders(positions, fund_pos, 4)
    for k, ref in enumerate(exact):
        row = out.y_data.magnitude[k]
        for p, value in ref.items():
            if index[0] <= p <= index[-1]:
                i = np.searchsorted(index, p)
                assert_allclose(row[i], value / d, rtol=1e-12, atol=1e-12)
        covered = {np.searchsorted(index, p) for p in ref if p in index}
        outside = [i for i in range(index.size) if i not in covered]
        assert_allclose(row[outside], 0.0, atol=1e-12)


# Box-shaped features at arbitrary positions. The binned representation
# (cell averages) converges with spacing; the analytic reference is the exact
# box convolution, cell-averaged onto the same bins.
_BOXES = ((1.13, 1.5, 1.0), (2.71, 0.6, 0.5))  # (start, length, height)


def _box_autoconvolution(x):
    """Exact density of the autoconvolution of the boxes at points x."""
    out = np.zeros_like(x)
    for pa, la, ha in _BOXES:
        for pb, lb, hb in _BOXES:
            lo = np.maximum(pa, x - pb - lb)
            hi = np.minimum(pa + la, x - pb)
            out += ha * hb * np.clip(hi - lo, 0.0, None)
    return out


def _box_cell_average(centres, d):
    """Exact cell average of the box density over bins of width d."""
    out = np.zeros(centres.size)
    for pa, la, ha in _BOXES:
        lo = np.maximum(centres - d / 2, pa)
        hi = np.minimum(centres + d / 2, pa + la)
        out += ha * np.clip(hi - lo, 0.0, None) / d
    return out


def _cell_average(fun, centres, d, samples=400):
    offsets = (np.arange(samples) + 0.5) / samples * d - d / 2
    return np.array([fun(c + offsets).mean() for c in centres])


def test_arbitrary_positions_error_falls_on_halving():
    errors = []
    for halvings in range(5):
        d = 0.8 / 2**halvings
        centres = np.arange(-2, int(np.ceil(10 / d))) * d
        density = _box_cell_average(centres, d)
        spec = _spectrum(centres, density, metadata={"quantum_order": 1})
        out = convolve_orders(spec, spec, max_order=2, bin_position="centre")
        reference = _cell_average(_box_autoconvolution, centres, d)
        errors.append(np.abs(out.y_data.magnitude[0] - reference).sum() * d)
    assert all(later < earlier for earlier, later in itertools.pairwise(errors))
    # Each halving reduces the error by roughly a factor of 2 or more
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    assert np.all(ratios > 1.5)


def test_output_density_bin_width_applied_once_per_order():
    # Unit-density box of length L = 20 bins: the order-2 density peak is L
    d = 0.1
    n_box = 20
    index = np.arange(-30, 50)
    box = np.where((index >= 0) & (index < n_box), 1.0, 0.0)
    centres = index * d
    init = _spectrum(centres, box)
    fund = _spectrum(centres, box)
    out = convolve_orders(
        init, fund, max_order=2, initial_order=1, bin_position="centre"
    )
    assert_allclose(out.y_data.magnitude[0].max(), n_box * d, rtol=1e-12)


def test_no_ghost_intensity_from_upper_edge():
    n = 60
    d = 0.5
    init_y = np.zeros(n)
    init_y[-3:] = 1.0 / d
    fund_y = np.zeros(n)
    fund_y[:5] = 1.0 / d
    values = _convolve_orders_arrays(init_y, fund_y, 5, shift=0, spacing=d)
    peak = np.abs(values).max()
    assert_allclose(values[:, :10], 0.0, atol=1e-12 * peak)


# ---------------------------------------------------------------------------
# Two-sided scenarios
# ---------------------------------------------------------------------------


def _enumerate_paths(start, steps, n, step_offsets, step_weights, spacing):
    """Sum weights of all paths of a given length that end on the grid."""
    out = np.zeros(n)
    for path in itertools.product(range(len(step_offsets)), repeat=steps):
        index = start
        weight = 1.0
        for j in path:
            index += step_offsets[j]
            weight *= step_weights[j] * spacing
        if 0 <= index < n:
            out[index] += weight
    return out


def test_excursion_above_grid_returns_through_negative_term():
    # Fundamentals: intensity at bin offsets -1 and +2 from the initial bin.
    # Starting at index 4 on a 6-bin grid, one +2 step leaves the grid (to 6),
    # and a following -1 step returns inside it (to 5).
    d = 0.5
    n = 6
    shift = -1
    fund = np.array([1.0, 0.0, 0.0, 1.0]) / d
    init = np.zeros(n)
    init[4] = 1.0 / d
    values = _convolve_orders_arrays(init, fund, 2, shift=shift, spacing=d)

    # Each step: offset shift + j for fundamentals bin j, weight density * d
    offsets = [shift + 0, shift + 3]
    weights = [1.0 / d, 1.0 / d]
    initial_density = 1.0 / d
    expected = np.array(
        [
            _enumerate_paths(4, k, n, offsets, weights, d) * initial_density
            for k in (1, 2)
        ]
    )
    assert_allclose(values, expected, rtol=0, atol=1e-12)
    # The order-3 path through the excursion (+2, then -1) contributes to bin 5
    assert values[1, 5] > 0.0


def test_two_sided_initial_on_negative_grid_with_nonnegative_fundamentals():
    d = 0.5
    init_centres = np.arange(-4, 4) * d
    init_y = np.zeros(init_centres.size)
    idx_minus_one = int(np.searchsorted(init_centres, -1.0))
    init_y[idx_minus_one] = 1.0 / d
    fund_centres = np.array([0.0, 0.5, 1.0])
    fund_y = np.array([1.0, 1.0, 1.0]) / d
    init = _spectrum(init_centres, init_y)
    fund = _spectrum(fund_centres, fund_y)
    out = convolve_orders(
        init, fund, max_order=3, initial_order=1, bin_position="centre"
    )
    expected = _convolve_orders_one_sided(init_y, fund_y, 2, shift=0, spacing=d)
    assert_allclose(
        out.y_data.magnitude, expected, rtol=0, atol=1e-12 * np.abs(expected).max()
    )
    # Intensity at -1 is displaced upward by the fundamentals at 0 (weight 1/d * d)
    assert_allclose(out.y_data.magnitude[0, idx_minus_one], 1.0 / d, rtol=1e-12)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def _meta(order):
    return None if order is None else {"quantum_order": order}


# Scenario -> (initial order metadata, initial_order argument, fundamentals order
# metadata, extra keyword arguments, expected row orders, expected warning).
_FUNDAMENTALS_WARNING = "treated as quantum order 1"
_ORDER_CASES = {
    "Default maximum order": (1, None, 1, {}, list(range(2, 11)), None),
    "Autoconvolution of order 1": (1, None, 1, {"max_order": 4}, [2, 3, 4], None),
    "Starting from a higher order": (2, None, 1, {"max_order": 5}, [3, 4, 5], None),
    "Initial order from metadata": (1, None, 1, {"max_order": 3}, [2, 3], None),
    "Explicit argument": (None, 2, 1, {"max_order": 4}, [3, 4], None),
    "Argument contradicts metadata": (1, 2, 1, {"max_order": 4}, [3, 4], "differs"),
    "Fundamentals labelled with another order": (
        1,
        None,
        2,
        {"max_order": 3},
        [2, 3],
        _FUNDAMENTALS_WARNING,
    ),
}


@pytest.mark.parametrize(
    ("initial_meta", "initial_order", "fund_meta", "max_kwargs", "expected", "warning"),
    _ORDER_CASES.values(),
    ids=_ORDER_CASES.keys(),
)
def test_row_orders_and_warnings(
    initial_meta, initial_order, fund_meta, max_kwargs, expected, warning
):
    y = _order1().y_data.magnitude
    init = _spectrum(_centres(0, 12, 0.5), y, metadata=_meta(initial_meta))
    fund = _spectrum(_centres(0, 12, 0.5), y, metadata=_meta(fund_meta))
    kwargs = {"bin_position": "centre", **max_kwargs}
    if initial_order is not None:
        kwargs["initial_order"] = initial_order

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = convolve_orders(init, fund, **kwargs)

    assert [m["quantum_order"] for m in out.iter_metadata()] == expected
    assert out.y_data.magnitude.shape == (len(expected), 12)
    if warning is None:
        assert not record
    else:
        assert len(record) == 1
        assert re.search(warning, str(record[0].message))
        assert record[0].filename == __file__


@pytest.mark.parametrize(
    ("n", "d", "max_order"),
    [pytest.param(12, 0.5, 3, id="Order 3"), pytest.param(10, 1.0, 4, id="Order 4")],
)
def test_no_scaling_matches_repeated_convolution(n, d, max_order):
    spec = _order1(n=n, d=d)
    out = convolve_orders(spec, spec, max_order=max_order, bin_position="centre")
    y1 = spec.y_data.magnitude
    current = y1.copy()
    for k in range(max_order - 1):
        current = convolve(current, y1 * d, mode="full")[:n]
        assert_allclose(out.y_data.magnitude[k], current, rtol=1e-12, atol=1e-12)


def test_centre_at_zero_matches_lower_edge_reading_on_shifted_grid():
    n = 10
    d = 0.5
    rng = np.random.default_rng(21)
    y = rng.random(n)
    # Centres at k d (zero on a centre) with k from -5 to 4
    centre_init = _spectrum(_centres(-5, n, d), y, metadata={"quantum_order": 1})
    centre_fund = _spectrum(_centres(-5, n, d), y, metadata={"quantum_order": 1})
    centre_out = convolve_orders(
        centre_init, centre_fund, max_order=4, bin_position="centre"
    )
    # Edges at k d (zero on an edge), k from -5 to 5: the same bin indices
    # with lower edges at the centre positions above.
    edge_init = _spectrum(_edges(-5, n, d), y, metadata={"quantum_order": 1})
    edge_fund = _spectrum(_edges(-5, n, d), y, metadata={"quantum_order": 1})
    with pytest.warns(UserWarning, match="lower edges"):
        edge_out = convolve_orders(edge_init, edge_fund, max_order=4)
    assert_allclose(centre_out.y_data.magnitude, edge_out.y_data.magnitude, atol=1e-12)


def test_atom_metadata_is_carried_and_weights_apply():
    atom = {"atom_symbol": "H", "mass": 1.00794, "atom_index": 0}
    spec = _spectrum(
        _centres(0, 12, 0.5),
        np.random.default_rng(3).random(12),
        metadata={**atom, "quantum_order": 1},
    )
    out = convolve_orders(spec, spec, max_order=4, bin_position="centre")
    for item in out.iter_metadata():
        for key, value in atom.items():
            assert item[key] == value
    weighted = apply_weights(out)
    assert weighted.y_data.magnitude.shape == out.y_data.magnitude.shape
    assert not np.allclose(weighted.y_data.magnitude, out.y_data.magnitude)
