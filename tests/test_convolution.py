"""Unit tests for abinslib.convolution"""

import itertools

from euphonic import Quantity
from euphonic.spectra import Spectrum1D, Spectrum1DCollection
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.signal import convolve

from abinslib.convolution import (
    _convolve_orders_arrays,
    _convolve_orders_one_sided,
    _convolve_orders_padded,
    _grid_from_axis,
    _shift_crop,
    _zero_placement,
    convolve_orders,
)
from abinslib.util import apply_weights


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


def _scipy_reference(initial, fundamentals, n_new, shift, spacing):
    """Repeated full scipy convolution, no intermediate crop, one final crop.

    Each order moves its result by the integer shift, which is tracked as an
    origin; the final array is cropped once to the initial grid.
    """
    full = np.asarray(initial, dtype=float)
    kernel = np.asarray(fundamentals, dtype=float) * spacing
    origin = 0
    for _ in range(n_new):
        full = convolve(full, kernel, mode="full")
        origin += shift
    n = len(initial)
    out = np.zeros(n)
    for m in range(n):
        c = m - origin
        if 0 <= c < full.size:
            out[m] = full[c]
    return out


def _brute_padded_reference(initial, fundamentals, n_new, shift, spacing):
    """Repeated convolution on a wide window, carried out with scipy."""
    n = len(initial)
    length = len(fundamentals)
    pad = (n_new + 1) * (abs(shift) + length + 1)
    grid = np.zeros(n + 2 * pad)
    grid[pad : pad + n] = initial
    kernel = np.asarray(fundamentals, dtype=float) * spacing
    rows = []
    for _ in range(n_new):
        conv = convolve(grid, kernel, mode="full")
        moved = np.zeros(grid.size)
        for q in range(grid.size):
            c = q - shift
            if 0 <= c < conv.size:
                moved[q] = conv[c]
        grid = moved
        rows.append(grid[pad : pad + n].copy())
    return np.array(rows)


# ---------------------------------------------------------------------------
# 1.1 Grid builder
# ---------------------------------------------------------------------------


def test_build_grid_edges():
    spec = _spectrum(_edges(0, 5, 0.5), np.ones(5))
    grid = _build_grid(spec)
    assert grid.is_edges
    assert grid.length == 5
    assert_allclose(grid.spacing, 0.5)
    assert_allclose(grid.first_edge, 0.0)
    assert grid.unit == "millielectron_volt"
    assert_allclose(grid.edge_offset, 0.0, atol=1e-12)


def test_build_grid_centres_phase():
    # Centres at 0, d, 2d..., so the first edge is at -d/2 and zero is on a centre
    spec = _spectrum(_centres(0, 5, 0.5), np.ones(5))
    grid = _build_grid(spec)
    assert not grid.is_edges
    assert_allclose(grid.spacing, 0.5)
    assert_allclose(grid.first_edge, -0.25)
    assert_allclose(grid.edge_offset, 0.5, atol=1e-12)


def test_build_grid_irregular_names_irregularity():
    x = np.array([0.0, 1.0, 2.5, 3.0])
    spec = _spectrum(x, np.ones(3))
    with pytest.raises(ValueError, match="irregular"):
        _build_grid(spec)


# ---------------------------------------------------------------------------
# 1.2 Consistency checks
# ---------------------------------------------------------------------------


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


def test_different_spacing_raises_naming_both():
    init = _spectrum(_centres(0, 6, 1.0), np.ones(6))
    fund = _spectrum(_centres(0, 6, 0.5), np.ones(6))
    with pytest.raises(ValueError, match="Bin spacings differ") as info:
        convolve_orders(init, fund, max_order=3, initial_order=1)
    message = str(info.value)
    assert "1" in message
    assert "0.5" in message


def test_shared_grid_same_info_as_equal_grids():
    x = _centres(-4, 9, 0.25)
    same = _spectrum(x, np.arange(9.0))
    shared = _spectrum(x, np.arange(9.0))
    assert _build_grid(same) == _build_grid(shared)
    assert_array_equal(same.x_data.magnitude, shared.x_data.magnitude)


def test_incompatible_energy_units_raise():
    init = _spectrum(_centres(0, 4, 1.0), np.ones(4), unit="meV")
    fund = _spectrum(_centres(0, 4, 1.0), np.ones(4), unit="kelvin")
    with pytest.raises(ValueError, match="incompatible"):
        convolve_orders(init, fund, max_order=3, initial_order=1)


def test_non_linear_energy_conversion_raises():
    # Wavelength to energy is not a linear rescaling
    init = _spectrum(_centres(0, 4, 1.0), np.ones(4), unit="meV")
    fund = _spectrum(_centres(0, 4, 1.0), np.ones(4), unit="angstrom")
    with pytest.raises(ValueError, match="not linear"):
        convolve_orders(init, fund, max_order=3, initial_order=1)


# ---------------------------------------------------------------------------
# 1.3 Zero alignment and bin reading
# ---------------------------------------------------------------------------


def test_edges_with_one_at_zero_lower_edge_reading():
    grid = _build_grid(_spectrum(_edges(-2, 6, 1.0), np.ones(6)))
    placement = _zero_placement(grid)
    assert placement.reading == "lower_edge"
    assert placement.shift == -2


def test_edges_at_half_spacing_centre_reading():
    # Edges at (k + 0.5) d: bin [-0.5 d, 0.5 d] has its centre at zero
    edges = (np.arange(7) - 3 + 0.5) * 0.5
    grid = _build_grid(_spectrum(edges, np.ones(6)))
    placement = _zero_placement(grid)
    assert placement.reading == "centre"
    # The first bin is centred at -1.0 = -2 spacings
    assert placement.shift == -2


def test_points_including_zero_centre_reading():
    grid = _build_grid(_spectrum(_centres(-3, 7, 0.5), np.ones(7)))
    placement = _zero_placement(grid)
    assert placement.reading == "centre"
    assert placement.shift == -3


def test_points_offset_by_half_spacing_lower_edge_reading():
    centres = (np.arange(7) - 3 + 0.5) * 0.5
    grid = _build_grid(_spectrum(centres, np.ones(7)))
    placement = _zero_placement(grid)
    assert placement.reading == "lower_edge"
    assert placement.shift == -3


def test_zero_off_lattice_raises_with_offset():
    grid = _build_grid(_spectrum(_edges(0, 6, 1.0) + 0.25, np.ones(6)))
    with pytest.raises(ValueError, match="0.25 bin spacings"):
        _zero_placement(grid)


def test_grid_starting_well_above_zero_integer_shift():
    # Edges at whole multiples of 0.5 starting at 5.0: shift is 10 bins
    grid = _build_grid(_spectrum(_edges(10, 4, 0.5), np.ones(4)))
    placement = _zero_placement(grid)
    assert placement.reading == "lower_edge"
    assert placement.shift == 10


def test_explicit_lower_edge_matches_no_warning(recwarn):
    fund = _spectrum(_edges(0, 6, 1.0), np.ones(6))
    init = _spectrum(_edges(0, 6, 1.0), np.ones(6))
    convolve_orders(init, fund, max_order=3, initial_order=1, bin_position="lower_edge")
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_lower_edge_default_warns_on_edge_grid():
    fund = _spectrum(_edges(0, 6, 1.0), np.ones(6))
    init = _spectrum(_edges(0, 6, 1.0), np.ones(6))
    with pytest.warns(UserWarning, match="lower edges"):
        convolve_orders(init, fund, max_order=3, initial_order=1)


def test_explicit_centre_mismatch_raises():
    fund = _spectrum(_edges(0, 6, 1.0), np.ones(6))
    init = _spectrum(_edges(0, 6, 1.0), np.ones(6))
    with pytest.raises(ValueError, match="does not match"):
        convolve_orders(init, fund, max_order=3, initial_order=1, bin_position="centre")


def test_explicit_lower_edge_mismatch_raises_on_centre_grid():
    fund = _spectrum(_centres(-2, 6, 1.0), np.ones(6))
    init = _spectrum(_centres(-2, 6, 1.0), np.ones(6))
    with pytest.raises(ValueError, match="does not match"):
        convolve_orders(
            init, fund, max_order=3, initial_order=1, bin_position="lower_edge"
        )


# ---------------------------------------------------------------------------
# 2.2 One-sided array routine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [0, 1, 3, 7])
def test_one_sided_matches_scipy_reference(shift):
    rng = np.random.default_rng(1000 + shift)
    for _ in range(3):
        n, length = 40, 25
        init = rng.random(n)
        fund = rng.random(length)
        spacing = 0.37
        got = _convolve_orders_arrays(init, fund, 4, shift=shift, spacing=spacing)
        expected = np.array(
            [_scipy_reference(init, fund, k, shift, spacing) for k in range(1, 5)]
        )
        scale = np.abs(expected).max()
        assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


def test_zero_shift_matches_mantid_style_per_order_crop():
    rng = np.random.default_rng(7)
    init = rng.random(50)
    fund = rng.random(30)
    spacing = 0.5
    cur = init.copy()
    expected = []
    for _ in range(5):
        cur = convolve(cur, fund * spacing, mode="full")[: init.size]
        expected.append(cur.copy())
    expected = np.array(expected)
    got = _convolve_orders_arrays(init, fund, 5, shift=0, spacing=spacing)
    scale = np.abs(expected).max()
    assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


# ---------------------------------------------------------------------------
# 2.3 Analytic references
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


def _nearest_bin_density(positions, weights, centres, d):
    """Intensity binned to the nearest centre, returned as density."""
    out = np.zeros(centres.size)
    for x, w in zip(positions, weights, strict=True):
        i = int(np.argmin(np.abs(centres - x)))
        out[i] += w
    return out / d


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


# ---------------------------------------------------------------------------
# 2.4 Wraparound
# ---------------------------------------------------------------------------


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
# 3.1 Padded path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [-1, -2, -4])
def test_padded_path_matches_no_intermediate_crop_reference(shift):
    rng = np.random.default_rng(42 + abs(shift))
    init = rng.random(30)
    fund = rng.random(12)
    spacing = 0.25
    got = _convolve_orders_padded(init, fund, 3, shift=shift, spacing=spacing)
    expected = _brute_padded_reference(init, fund, 3, shift, spacing)
    scale = np.abs(expected).max()
    assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize("shift", [0, 2, 5])
def test_padded_path_agrees_with_one_sided_for_non_negative_shift(shift):
    rng = np.random.default_rng(3)
    init = rng.random(35)
    fund = rng.random(20)
    padded = _convolve_orders_padded(init, fund, 4, shift=shift, spacing=0.3)
    one_sided = _convolve_orders_one_sided(init, fund, 4, shift=shift, spacing=0.3)
    scale = np.abs(one_sided).max()
    assert_allclose(padded, one_sided, rtol=0, atol=1e-12 * scale)


def test_dispatch_selects_two_sided_path_for_negative_shift():
    rng = np.random.default_rng(11)
    init = rng.random(20)
    fund = rng.random(8)
    got = _convolve_orders_arrays(init, fund, 2, shift=-2, spacing=1.0)
    expected = _convolve_orders_padded(init, fund, 2, shift=-2, spacing=1.0)
    assert_array_equal(got, expected)


def test_shift_crop_moves_and_drops():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert_array_equal(_shift_crop(values, 2, 5), [0, 0, 1, 2, 3])
    assert_array_equal(_shift_crop(values, -1, 3), [2, 3, 4])
    assert_array_equal(_shift_crop(values, 0, 4), values)


# ---------------------------------------------------------------------------
# 3.2 Two-sided scenarios
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
# 4.1 Public function
# ---------------------------------------------------------------------------


def _order1(n=12, d=0.5, first=0):
    centres = _centres(first, n, d)
    rng = np.random.default_rng(99)
    y = rng.random(n)
    return _spectrum(centres, y, metadata={"quantum_order": 1})


def test_derived_orders_default_max_order_gives_nine_rows():
    spec = _order1()
    out = convolve_orders(spec, spec, bin_position="centre")
    assert out.y_data.magnitude.shape == (9, 12)
    assert [m["quantum_order"] for m in out.iter_metadata()] == list(range(2, 11))


def test_autoconvolution_to_max_order_four_has_three_rows():
    spec = _order1()
    out = convolve_orders(spec, spec, max_order=4, bin_position="centre")
    assert out.y_data.magnitude.shape[0] == 3
    assert [m["quantum_order"] for m in out.iter_metadata()] == [2, 3, 4]


def test_starting_from_order_two():
    init = _spectrum(_centres(0, 12, 0.5), np.ones(12), metadata={"quantum_order": 2})
    fund = _order1()
    out = convolve_orders(init, fund, max_order=5, bin_position="centre")
    assert [m["quantum_order"] for m in out.iter_metadata()] == [3, 4, 5]


def test_no_scaling_applied_matches_repeated_convolution():
    spec = _order1()
    out = convolve_orders(spec, spec, max_order=3, bin_position="centre")
    d = 0.5
    cur = spec.y_data.magnitude.copy()
    for k in range(2):
        cur = convolve(cur, spec.y_data.magnitude * d, mode="full")[:12]
        assert_allclose(out.y_data.magnitude[k], cur, rtol=1e-12, atol=1e-12)


def test_max_order_not_above_initial_raises():
    spec = _order1()
    with pytest.raises(ValueError, match="greater than the initial order"):
        convolve_orders(spec, spec, max_order=1, bin_position="centre")


def test_initial_order_from_metadata():
    spec = _order1()
    out = convolve_orders(spec, spec, max_order=3, bin_position="centre")
    assert [m["quantum_order"] for m in out.iter_metadata()] == [2, 3]


def test_initial_order_from_argument_when_no_metadata():
    base = _spectrum(_centres(0, 12, 0.5), np.ones(12))
    fund = _order1()
    out = convolve_orders(
        base, fund, max_order=4, initial_order=2, bin_position="centre"
    )
    assert [m["quantum_order"] for m in out.iter_metadata()] == [3, 4]


def test_argument_contradicting_metadata_warns_and_uses_argument():
    spec = _order1()
    with pytest.warns(UserWarning, match="differs") as record:
        out = convolve_orders(
            spec, spec, max_order=4, initial_order=2, bin_position="centre"
        )
    assert record[0].filename == __file__
    assert [m["quantum_order"] for m in out.iter_metadata()] == [3, 4]


def test_no_initial_order_raises():
    base = _spectrum(_centres(0, 12, 0.5), np.ones(12))
    fund = _order1()
    with pytest.raises(ValueError, match="Initial quantum order is unknown"):
        convolve_orders(base, fund, max_order=3, bin_position="centre")


def test_fundamentals_labelled_other_order_warns_and_labels_as_order_one():
    spec = _order1()
    fund = _spectrum(
        _centres(0, 12, 0.5),
        spec.y_data.magnitude,
        metadata={"quantum_order": 2},
    )
    with pytest.warns(UserWarning, match="treated as quantum order 1") as record:
        out = convolve_orders(spec, fund, max_order=3, bin_position="centre")
    assert record[0].filename == __file__
    assert [m["quantum_order"] for m in out.iter_metadata()] == [2, 3]


def test_metadata_propagation_keeps_atom_metadata():
    atom = {"atom_symbol": "H", "mass": 1.008, "atom_index": 0}
    spec = _spectrum(
        _centres(0, 12, 0.5),
        np.random.default_rng(3).random(12),
        metadata={**atom, "quantum_order": 1},
    )
    out = convolve_orders(spec, spec, max_order=3, bin_position="centre")
    for item in out.iter_metadata():
        assert item["atom_symbol"] == "H"
        assert item["mass"] == 1.008
        assert item["atom_index"] == 0


def test_non_spectrum_input_raises_type_error():
    spec = _order1()
    collection = Spectrum1DCollection(
        x_data=spec.x_data,
        y_data=Quantity(np.ones((2, 12)), "1/meV"),
    )
    with pytest.raises(TypeError):
        convolve_orders(collection, spec, max_order=3, bin_position="centre")


# ---------------------------------------------------------------------------
# 4.2 Integration
# ---------------------------------------------------------------------------


def test_autoconvolution_matches_hand_built_collection():
    spec = _order1(n=10, d=1.0)
    out = convolve_orders(spec, spec, max_order=4, bin_position="centre")

    y1 = spec.y_data.magnitude
    expected = []
    cur = y1
    for _ in range(3):
        cur = convolve(cur, y1 * 1.0, mode="full")[:10]
        expected.append(cur)
    reference = Spectrum1DCollection(
        x_data=spec.x_data,
        y_data=Quantity(np.array(expected), "1/meV"),
        metadata={"line_data": [{"quantum_order": q} for q in (2, 3, 4)]},
    )
    assert_allclose(out.y_data.magnitude, reference.y_data.magnitude, atol=1e-12)
    assert [m["quantum_order"] for m in out.iter_metadata()] == [2, 3, 4]


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


def test_apply_weights_on_result_with_atom_metadata():
    atom_meta = {"atom_symbol": "H", "mass": 1.00794, "atom_index": 0}
    spec = _spectrum(
        _centres(0, 12, 0.5),
        np.random.default_rng(4).random(12),
        metadata={**atom_meta, "quantum_order": 1},
    )
    out = convolve_orders(spec, spec, max_order=4, bin_position="centre")
    weighted = apply_weights(out)
    assert weighted.y_data.magnitude.shape == out.y_data.magnitude.shape
    assert not np.allclose(weighted.y_data.magnitude, out.y_data.magnitude)
    assert [m["quantum_order"] for m in weighted.iter_metadata()] == [2, 3, 4]
