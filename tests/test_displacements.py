"""Unit tests for abinslib.displacements module"""

from itertools import product
import json
import warnings

from euphonic import Quantity
import numpy as np
from numpy.testing import assert_allclose
from packaging.version import InvalidVersion
import pytest

from abinslib.displacements import Displacements
from abinslib.util import get_version


@pytest.fixture
def sample_displacements_kwargs(rng):
    """Sample kwargs for creating a Displacements instance."""
    return {
        "displacements": Quantity(rng.random((2, 4, 5, 3, 3)), "bohr**2"),
        "weights": np.array((0.2, 0.8)),
        "bose_n": rng.random((2, 4)),
        "temperature": Quantity(10, "K"),
    }


@pytest.fixture
def sample_displacements_data_dict(sample_displacements_kwargs):
    """Sample JSON-serializable dictionary for Displacements with metadata."""
    displacements = Displacements(**sample_displacements_kwargs)
    data_dict = displacements.to_dict()
    data_dict["__abinslib_class__"] = "Displacements"
    data_dict["__abinslib_version__"] = "0.0.91"
    return data_dict


def assert_displacements_equal(actual: Displacements, expected: Displacements) -> None:
    """Assert two Displacements objects are equivalent."""
    assert actual.displacements.units == expected.displacements.units
    assert_allclose(actual.displacements.magnitude, expected.displacements.magnitude)
    assert_allclose(actual.weights, expected.weights)
    assert_allclose(actual.bose_n, expected.bose_n)
    assert actual.temperature == expected.temperature


def test_displacements(sample_displacements_kwargs):
    """Self-consistency check of displacements properties"""
    displacements = Displacements(**sample_displacements_kwargs)

    assert_allclose(
        (displacements.n / displacements.one).magnitude[:, :, 0, 0, 0],
        displacements.bose_n,
    )

    assert_allclose(
        (displacements.two_n_plus_one - displacements.one).magnitude,
        (displacements.n * 2.0).magnitude,
    )


def test_displacements_to_dict_and_json(tmp_path, sample_displacements_kwargs):
    """Test serialization of Displacements to dict, JSON string, and JSON file"""
    displacements = Displacements(**sample_displacements_kwargs)

    data_dict = displacements.to_dict()
    assert isinstance(data_dict, dict)
    assert data_dict["displacements_unit"] == "bohr ** 2"
    assert data_dict["temperature"] == 10.0
    assert data_dict["temperature_unit"] == "kelvin"
    assert isinstance(data_dict["displacements"], list)
    assert isinstance(data_dict["weights"], list)
    assert isinstance(data_dict["bose_n"], list)

    json_str = displacements.to_json()
    data = json.loads(json_str)
    assert data["__abinslib_class__"] == "Displacements"
    assert data["__abinslib_version__"] == get_version()

    # Test roundtrip from string
    reloaded_str = Displacements.from_json(json_str)
    assert_displacements_equal(reloaded_str, displacements)

    # Test roundtrip from file
    json_path = tmp_path / "displacements.json"
    displacements.to_json_file(json_path)
    assert json_path.is_file()

    reloaded_file = Displacements.from_json_file(json_path)
    assert_displacements_equal(reloaded_file, displacements)


DELETE = object()


@pytest.mark.parametrize(
    ("replace", "expected"),
    [
        (
            ("__abinslib_class__", "WrongClass"),
            (ValueError, "does not match expected class"),
        ),
        (
            ("__abinslib_class__", DELETE),
            (ValueError, "missing required '__abinslib_class__' key"),
        ),
        (
            ("__abinslib_version__", "DEVELOPMENT"),
            (UserWarning, "generated with a DEVELOPMENT version"),
        ),
        (
            ("__abinslib_version__", "99.0.0"),
            (UserWarning, "is newer than current abinslib version"),
        ),
        (
            ("__abinslib_version__", "not_a_valid_version_!!!"),
            (InvalidVersion, None),
        ),
        (
            ("__abinslib_version__", DELETE),
            (ValueError, "missing required '__abinslib_version__' key"),
        ),
    ],
    ids=lambda x: f"{x[0]}_{'DELETE' if x[1] is DELETE else x[1]}",
)
def test_displacements_from_json_validation(
    sample_displacements_data_dict, monkeypatch, replace, expected
):
    """Test validation of class name and version warnings during JSON loading."""
    current_version = sample_displacements_data_dict["__abinslib_version__"]
    monkeypatch.setattr("abinslib.io.get_version", lambda: current_version)

    key, val = replace
    if val is DELETE:
        del sample_displacements_data_dict[key]
    else:
        sample_displacements_data_dict[key] = val

    json_str = json.dumps(sample_displacements_data_dict)
    expected_cls, match_str = expected

    if issubclass(expected_cls, Warning):
        ctx = pytest.warns(expected_cls, match=match_str)
    else:
        ctx = pytest.raises(expected_cls, match=match_str)

    with ctx:
        Displacements.from_json(json_str)


def test_displacements_from_json_dev_build(sample_displacements_data_dict, monkeypatch):
    """Test DEVELOPMENT build does not issue JSON version warnings."""
    monkeypatch.setattr("abinslib.io.get_version", lambda: "DEVELOPMENT")
    newer_dev_data = sample_displacements_data_dict | {"__abinslib_version__": "99.0.0"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Displacements.from_json(json.dumps(newer_dev_data))


@pytest.mark.parametrize(
    ("modes", "abins_average_a_traces"),
    [("GaSb", (0.01321831, 0.01127088))],
    indirect=("modes",),
)
def test_calculate_adp(modes, abins_average_a_traces):
    """Check ADP agrees with Euphonic and Abins implementations

    Euphonic reference is calculated on-the-fly

    Abins reference average_a_traces are from Abins calculate_isotropic_dw
    method, which takes weighted sum over traces at each k-point. Note that
    there seems to be a difference in scale convention: Abins value is twice
    as large.

    i.e. in Euphonic coherent Debye-Waller term exp(-W) = exp(-Q^2.<dw>),
    in Mantid Abins incoherent term exp(-2W) = exp(-Q^2 tr(A)/3)
    - the 2 in exponent has been absorbed into A

    """

    atomic_displacements = Displacements.from_modes(
        modes, temperature=Quantity(100, "K")
    ).to_atomic_displacements()

    euphonic_dw = modes.calculate_debye_waller(
        temperature=Quantity(100, "K"),
        frequency_min=Quantity(0.01, "meV"),
        symmetrise=False,
    ).debye_waller

    assert atomic_displacements.units == euphonic_dw.units
    assert_allclose(atomic_displacements.magnitude, euphonic_dw.magnitude)

    assert_allclose(
        np.trace(atomic_displacements.to("angstrom^2").magnitude, axis1=1, axis2=2),
        np.array(abins_average_a_traces) / 2,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("modes", "temperature_k"), list(product(["GaSb"], [0, 100])), indirect=("modes",)
)
def test_dw_regression(modes, temperature_k, ndarrays_regression):
    dw = Displacements.from_modes(
        modes, temperature=Quantity(temperature_k, "K")
    ).to_atomic_displacements()
    ndarrays_regression.check({"dw": dw.to("angstrom^2").magnitude})


@pytest.mark.parametrize(
    ("modes", "ref_npz"), [("GaSb", "GaSb_abins_isotropic_dw.npz")], indirect=True
)
def test_a_abins_ref(modes, ref_npz) -> None:
    """Check calculated A against Abins isotropic calculation

    The reference average_a_traces are from Abins calculate_isotropic_dw method
    which takes weighted sum over traces at each k-point.

    """
    ref_a_traces = ref_npz["a_traces"]

    dw = Displacements.from_modes(
        modes, temperature=Quantity(100, "K")
    ).to_atomic_displacements()
    assert_allclose(
        np.trace(dw.to("angstrom^2").magnitude, axis1=1, axis2=2),
        ref_a_traces / 2,
        atol=1e-8,
    )


@pytest.mark.parametrize(
    ("modes", "temperature_k", "ref_npz"),
    [
        ("GaSb", 0, "GaSb_abins_0k_B.npz"),
        ("GaSb", 100, "GaSb_abins_100k_B.npz"),
    ],
    indirect=("modes", "ref_npz"),
)
def test_displacements_abins_ref(modes, temperature_k, ref_npz) -> None:
    """Check calculated displacements against Mantid-Abins reference

    Note that as in ADP there seems to be a factor two difference as Mantid
    implementation has absorbed the "2" to construct 2W when summing over B

    """
    b = Displacements.from_modes(
        modes,
        temperature=Quantity(temperature_k, "kelvin"),
    )

    assert_allclose(
        b.n_plus_one.to("angstrom^2").magnitude[1],
        np.swapaxes(ref_npz["qpt-1"], 0, 1),
    )
