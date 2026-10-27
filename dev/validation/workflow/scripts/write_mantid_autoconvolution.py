"""Write autoconvolution quantum-order spectra using Mantid.

Runs Abins with ``Autoconvolution=True`` from the initial quantum order given
in the rule parameters. Per-atom workspaces are grouped by quantum order, summed
over atoms, and written as one ``Spectrum1DCollection`` with one row per order.
"""

from collections import defaultdict
import re
from tempfile import TemporaryDirectory

from euphonic import Quantity
from euphonic.spectra import Spectrum1DCollection
import mantid.simpleapi
from snakemake.script import snakemake

import abins.parameters  # isort:skip


WORKSPACE = "wrk"
_ORDER_PATTERN = re.compile(r"_quantum_event_(\d+)")

if snakemake.params["raw"]:
    # Disable instrumental resolution broadening
    abins.parameters.sampling["broadening_scheme"] = "none"

with TemporaryDirectory() as tmpdir:
    abins_kwargs = {
        "VibrationalOrPhononFile": snakemake.input[0],
        "AbInitioProgram": "JSON",
        "OutputWorkspace": WORKSPACE,
        "CacheDirectory": tmpdir,
    } | snakemake.params["abins_kwargs"]
    abins_kwargs["QuantumOrderEventsNumber"] = str(snakemake.params["initial_order"])
    abins_kwargs["Autoconvolution"] = True

    mantid.simpleapi.Abins(**abins_kwargs)


def _group_workspaces_by_order() -> dict[int, list]:
    """Group per-atom quantum-event workspaces by quantum order."""
    grouped: dict[int, list] = defaultdict(list)
    for name in mantid.simpleapi.mtd.getObjectNames():
        if match := _ORDER_PATTERN.search(name):
            grouped[int(match.group(1))].append(mantid.simpleapi.mtd[name])
    return dict(grouped)


workspaces_by_order = _group_workspaces_by_order()
orders = sorted(workspaces_by_order)
if orders != list(range(1, len(orders) + 1)):
    msg = f"Expected contiguous quantum orders starting at 1, found {orders}"
    raise ValueError(msg)

first_workspace = workspaces_by_order[orders[0]][0]
energy = first_workspace.getAxis(0).extractValues()
bin_width = energy[1] - energy[0]
energy_unit = first_workspace.getAxis(0).getUnit().symbol().ascii()

intensity = []
for order in orders:
    summed = sum(workspace.extractY()[0] for workspace in workspaces_by_order[order])
    intensity.append(summed / bin_width)

spectra = Spectrum1DCollection(
    x_data=Quantity(energy, str(energy_unit)),
    y_data=Quantity(intensity, f"barn / {energy_unit}"),
    metadata={
        "temperature": str(abins_kwargs["TemperatureInKelvin"]),
        "line_data": [{"quantum_order": order} for order in orders],
    },
)

spectra.to_json_file(snakemake.output[0])
