"""Write mantid-like second-order data from abinslib implementation."""

from euphonic import QpointPhononModes, Quantity, Spectrum1D
import numpy as np
from snakemake.script import snakemake

from abinslib.almost_isotropic_incoherent import (
    mantid_like_combination_spectra,
)
from abinslib.displacements import Displacements
from abinslib.util import calculate_indirect_q2

modes = QpointPhononModes.from_json_file(snakemake.input[0])
mantid_data = Spectrum1D.from_json_file(snakemake.input[1])
temperature = Quantity(float(mantid_data.metadata["temperature"]), "kelvin")

bins = mantid_data.get_bin_edges(restrict_range=False)

displacements = Displacements.from_modes(modes, temperature=temperature)
dw = displacements.to_atomic_displacements()

q2 = calculate_indirect_q2(
    mantid_data.get_bin_centres(),
    angle=(134.98885653282196 * np.pi / 180),
    final_energy=Quantity(32.0, "cm_1").to("hartree"),
)

spectra = mantid_like_combination_spectra(modes, displacements, dw, q2, bins)
spectrum = spectra.sum()

spectrum.to_json_file(snakemake.output[0])
