"""Plot comparison between Mantid and abinslib calculations."""

from euphonic import Quantity, Spectrum1D
from euphonic.plot import plot_1d_to_axis
import matplotlib.pyplot as plt
from snakemake.script import snakemake

width = Quantity(1, "meV")

abinslib_data = Spectrum1D.from_json_file(snakemake.input[0]).broaden(width)
mantid_data = Spectrum1D.from_json_file(snakemake.input[1]).broaden(width)

fig, ax = plt.subplots()

plot_1d_to_axis(abinslib_data, ax, label=snakemake.params["label1"])
plot_1d_to_axis(mantid_data, ax, linestyle="--", label=snakemake.params["label2"])
ax.legend()
ax.set_title(snakemake.params["title"] + f"\n{width:~P} broadening")
ax.set_xlim(0, snakemake.params["xlim"])
ax.set_xlabel("Energy transfer / meV")
ax.set_ylabel("$S(\omega)$ / barn meV$^{-1}$")
if not snakemake.params.get("linear"):
    ax.set_yscale("log")

fig.tight_layout()
fig.savefig(snakemake.output[0])
