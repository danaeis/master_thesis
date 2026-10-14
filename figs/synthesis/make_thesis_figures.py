"""
Thesis versions of the downstream figures (chapter 5).

Wraps synthetic_CECT/scripts/make_downstream_figures.py and changes three things:
  * one run per configuration: the extra checkpoints of E and F are dropped and
    no label carries an epoch number;
  * the downstream quantity is the vascular Dice change vs NCCT, not the
    "net effect" (vascular minus skeletal);
  * the tax-vs-gain figure no longer draws the y = x "no net effect" line.

Run from anywhere:
    python figs/synthesis/make_thesis_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path("/Users/macbook/develope/thesis/synthetic_CECT")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
import make_downstream_figures as m  # noqa: E402

DROPPED = {"mp_lvl_adv_best", "mp_lvl_adv_slc11_last", "slc11_popphase_ven"}
VENOUS = [a for a in m.VENOUS_ARMS if a not in DROPPED]

m.DISPLAY.update({"plcnet_e": "PLC-Net 2D", "plcnet_f": "PLC-Net 2.5D",
                  "slc11_popphase_ven": "PLC-Net 2.5D, phase-specific reference"})
m.BROKEN = {"ddpm", "cytran", "gea_gan", "dea_gan"}
m.PHASE_PAIRS = [p for p in m.PHASE_PAIRS
                 if p[0] not in {"mp_lvl_adv_best", "mp_lvl_adv_slc11_last"}]
m.LADDER = [
    ("mp_uncond",         "A\nno conditioning"),
    ("multiphase_film",   "B\n+ phase FiLM"),
    ("mp_film_adv",       "C\n+ adversarial\n(λ_adv = 2.0)"),
    ("mp_film_adv_slc11", "D\n+ 11 slices"),
    ("plcnet_e",          "E: back to 1 slice; + level,\nλ_adv = 0.5, FM, GroupNorm\n(PLC-Net 2D)"),
    ("plcnet_f",          "F: E + 11 slices\n(PLC-Net 2.5D)"),
]
m.BENCH_NAME = {a: b for a, b in m.BENCH_NAME.items() if a in VENOUS}

VASC_LABEL = "Δ vascular Dice vs NCCT"


def vascular_delta(rows, arm):
    return m.paired(m.per_case(rows, arm, group="vascular"),
                    m.per_case(rows, "ncct", group="vascular"))[0]


m.net_effect = vascular_delta       # fig 3 and fig 7 plot whatever this returns


def relabel(name, ylabel):
    fig = plt.gcf()
    for ax in fig.axes:
        if "net downstream effect" in ax.get_ylabel():
            ax.set_ylabel(ylabel, fontsize=10)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = m.load_rows(REPO / "analysis/downstream_seg_all/per_case.csv")
    bench = REPO / "analysis/bench_ncct2cect/master_table.csv"
    tmp = OUT / "_tmp"
    tmp.mkdir(exist_ok=True)
    np.random.seed(0)

    m.fig_per_organ(rows, OUT, m.present(rows, m.HEADLINE_ARMS))
    m.fig_phase(rows, OUT)
    m.fig_gallbladder(rows, OUT)

    # figures 3 and 7: drawn by the original code, y-label replaced before saving
    save = m.save
    m.save = lambda fig, out, name: None
    m.fig_fidelity(rows, bench, tmp)
    relabel("fig3_fidelity_vs_utility.png", VASC_LABEL)
    m.fig_ladder(rows, tmp)
    relabel("fig7_ablation_ladder.png", VASC_LABEL + "\n(mean ± 95% CI)")

    # figure 4 without the y = x line
    m.fig_tax(rows, tmp, m.present(rows, m.TAX_ARMS), m.present(rows, VENOUS))
    m.save = save
    fig = plt.gcf()
    ax = fig.axes[0]
    for line in list(ax.lines):
        if line.get_label().startswith("y = x"):
            line.remove()
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    fig.savefig(OUT / "fig4_tax_vs_gain.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
