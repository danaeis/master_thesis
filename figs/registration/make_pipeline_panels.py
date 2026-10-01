"""
Real-data panels for Figure 3-1 (figs/pipeline_final.tex).

One strip per pipeline step, all from a single VinDr-Multiphase study
(NC fixed, venous moving).  Volumes and masks are the pipeline's own
outputs; the Dice values in row 7 are copied from the pipeline's
evaluation CSVs (A2_zalign / A3_pass0 / A4_pass01, venous phase).

Run:  python3 figs/registration/make_pipeline_panels.py
Needs: nibabel, numpy, scipy, matplotlib
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import patches
from scipy import ndimage as ndi

ROOT = Path("/Users/macbook/develope/thesis/ncct_cect/vindr_ds")
EVAL = Path("/Users/macbook/develope/thesis/multi-phase_registration/"
            "thesis_server_backup/vindr_ds/all_baseline_algorithms")
OUT = Path(__file__).resolve().parent

STUDY = "1.2.840.113619.2.278.3.717616.204.1584947888.428"
SER = {"NC": "1.2.840.113619.2.278.3.717616.204.1584947888.433.3",
       "ART": "1.2.840.113619.2.278.3.717616.204.1584947888.518.4198401",
       "VEN": "1.2.840.113619.2.278.3.717616.204.1584947888.518.4202498"}

# ── geometry of the strips (cm) — must match figs/pipeline_final.tex ──
STRIP_W, STRIP_H = 10.0, 2.55
CM = 1 / 2.54

# ── colours (thesis palette + bright variants for use on dark CT) ──
XABI, XSABZ, XGHERMEZ, XTIRE = "#1F6E80", "#2F6D4C", "#9C3F2B", "#3E525C"
C_BONE = "#56B4E9"      # tier 1
C_SOFT = "#5FCB8C"      # tier 2
C_EVAL = "#F0C75E"      # tier 3 (evaluation only)
C_FIX = "#5FD3F0"       # fixed (NC) contour
C_MOV = "#FF8A3D"       # moving (venous) contour
C_WIN = "#7CE08F"       # common window

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 6.5,
    "axes.linewidth": 0.4,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "pdf.fonttype": 42,
})

WL, WW = 40, 400                       # abdominal window
LO, HI = WL - WW / 2, WL + WW / 2

# seg_full label groups (segmentation/combine_masks.py)
TIER1 = set(range(21, 100))
TIER2 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19}
TIER3 = set(range(100, 150))
HEART, BLADDER = 119, 102


# ════════════════════════════ loading ════════════════════════════
def _p(folder, sid, suffix):
    sub = "" if folder in ("nifti_unprocessed_volumes", "ts_segmentations") else f"{STUDY}/"
    return ROOT / folder / sub / f"{STUDY}_{sid}{suffix}"


def load(folder, phase, suffix, seg=False):
    im = nib.load(_p(folder, SER[phase], suffix))
    arr = np.asarray(im.dataobj)
    arr = arr.astype(np.int16) if seg else arr.astype(np.float32)
    return arr, im.affine


# volumes are LPS voxel order: axis0 → L, axis1 → P, axis2 → S
def axial(v, z):      # radiological: patient right on image left, anterior up
    return v[:, :, z].T


def coronal(v, y):    # superior up
    return v[:, y, :].T[::-1]


def gray(ax, img, extent=None):
    ax.imshow(img, cmap="gray", vmin=LO, vmax=HI, interpolation="lanczos",
              extent=extent, aspect="equal", rasterized=True)


def overlay(ax, lab, colours, alpha=0.55, extent=None):
    rgba = np.zeros(lab.shape + (4,))
    for ids, c in colours:
        m = np.isin(lab, list(ids))
        rgba[m, :3] = matplotlib.colors.to_rgb(c)
        rgba[m, 3] = alpha
    ax.imshow(rgba, interpolation="nearest", extent=extent, aspect="equal", rasterized=True)


def contour(ax, mask, colour, lw=0.7, ls="-", extent=None):
    if mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[colour], linewidths=lw,
                   linestyles=ls, extent=extent)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tag(ax, text, loc="tl", colour="white", size=6.5, box=False):
    x, ha = (0.04, "left") if "l" in loc else (0.96, "right")
    y, va = (0.96, "top") if "t" in loc else (0.04, "bottom")
    bb = dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.6) if box else None
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=colour,
            fontsize=size, fontweight="bold", bbox=bb)


def legend_patch(ax, y, colour, text, alpha=1.0):
    ax.add_patch(patches.Rectangle((0.0, y - 0.045), 0.13, 0.09, facecolor=colour, alpha=alpha,
                                   transform=ax.transAxes, clip_on=False))
    ax.text(0.19, y, text, va="center", transform=ax.transAxes)


def legend_line(ax, y, colour, text, ls="-"):
    ax.plot([0.0, 0.13], [y, y], color=colour, lw=1.1, ls=ls, transform=ax.transAxes,
            clip_on=False)
    ax.text(0.19, y, text, va="center", transform=ax.transAxes)


def legend_note(ax, y, text):
    ax.text(0.0, y, text, va="center", color=XTIRE, transform=ax.transAxes, linespacing=1.25,
            fontsize=6)


def body_bbox(v, pad=6):
    """x/y bounding box of the body (HU > -500) over the whole volume."""
    m = (v > -500).any(axis=2)
    xs, ys = np.where(m)
    return (max(xs.min() - pad, 0), min(xs.max() + pad, v.shape[0]),
            max(ys.min() - pad, 0), min(ys.max() + pad, v.shape[1]))


def new_strip():
    return plt.figure(figsize=(STRIP_W * CM, STRIP_H * CM))


def add_ax(fig, x_cm, w_cm, y_cm=0.0, h_cm=STRIP_H):
    return fig.add_axes([x_cm / STRIP_W, y_cm / STRIP_H, w_cm / STRIP_W, h_cm / STRIP_H])


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", dpi=450)
    fig.savefig(OUT / f"{name}.png", dpi=250)
    plt.close(fig)


# ════════════════════════════ data ════════════════════════════
std = {ph: load("nifti_unprocessed_volumes", ph, "_standardized.nii.gz") for ph in SER}
seg_full_std = {ph: load("ts_segmentations", ph, "_seg_full.nii.gz", seg=True)[0] for ph in SER}

aln_fx, A_aln = load("aligned_volumes", "NC", "_aligned.nii.gz")
aln_fx_seg = load("aligned_volumes", "NC", "_aligned_seg_full.nii.gz", seg=True)[0]
mov = {
    "aligned": load("aligned_volumes", "VEN", "_aligned.nii.gz")[0],
    "rigid0": load("fixed_aligned_rigid_registered", "VEN", "_rigid0.nii.gz")[0],
    "rigid1": load("fixed_aligned_rigid_registered", "VEN", "_rigid1.nii.gz")[0],
}
mov_seg = {
    "aligned": load("aligned_volumes", "VEN", "_aligned_seg_full.nii.gz", seg=True)[0],
    "rigid0": load("fixed_aligned_rigid_registered", "VEN", "_rigid0_seg_full.nii.gz", seg=True)[0],
    "rigid1": load("fixed_aligned_rigid_registered", "VEN", "_rigid1_seg_full.nii.gz", seg=True)[0],
}


def world_z(aff, k):
    return aff[2, 3] + k * aff[2, 2]


def z_index(aff, z_mm):
    return int(round((z_mm - aff[2, 3]) / aff[2, 2]))


def y_index(aff, y_mm):       # nibabel world is RAS; axis1 of LPS data runs toward P
    return int(round((y_mm - aff[1, 3]) / aff[1, 1]))


# Reference levels, chosen from the NC organ centroids (world RAS mm)
nc_seg = seg_full_std["NC"]
A_nc = std["NC"][1]
cen = lambda lab: A_nc @ np.r_[ndi.center_of_mass(nc_seg == lab), 1]
Z_KIDNEY = 0.5 * (cen(3)[2] + cen(4)[2]) + 12          # hilum level: liver, both kidneys, spleen tip
Y_CORONAL = 0.5 * (cen(3)[1] + cen(4)[1]) + 6          # through both kidneys and the spine


# ════════════════════════ 1 · series selection ════════════════════════
def row1():
    fig = new_strip()
    w = 3.2
    for i, ph in enumerate(["NC", "ART", "VEN"]):
        v, a = std[ph]
        img = axial(v, z_index(a, Z_KIDNEY))
        ax = add_ax(fig, 0.1 + i * (w + 0.1), w)
        gray(ax, _fit(img, w / STRIP_H))
        clean(ax)
        tag(ax, {"NC": "non-contrast", "ART": "arterial", "VEN": "portal-venous"}[ph], box=True)
    save(fig, "row1_series")


def _fit(img, aspect, pad=4, labels=None):
    """Crop a slice to the patient body (largest HU > -400 component), then pad with air
    to the requested width/height aspect.  `labels` (same shape) is cropped alike."""
    lab, n = ndi.label(img > -400)
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    r, c = np.where(body)
    r0, r1, c0, c1 = r.min() - pad, r.max() + pad, c.min() - pad, c.max() + pad
    h, w = r1 - r0, c1 - c0
    if w / h < aspect:
        d = int(round((h * aspect - w) / 2)); c0, c1 = c0 - d, c1 + d
    else:
        d = int(round((w / aspect - h) / 2)); r0, r1 = r0 - d, r1 + d
    rr0, cc0 = max(r0, 0), max(c0, 0)
    rr1, cc1 = min(r1, img.shape[0]), min(c1, img.shape[1])

    def crop(a, fill):
        out = np.full((r1 - r0, c1 - c0), fill, dtype=a.dtype)
        out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = a[rr0:rr1, cc0:cc1]
        return out
    return crop(img, -1024.0) if labels is None else (crop(img, -1024.0), crop(labels, 0))


# ════════════════════ 2 · standardization (common z axis) ════════════════════
def _zspan(ph):
    v, a = std[ph]
    return world_z(a, 0), world_z(a, v.shape[2] - 1)


def row2():
    fig = new_strip()
    spans = {ph: _zspan(ph) for ph in SER}
    zmin = min(s[0] for s in spans.values()); zmax = max(s[1] for s in spans.values())
    common = (max(s[0] for s in spans.values()), min(s[1] for s in spans.values()))
    x0, x1, _, _ = body_bbox(std["NC"][0])
    w_mm = (x1 - x0) * 1.5
    w_cm = STRIP_H * w_mm / (zmax - zmin) * 1.0
    for i, ph in enumerate(["NC", "ART", "VEN"]):
        v, a = std[ph]
        ax = add_ax(fig, 0.12 + i * (w_cm + 0.35), w_cm)
        ax.set_facecolor("#15191c")
        s0, s1 = spans[ph]
        gray(ax, coronal(v, y_index(a, Y_CORONAL))[:, x0:x1], extent=(0, w_mm, s0, s1))
        ax.set_xlim(0, w_mm); ax.set_ylim(zmin, zmax)
        # parts outside the common range
        for lo, hi in [(s0, common[0]), (common[1], s1)]:
            if hi - lo > 0.5:
                ax.add_patch(patches.Rectangle((0, lo), w_mm, hi - lo, facecolor=XGHERMEZ,
                                               alpha=0.55, edgecolor="none"))
        clean(ax)
        tag(ax, ph, "tl", box=True)
    # legend
    lx = 0.12 + 3 * (w_cm + 0.35) + 0.1
    ax = add_ax(fig, lx, STRIP_W - lx)
    ax.axis("off")
    legend_patch(ax, 0.70, XGHERMEZ, "not in all\nphases", alpha=0.7)
    legend_note(ax, 0.36, "one z axis (mm);\nLPS, 1.5 mm\nisotropic")
    save(fig, "row2_standardize")


# ════════════════════════ 3 · segmentation / masks ════════════════════════
COLS = [(TIER1, C_BONE), (TIER2, C_SOFT), (TIER3, C_EVAL)]


def row3():
    fig = new_strip()
    v, a = std["NC"]
    s = seg_full_std["NC"]
    x0, x1, y0, y1 = body_bbox(v)
    # coronal
    yv = y_index(a, Y_CORONAL)
    cor, corl = coronal(v, yv)[:, x0:x1], coronal(s, yv)[:, x0:x1]
    wc = STRIP_H * cor.shape[1] / cor.shape[0]
    ax = add_ax(fig, 0.12, wc)
    gray(ax, cor); overlay(ax, corl, COLS); clean(ax); ax.set_aspect("auto")
    tag(ax, "NC", box=True)
    # axial
    z = z_index(a, Z_KIDNEY)
    wa = 3.0
    axi, axl = _fit(axial(v, z), wa / STRIP_H, labels=axial(s, z))
    ax = add_ax(fig, 0.12 + wc + 0.12, wa)
    gray(ax, axi); overlay(ax, axl, COLS); clean(ax); ax.set_aspect("auto")
    # venous coronal
    vv, av = std["VEN"]; sv = seg_full_std["VEN"]
    yv2 = y_index(av, Y_CORONAL)
    ax = add_ax(fig, 0.12 + wc + 0.12 + wa + 0.12, wc)
    gray(ax, coronal(vv, yv2)[:, x0:x1]); overlay(ax, coronal(sv, yv2)[:, x0:x1], COLS)
    clean(ax); ax.set_aspect("auto"); tag(ax, "VEN", box=True)
    # legend
    lx = 0.12 + 2 * wc + wa + 0.5
    ax = add_ax(fig, lx, STRIP_W - lx); ax.axis("off")
    for k, (c, t) in enumerate([(C_BONE, "bone"), (C_SOFT, "stable soft"),
                                (C_EVAL, "eval. only")]):
        legend_patch(ax, 0.84 - k * 0.17, c, t)
    legend_note(ax, 0.22, "reg. mask =\nbone + stable soft;\nbowel excluded")
    save(fig, "row3_segment")


# ════════════════════════ 4 · common crop window ════════════════════════
def row4():
    fig = new_strip()
    spans = {ph: _zspan(ph) for ph in ("NC", "VEN")}
    zmin = min(s[0] for s in spans.values()); zmax = max(s[1] for s in spans.values())
    win = (world_z(A_aln, 0), world_z(A_aln, aln_fx.shape[2] - 1))
    x0, x1, _, _ = body_bbox(std["NC"][0])
    w_mm = (x1 - x0) * 1.5
    w_cm = STRIP_H * w_mm / (zmax - zmin)
    for i, ph in enumerate(["NC", "VEN"]):
        v, a = std[ph]
        s0, s1 = spans[ph]
        yv = y_index(a, Y_CORONAL)
        ax = add_ax(fig, 0.12 + i * (w_cm + 0.35), w_cm)
        ax.set_facecolor("#15191c")
        ext = (0, w_mm, s0, s1)
        gray(ax, coronal(v, yv)[:, x0:x1], extent=ext)
        ax.set_xlim(0, w_mm); ax.set_ylim(zmin, zmax)
        for lo, hi in [(zmin, win[0]), (win[1], zmax)]:
            ax.add_patch(patches.Rectangle((0, lo), w_mm, hi - lo, facecolor="black",
                                           alpha=0.6, edgecolor="none"))
        ax.add_patch(patches.Rectangle((1, win[0]), w_mm - 2, win[1] - win[0], fill=False,
                                       edgecolor=C_WIN, lw=0.8, ls=(0, (2.5, 1.5))))
        clean(ax); tag(ax, ph, box=True)
    # overlay of the cropped pair before any registration: checkerboard
    cb = _checker(aln_fx, mov["aligned"], y_index(A_aln, Y_CORONAL))
    x0a, x1a, _, _ = body_bbox(aln_fx)
    cb = cb[:, x0a:x1a]
    wcb = STRIP_H * cb.shape[1] / cb.shape[0]
    lx = 0.12 + 2 * (w_cm + 0.35) + 0.1
    ax = add_ax(fig, lx, wcb)
    gray(ax, cb); clean(ax); ax.set_aspect("auto")
    tag(ax, "NC | VEN", "tl", box=True)
    # legend
    lx2 = lx + wcb + 0.15
    ax = add_ax(fig, lx2, STRIP_W - lx2); ax.axis("off")
    legend_line(ax, 0.80, XSABZ, "common\nwindow", ls=(0, (2.5, 1.5)))
    legend_patch(ax, 0.56, "black", "removed", alpha=0.6)
    legend_note(ax, 0.22, "right: NC/VEN\ncheckerboard,\nno registration")
    save(fig, "row4_crop")


def _checker(a, b, y, tile=16):
    ca, cb = coronal(a, y), coronal(b, y)
    ii, jj = np.indices(ca.shape)
    return np.where(((ii // tile) + (jj // tile)) % 2 == 0, ca, cb)


# ════════════════════════ 5/6 · rigid passes (zoomed contours) ════════════════════════
# Zoom windows (aligned grid, voxel units): organ, plane, slice, centre
def _zoom_specs():
    specs = []
    for lab, name, plane in [(2, "spleen", "cor"), (4, "right kidney", "cor"), (5, "pancreas", "ax")]:
        m = (aln_fx_seg == lab) | (mov_seg["aligned"] == lab)
        x = (aln_fx_seg == lab) ^ (mov_seg["aligned"] == lab)
        if plane == "cor":
            k = int(np.argmax(x.sum((0, 2))))
            c = ndi.center_of_mass(m[:, k, :])
        else:
            k = int(np.argmax(x.sum((0, 1))))
            c = ndi.center_of_mass(m[:, :, k])
        specs.append((lab, name, plane, k, c))
    return specs


ZOOM_MM = 105


def _zoom(vol, plane, k, c):
    h = int(ZOOM_MM / 1.5 / 2)
    if plane == "cor":
        img = coronal(vol, k)
        r = vol.shape[2] - 1 - int(round(c[1])); col = int(round(c[0]))
    else:
        img = axial(vol, k)
        r = int(round(c[1])); col = int(round(c[0]))
    r0, c0 = max(r - h, 0), max(col - h, 0)
    return img[r0:r0 + 2 * h, c0:c0 + 2 * h]


# eroded Dice from the pipeline's own evaluation (venous phase)
def _eval_dice():
    out = {}
    for key, cond in [("aligned", "A2_zalign"), ("rigid0", "A3_pass0"), ("rigid1", "A4_pass01")]:
        d = pd.read_csv(EVAL / cond / "eval_detail.csv")
        d = d[(d.study_id == STUDY) & (d.phase == "Venous")]
        out[key] = dict(zip(d.organ, d.eroded_dice))
    return out


DICE = _eval_dice()
EVAL_NAME = {2: "spleen", 4: "kidney_right", 5: "pancreas"}


def row_pass(stage, name):
    fig = new_strip()
    specs = _zoom_specs()
    w = STRIP_H
    for i, (lab, oname, plane, k, c) in enumerate(specs):
        ax = add_ax(fig, 0.12 + i * (w + 0.12), w)
        gray(ax, _zoom(mov[stage], plane, k, c))
        contour(ax, _zoom(aln_fx_seg == lab, plane, k, c), C_FIX, lw=0.9)
        contour(ax, _zoom(mov_seg[stage] == lab, plane, k, c), C_MOV, lw=0.9)
        clean(ax)
        tag(ax, oname, "tl", size=6, box=True)
        before, now = DICE["aligned"][EVAL_NAME[lab]], DICE[stage][EVAL_NAME[lab]]
        tag(ax, f"{before:.2f} → {now:.2f}", "bl", colour="#FFE9A8", size=6, box=True)
    lx = 0.12 + 3 * (w + 0.12) + 0.1
    ax = add_ax(fig, lx, STRIP_W - lx); ax.axis("off")
    legend_line(ax, 0.84, C_FIX, "NC\n(fixed)")
    legend_line(ax, 0.60, C_MOV, "VEN\n(moving)")
    legend_note(ax, 0.24, "eroded Dice:\ncrop only →\nthis pass")
    save(fig, name)


# ════════════════════════ 7 · evaluation ════════════════════════
def row7():
    fig = new_strip()
    organs = ["liver", "spleen", "kidney_left", "kidney_right", "pancreas", "gallbladder",
              "aorta", "inferior_vena_cava", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3"]
    short = ["liver", "spleen", "kidney\nleft", "kidney\nright", "pancreas", "gall-\nbladder",
             "aorta", "IVC", "L1", "L2", "L3"]
    ax = add_ax(fig, 0.95, STRIP_W - 1.05, y_cm=0.62, h_cm=STRIP_H - 1.0)
    x = np.arange(len(organs))
    style = [("aligned", "crop only", "#9AA7AE", "o"), ("rigid0", "pass 0 (Kabsch)", XABI, "s"),
             ("rigid1", "pass 1 (Nelder–Mead)", XSABZ, "D")]
    for j, (key, lbl, col, mk) in enumerate(style):
        ax.scatter(x + (j - 1) * 0.2, [DICE[key][o] for o in organs], s=9, marker=mk,
                   color=col, label=lbl, zorder=3, linewidths=0)
    for i, o in enumerate(organs):
        ax.plot([i - 0.2, i + 0.2], [DICE["aligned"][o], DICE["rigid1"][o]], color="#C3CFD4",
                lw=0.5, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6)
    ax.set_ylim(0.35, 1.0); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("eroded Dice", fontsize=6.5)
    ax.grid(axis="y", color="#E3E8EA", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6,
              handletextpad=0.2, columnspacing=1.2, borderaxespad=0.1)
    save(fig, "row7_eval")


if __name__ == "__main__":
    row1(); row2(); row3(); row4()
    row_pass("rigid0", "row5_pass0")
    row_pass("rigid1", "row6_pass1")
    row7()
    print("done")
