"""Figures for the reproduction (matplotlib, CPU). Saved as PNG into outdir/images."""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fig_claim1_eigs(eigs9, outdir, K):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(range(1, len(eigs9) + 1), eigs9, color="#2a6")
    ax.set_xlabel(r"outlier index  (over the $K^2$ directions)")
    ax.set_ylabel("eigenvalue")
    ax.set_title(f"Theorem 4.1: $K^2={K*K}$ nonzero Hessian eigenvalues\n"
                 f"all equal (max/min={max(eigs9)/min(eigs9):.2e})")
    fig.tight_layout()
    p = os.path.join(outdir, "claim1_hessian_eigs.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_claim5_spectrum(dyn_rows, outdir, K):
    epochs = [r["step"] for r in dyn_rows]
    fig, axes = plt.subplots(1, len(epochs), figsize=(2.0 * len(epochs), 2.6), sharey=False)
    if len(epochs) == 1:
        axes = [axes]
    for ax, r in zip(axes, dyn_rows):
        top = r["top_eigs"]
        bulk_max = r["bulk_max"]
        allv = np.array(top + [bulk_max])
        lo = max(allv.min() * 0.5, 1e-9)
        bins = np.logspace(np.log10(lo), np.log10(allv.max() * 1.1), 25)
        ax.hist(top, bins=bins, color="#c33", alpha=0.85, label=f"$K^2$ outliers")
        ax.axvline(bulk_max, color="#888", ls="--", label="bulk max")
        ax.set_xscale("log")
        ax.set_title(f"epoch {r['step']}")
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Claim 5 (linear UFM): $K^2=9$ Hessian outliers separate\nand converge to one value",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(outdir, "claim5_spectrum.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_claim5_fcc(dyn_rows, outdir):
    steps = [r["step"] for r in dyn_rows]
    means = [r["fcc_mean"] for r in dyn_rows]
    mins = [r["fcc_min"] for r in dyn_rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(steps, means, "o-", color="#257", label=r"$f_{cc'}$ mean")
    ax.plot(steps, mins, "s--", color="#e80", label=r"$f_{cc'}$ min")
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel(r"eigenvector alignment $f_{cc'}$")
    ax.set_ylim(0, 1.1)
    ax.set_title("Claim 5: predicted Hessian eigenvectors\nbecome exact ($f\\to 1$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "claim5_fcc.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_claim6(relu_eigs, relu_coeffs, outdir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.0))
    a1.bar(range(1, len(relu_eigs) + 1), relu_eigs, color="#7a3")
    a1.set_title(f"Claim 6 (ReLU): $K^2=9$ outliers separate\nbut are NOT equal")
    a1.set_xlabel("outlier index"); a1.set_ylabel("eigenvalue")
    a2.bar(range(1, len(relu_coeffs) + 1), relu_coeffs, color="#739")
    a2.set_title("gradient coefficients in the\ntrue Hessian eigenbasis")
    a2.set_xlabel("eigenvector (sorted)"); a2.set_ylabel(r"$\tilde f_{cc'}$")
    fig.tight_layout()
    p = os.path.join(outdir, "claim6_relu.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_gn_decomp(c2, outdir):
    """Stacked-rank bar: ranks of G_class / G_cross / G_within vs predicted."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    names = ["G_within", "G_cross", "G_class"]
    got = [c2["rank_within"], c2["rank_cross"], c2["rank_class"]]
    exp = [c2["exp_within"], c2["exp_cross"], c2["exp_class"]]
    x = np.arange(3)
    ax.bar(x - 0.2, got, 0.4, label="observed", color="#2a6")
    ax.bar(x + 0.2, exp, 0.4, label="predicted", color="#c63")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("rank")
    ax.set_title("Theorem 4.2: GN decomposition ranks")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "claim2_gn_decomp.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p
