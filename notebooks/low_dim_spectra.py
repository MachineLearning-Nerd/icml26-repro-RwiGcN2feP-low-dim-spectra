"""marimo notebook — Unifying Low-Dimensional Spectra in Deep Learning (2404.06106).

Tutorial walkthrough of the central claim and the reproduced evidence. Opens with the
already-produced results (figures fetched from the public Space), so readers need not
rerun the full verification. For the live experiment use `bash repro/run.sh`.

Run locally:  marimo edit notebooks/low_dim_spectra.py
              marimo run  notebooks/low_dim_spectra.py
"""
import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        """# Unifying Low-Dimensional Spectra in Deep Learning

        **arXiv 2404.06106v1 · OpenReview RwiGcN2feP — 6/6 scoped checks pass (CPU).**

        The paper asks *why* a deep network's Hessian, gradient, and weight spectra are
        dominated by a handful of **outlier eigenvalues** whose count equals the number
        of classes `K`. Its answer: at convergence the network undergoes **Deep Neural
        Collapse (DNC)**, and that collapse *forces* the low-dimensional spectra. We
        reproduce the four v1 theorems (4.1–4.4) and both empirical figures at the
        pinned protocol (`K=3, d=60, n=40, L=5`). The strict paper-level gate remains
        closed because the training horizons and current arXiv revision differ."""
    )
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """## The model and its optimum

        The deep linear **unconstrained-features model**: an `L`-layer linear chain
        `W_L…W_1` maps *freely optimised* features `H_1` to logits, with MSE loss +
        weight decay. At the global optimum, DNC gives orthogonal class means `μ_c` and
        weights aligned to them. We construct this optimum to machine precision
        (`‖∇L‖∞ ≈ 5e-16`) and **also recover it by gradient descent** (same loss) — two
        independent routes."""
    )
    return


@app.cell
def _(mo):
    mo.md(
        """## Why the Hessian check is not circular

        The layer-wise Hessian is computed **two independent ways** and shown identical
        at an arbitrary point: the paper's Kronecker formula vs JAX `hessian` of the MSE
        loss. Max-abs difference **≈ 1e-17**. So the spectrum is the true loss Hessian's
        spectrum, not a restatement of the theorem."""
    )
    return


@app.cell
def _(mo):
    mo.md("### Claim 1 (Thm 4.1): the layer Hessian has exactly K²=9 equal non-zero eigenvalues")
    return


@app.cell
def _(mo):
    mo.image(src="https://huggingface.co/spaces/DineshAI/RwiGcN2feP/resolve/main/runs/63f3331/images/claim1_hessian_eigs.png")
    return


@app.cell
def _(mo):
    mo.md("### Claim 5 (Figs 3–4): over training, f_cc rises ~0.2 → 1 and the 9 outliers equalise")
    return


@app.cell
def _(mo):
    mo.image(src="https://huggingface.co/spaces/DineshAI/RwiGcN2feP/resolve/main/runs/63f3331/images/claim5_fcc.png")
    return


@app.cell
def _(mo):
    mo.md("### Claim 6 (Fig 9, Table 2): with ReLU, outliers separate but stay *unequal* (4.98×), unlike the linear 1.0×")
    return


@app.cell
def _(mo):
    mo.image(src="https://huggingface.co/spaces/DineshAI/RwiGcN2feP/resolve/main/runs/63f3331/images/claim6_relu.png")
    return


@app.cell
def _(mo):
    mo.md(
        """## Reproduce

        ```bash
        bash repro/run.sh    # uv sync --frozen && uv run python -m repro.verify_all
        ```

        Full per-claim evidence and raw JSON/CSV: the
        [Verification run](https://huggingface.co/spaces/DineshAI/RwiGcN2feP) page on the
        Space. Illustrated report: `reports/low-dim-spectra/report.md`."""
    )
    return


if __name__ == "__main__":
    app.run()
