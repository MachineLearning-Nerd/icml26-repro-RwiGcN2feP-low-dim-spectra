# ICML 2026 — Unifying Low-Dimensional Spectra

Scoped reproduction and claim audit for Connall Garrod and Jonathan P. Keating’s
paper on deep neural collapse and low-dimensional spectra in deep learning.

## Paper

- **Reproduced source:** [arXiv:2404.06106v1](https://arxiv.org/abs/2404.06106v1),
  *Unifying Low Dimensional Observations in Deep Learning Through the Deep Linear
  Unconstrained Feature Model*, OpenReview `RwiGcN2feP`.
- **Current arXiv record:** [arXiv:2404.06106v3](https://arxiv.org/abs/2404.06106),
  retitled *Unifying Low Dimensional Spectra in Deep Learning* and accepted at
  ICML 2026. The repository keeps the v1 theorem and section labels explicit so
  that old and revised versions are not silently mixed.
- **Authors:** Connall Garrod and Jonathan P. Keating.

The paper argues that deep neural collapse (DNC) explains the recurring
low-dimensional structure of Hessian, gradient, and weight spectra. In the deep
linear unconstrained-feature model (UFM), class-mean geometry supplies explicit
eigenvalues and eigenvectors; the paper then studies how those structures emerge
during training and how ReLU changes the equality properties.

## Status at a glance

**Scoped result: 6/6 claim checks pass at the pinned v1 UFM protocol.** The strict
paper-level gate is intentionally **not passed**: Claim 5 uses 30,000 rather than
the paper’s longer training horizon, Claim 6 uses 150,000 rather than 1,000,000
ReLU epochs, and the theorem checks cover one balanced, rotation-representative
global optimum. See [`publication_gate.json`](publication_gate.json) for the
machine-readable decision.

| Claim | Paper statement | Evidence status | Result and boundary |
|---|---|---|---|
| C1 — v1 Thm. 4.1 | Layer Hessian has rank `K²` and equal non-zero eigenvalues | **VERIFIED_CONDITIONAL** | Rank 9, eigenvalue ratio 1.00, eigenvector alignment 1.00; verified at the analytic optimum and cross-checked against JAX autodiff |
| C2 — v1 Thm. 4.2 | Gauss–Newton split has ranks `0 / K(K−1) / K` | **VERIFIED_CONDITIONAL** | Observed `0 / 6 / 3`, equal component eigenvalues, orthogonality error `5.8e−18`; same optimum scope |
| C3 — v1 Thm. 4.3 | Gradient uses `K` equal coefficients in the natural basis | **VERIFIED_CONDITIONAL** | Three coefficients of `1/3`, zero off-diagonal coefficient; basis choice matters because the top eigenvalue is degenerate |
| C4 — v1 Thm. 4.4 | Weight Gram matrices have rank `K` and the claimed eigenvalue recurrence | **VERIFIED_CONDITIONAL** | Rank 3, formula relative error 0, recurrence error 0 |
| C5 — v1 Figs. 3–4 | Linear-UFM outliers separate, equalise, and align with DNC means | **REPRODUCED_SCOPED** | Top-9 ratio `2.12 → 1.00`, `f_cc` `0.184 → 1.00`; 30k epochs, with convergence by about epoch 3k |
| C6 — v1 Fig. 9/Table 2 | ReLU outliers remain unequal and gradient coefficients remain unequal | **REPRODUCED_SCOPED** | Outlier ratio 4.98 and top-3 coefficient ratio 31.1; 150k epochs versus the paper’s 1m |

## How each claim is produced

| Claim | Producer path | Evidence snapshot | Control or limitation |
|---|---|---|---|
| C1 | `repro/verify_all.py` → `repro/theorems.py:claim1_hessian`; `repro/ufm.py:analytic_optimum`; `repro/hessian.py` validates the Kronecker Hessian against JAX | [`claim1.json`](docs/evidence/faithful_run_63f3331/claim1.json), [`optimum_meta.json`](docs/evidence/faithful_run_63f3331/optimum_meta.json) | Random non-optimal parameters do not have the predicted alignment; theorem is checked on a balanced representative |
| C2 | `repro/verify_all.py` → `repro/theorems.py:claim2_gn_decomp`; compact outer-product factors build within/cross/class components | [`claim2.json`](docs/evidence/faithful_run_63f3331/claim2.json) | Random features make the within term non-zero; the exact MSE three-way split is used |
| C3 | `repro/verify_all.py` → `repro/theorems.py:claim3_gradient`; JAX data-gradient is projected onto the theorem’s natural class-mean basis | [`claim3.json`](docs/evidence/faithful_run_63f3331/claim3.json) | A degenerate eigenspace admits other bases; the reported coefficients use the paper’s natural basis |
| C4 | `repro/verify_all.py` → `repro/theorems.py:claim4_gram`; intermediate `W_lᵀW_l` is compared with the closed-form eigenvalues and recurrence | [`claim4.json`](docs/evidence/faithful_run_63f3331/claim4.json) | Random weights are full rank; the last `K×d` layer has the analogous transposed statement |
| C5 | `repro/train.py:train_ufm(..., "linear")` → `repro/dynamics.py:linear_dynamics`; checkpoints are written to the dynamics CSV and plotted | [`claim5.json`](docs/evidence/faithful_run_63f3331/claim5.json), [`claim5_dynamics.csv`](docs/evidence/faithful_run_63f3331/claim5_dynamics.csv) | The run uses 30k epochs and reaches the analytic loss; it is a scoped dynamics reproduction, not a full-horizon rerun |
| C6 | `repro/train.py:train_ufm(..., "relu")` → `repro/dynamics.py:relu_hessian_topk` and `relu_gradient_coeffs` | [`claim6.json`](docs/evidence/faithful_run_63f3331/claim6.json) | CPU downscale to 150k epochs; exact non-zero count is basis-sensitive, so the gate tests separation and inequality |

The raw evidence was generated at faithful source commit
`63f3331909ff0d1832527e582d18263991d9a13e` and is mirrored in the public
[Hugging Face run](https://huggingface.co/spaces/DineshAI/RwiGcN2feP/tree/main/runs/63f3331).
The committed snapshot above keeps the claim results reviewable without relying
on the external Space.

## Reproduce

The canonical branch contains the faithful pipeline and its locked environment:

```bash
bash repro/run.sh
```

This runs `uv sync --frozen --no-progress` and then
`uv run python -m repro.verify_all`. It is CPU-only and writes regenerated files
under `outputs/`; the committed, reviewable snapshot is under
`docs/evidence/faithful_run_63f3331/`. The pinned dependency set is in
[`uv.lock`](uv.lock), and the interactive walkthrough is
[`notebooks/low_dim_spectra.py`](notebooks/low_dim_spectra.py).

## Branch and repository provenance

Before cleanup, the repository had two meaningful refs:

| Historical ref | Role | Decision |
|---|---|---|
| `master` (`0194fc0`) | Publication README, report, figures, and notebook; it still pointed at the older toy verifier and stale 5/6 gate | Content retained, stale verifier moved under `repro/legacy/` |
| `orx/baseline-faithful-spectra` (`63f3331`) | Faithful six-claim implementation, fixed `uv` command, and the raw Space run | Merged into canonical `main`; branch is no longer a supported reader surface |
| shared ancestor (`70e0213`) | Earlier toy K=2/D=5 verifier | Preserved only as `repro/legacy/`; not evidence for the six claims |

The faithful branch was merged before the repository cleanup so that the final
`main` branch contains the code used to produce the evidence. The old toy
verifier is deliberately labeled historical and is not used by the publication
gate.

## Citation

For the protocol reproduced here, cite the v1 source:

```bibtex
@article{garrod2024lowdimensional,
  author  = {Connall Garrod and Jonathan P. Keating},
  title   = {Unifying Low Dimensional Observations in Deep Learning Through the Deep Linear Unconstrained Feature Model},
  journal = {arXiv preprint arXiv:2404.06106},
  year    = {2024},
  doi     = {10.48550/arXiv.2404.06106},
  url     = {https://arxiv.org/abs/2404.06106v1}
}
```

If you use the revised ICML 2026 version, cite the current title shown on
[arXiv](https://arxiv.org/abs/2404.06106) and note that this repository pins the
earlier theorem numbering for reproducibility.

## Thank you

Thank you to **Connall Garrod** and **Jonathan P. Keating** for developing the
deep-UFM explanation of low-dimensional spectra and for making the paper’s
mathematical structure concrete enough to audit. The reproduction is intended
as a respectful, source-linked companion: it records the checks that pass, the
exact producer paths, and the compute/version limits that remain.

## Maintainer

Repository and publication-surface commits are maintained under the
**MachineLearning-Nerd** GitHub identity. This is an independent reproduction,
not the authors’ official implementation.
