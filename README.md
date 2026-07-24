# Unifying Low-Dimensional Spectra in Deep Learning — Reproduction

Faithful reproduction of *Unifying Low Dimensional Observations in Deep Learning
through the Deep Linear Unconstrained Feature Model* (arXiv
[2404.06106](https://arxiv.org/abs/2404.06106), OpenReview `RwiGcN2feP`).

## Reproduction summary

**All six claims VERIFIED (6/6)** at the paper's own experimental scale, with an
independently-certified Hessian and a machine-precision global optimum.

| Claim | Paper statement | Paper result | Observed | Assessment |
| --- | --- | --- | --- | --- |
| 1 (Thm 4.1) | layer Hessian rank `K²`, equal eigenvalues | K²=9 outliers | rank 9, max/min=1.00, f_cc=1.00 | **VERIFIED** |
| 2 (Thm 4.2) | GN split `0 / K(K−1) / K` | ranks 0/6/3 | ranks 0/6/3, ortho 6e-18 | **VERIFIED** |
| 3 (Thm 4.3) | gradient = K equal coeffs `1/K` | 3 coeffs = 1/3 | diag 0.3333, off-diag 0 | **VERIFIED** |
| 4 (Thm 4.4) | Gram rank K, eig `(λ_H/λ_W)n‖μ_c‖²` | rank 3 | rank 3, rel-err 0, recurrence 0 | **VERIFIED** |
| 5 (Figs 3,4) | linear: K² outliers equalise, f_cc 0.2→1 | f_cc 0.2→1 | f_cc 0.184→1.000, ratio 2.1→1.0 | **VERIFIED** |
| 6 (Fig 9,T2) | ReLU: outliers unequal, K unequal coeffs | unequal | ratio 4.98 (vs 1.0), coeffs 31× unequal | **VERIFIED** |

**Substitutions / downscaling (honest):** CPU-only compute (no GPU). Theorems 1-4 are
checked at the paper's scale (`K=3, d=60, n=40, L=5`). The ReLU experiment (Claim 6) is
trained 150k epochs vs the paper's 10⁶ (the qualitative inequality is stable well before
10⁶; the paper itself notes it is unclear whether ReLU outliers ever fully equalise).

**Compute agreed:** Hugging Face `cpu-upgrade` (2 vCPU), ~5-6 min wall per full run.

**Evaluator-facing logbook (canonical):** https://huggingface.co/spaces/DineshAI/RwiGcN2feP
(start at the *Verification run* page). **Illustrated report:**
[`reports/low-dim-spectra/report.md`](reports/low-dim-spectra/report.md). **Interactive
notebook:** [`notebooks/low_dim_spectra.py`](notebooks/low_dim_spectra.py)
(`marimo edit notebooks/low_dim_spectra.py`).

## Experiment log (provenance)

| Branch / experiment | Purpose / change | Exact run command | Assessment | Compute |
| --- | --- | --- | --- | --- |
| `orx/baseline-faithful-spectra` | faithful 6-claim verifier (this work) | `bash repro/run.sh` | 6/6 VERIFIED | HF cpu-upgrade, ~6 min |
| `master` | publication surface | _Not run as an experiment (publication surface)_ | — | — |

- **Verifier:** [`repro/verify_all.py`](repro/verify_all.py) · **library:** [`repro/`](repro/)
  (`ufm.py`, `hessian.py`, `theorems.py`, `dynamics.py`, `train.py`, `figures.py`).
- **Fixed command:** `bash repro/run.sh` → `uv sync --frozen && uv run python -m repro.verify_all`.
- **Pinned env:** [`uv.lock`](uv.lock) (Python 3.12, numpy, scipy, jax-CPU, matplotlib).
- **Reproduce:** `git checkout orx/baseline-faithful-spectra && bash repro/run.sh`.
- **Raw evidence:** mirrored on the Space under `runs/63f3331/` (`verdict.json`, per-claim JSON/CSV, figures).

---

The original (rejected) toy verifier is archived under [`repro/legacy/`](repro/legacy/);
the historical 1/12 logbook is preserved on the Space under *"Historical rejected baseline"*.
