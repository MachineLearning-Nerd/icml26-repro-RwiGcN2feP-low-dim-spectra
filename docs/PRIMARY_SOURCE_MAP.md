# Primary source map

## Paper sources

| Purpose | Source |
|---|---|
| Exact source reproduced by the faithful code | [arXiv 2404.06106v1](https://arxiv.org/abs/2404.06106v1) |
| Current paper record and ICML 2026 revision | [arXiv 2404.06106](https://arxiv.org/abs/2404.06106) |
| Competition/OpenReview identifier | [OpenReview `RwiGcN2feP`](https://openreview.net/forum?id=RwiGcN2feP) |

The v1 HTML source is the authoritative source for the theorem labels used by
this repository: Theorems 4.1–4.4 cover the Hessian, Gauss–Newton decomposition,
gradient coefficients, and weight Gram structure. The v3 record is linked for
current metadata but is not silently substituted into this v1 reproduction.

## Code and evidence

| Evidence | Repository path or immutable source |
|---|---|
| Primary verifier | `repro/verify_all.py` at faithful source commit `63f3331` |
| Model and analytic optimum | `repro/ufm.py` |
| Theorem checks | `repro/theorems.py` |
| Dynamics | `repro/train.py` and `repro/dynamics.py` |
| Environment | `pyproject.toml` and `uv.lock` |
| Committed raw snapshot | `docs/evidence/faithful_run_63f3331/` |
| Public raw mirror | [Hugging Face run `63f3331`](https://huggingface.co/spaces/DineshAI/RwiGcN2feP/tree/main/runs/63f3331) |
| Historical rejected checker | `repro/legacy/verify_spectra_rejected.py` |

All claim statuses are scoped to the producer paths and evidence above. The
repository does not claim that a finite numerical run replaces the paper’s
proofs or proves the revised v3 protocol without a separate audit.
