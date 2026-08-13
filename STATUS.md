# Reproduction status

## Decision

- **Scoped reproduction:** `VERIFIED` for all six claim checks at the pinned
  arXiv v1 UFM protocol.
- **Strict publication gate:** `NOT_PASSED`.
- **Reason:** the theorem checks use one balanced representative of the global
  optimum; Claim 5 stops at 30,000 epochs; Claim 6 stops at 150,000 ReLU epochs
  versus the paper’s 1,000,000; and the current arXiv v3 revision has a changed
  title and numbering.
- **Raw run:** faithful source commit `63f3331`, mirrored under
  `docs/evidence/faithful_run_63f3331/` and the linked Hugging Face run.

## Claim ledger

| Claim | Status | Producer | Evidence |
|---|---|---|---|
| C1 / v1 Thm. 4.1 | `VERIFIED_CONDITIONAL` | `repro/theorems.py:claim1_hessian`, `repro/hessian.py` | `claim1.json`; rank 9, ratio 1.00, formula/autodiff error 0 |
| C2 / v1 Thm. 4.2 | `VERIFIED_CONDITIONAL` | `repro/theorems.py:claim2_gn_decomp` | `claim2.json`; ranks 0/6/3, orthogonality error `5.8e−18` |
| C3 / v1 Thm. 4.3 | `VERIFIED_CONDITIONAL` | `repro/theorems.py:claim3_gradient` | `claim3.json`; 3 equal `1/3` coefficients, off-diagonal 0 |
| C4 / v1 Thm. 4.4 | `VERIFIED_CONDITIONAL` | `repro/theorems.py:claim4_gram` | `claim4.json`; rank 3, eigenvalue and recurrence errors 0 |
| C5 / v1 Figs. 3–4 | `REPRODUCED_SCOPED` | `repro/train.py`, `repro/dynamics.py:linear_dynamics` | `claim5.json` and `claim5_dynamics.csv`; ratio 2.12→1.00, `f_cc` 0.184→1.00 |
| C6 / v1 Fig. 9/Table 2 | `REPRODUCED_SCOPED` | `repro/train.py`, `repro/dynamics.py:relu_hessian_topk`, `relu_gradient_coeffs` | `claim6.json`; outlier ratio 4.98, coefficient ratio 31.1 |

## Reproduction protocol

```text
K=3, n=40, L=5, lambda_W=lambda_H=0.01, layer l=3
linear: d=60, seed=20240406, 30,000 epochs
ReLU: d=65, layer l=4, seed=20240407, 150,000 epochs
command: bash repro/run.sh
```

The analytic optimum has loss `0.16483787272447786` and full-gradient infinity
norm `4.8e−16`. The linear trajectory reaches the same loss. The ReLU result is
qualitative evidence, not an exact 1,000,000-epoch reproduction.

## Source/version boundary

The faithful implementation follows [arXiv v1](https://arxiv.org/abs/2404.06106v1),
where the checked results are Theorems 4.1–4.4. The current [arXiv record](https://arxiv.org/abs/2404.06106)
is v3 and has a revised title and renumbered sections. Any future v3 audit must
be tracked as a separate source pin rather than silently relabeling this run.

## Required evidence for a strict gate

1. Re-run the linear and ReLU experiments at the exact current paper protocol,
   including the full ReLU horizon, with raw checkpoints retained.
2. Audit whether the v3 theorem statements and numerical settings are identical
   to the v1 claims used here.
3. Preserve independent raw output and environment metadata for every run.
