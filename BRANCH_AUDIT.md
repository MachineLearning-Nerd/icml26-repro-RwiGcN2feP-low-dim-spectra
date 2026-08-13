# Branch audit

This audit records the repository state before the cleanup so that deleting
experiment refs does not erase provenance.

| Ref before cleanup | Tip before cleanup | Role | Final treatment |
|---|---|---|---|
| `master` | `0194fc02f4d2e14e0cd3a98e81b7c98b495810b3` | Publication README/report/figures/notebook; still linked to the old toy verifier and stale 5/6 gate | Merged content retained on `main`; stale toy code moved to `repro/legacy/` |
| `orx/baseline-faithful-spectra` | `63f3331909ff0d1832527e582d18263991d9a13e` | Faithful `uv`-locked six-claim pipeline and the source of the public raw run | Merged into `main`; raw run pinned in `docs/evidence/faithful_run_63f3331/` |
| shared ancestor | `70e0213` | Earlier toy K=2/D=5 verifier | Preserved as `repro/legacy/`; excluded from the gate |

The faithful ref diverged from the publication ref after the shared ancestor.
The merge was necessary because the old publication branch linked to files that
were not present on that branch. The final reader-facing surface is `main`; no
`orx/*` branch is required to reproduce the documented results.

## Historical verifier boundary

The former `repro/src/verify_spectra.py` reported a toy K=2/D=5 result and had a
serialization failure in one historical run. It is now named
`repro/legacy/verify_spectra_rejected.py`. Its output is not evidence for Claims
1–6 and is not referenced by the publication gate.
