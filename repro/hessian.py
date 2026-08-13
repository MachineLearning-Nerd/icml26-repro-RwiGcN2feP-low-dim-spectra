"""Layer-wise Hessian of the deep linear UFM (arXiv 2404.06106, section 4.1).

Two INDEPENDENT computations of Hess_l = d^2(data_loss)/d W_l^2 (d^2 x d^2):

  (1) kron_formula: Hess_l = A^{(l+1)T} A^{(l+1)} (x) [Av_ic{ h_ic^{(l)} h_ic^{(l)T} }]
      (paper's Eq., derived in Appendix A).
  (2) autodiff     : exact JAX second derivative of the MSE data loss w.r.t. W_l,
      knowing nothing about the Kronecker structure.

Cross-checking (1)==(2) at *arbitrary* (non-optimal) points validates the formula
without circularity; at the optimum we then trust either to report the spectrum.

Flattening: w_l[a*d+b] = (W_l)_{ab} (numpy row-major). Under this convention
Hess = np.kron(P, Q) with P = A^{(l+1)T}A^{(l+1)}, Q = Av{h^{(l)}h^{(l)T}.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _factors_np(Ws, H1, l, K, n):
    """Return (P, Q): the two Kronecker factors P=A^{(l+1)T}A^{(l+1)}, Q=Av{hh^T}.

    A^{(l+1)} = W_L W_{L-1} ... W_{l+1}  (K x d); built left-to-right as
    Ws[L-1] @ ... @ Ws[l]  (0-indexed).
    """
    A = Ws[l].copy()                       # W_{l+1}  (0-indexed)
    for i in range(l + 1, len(Ws)):        # left-multiply W_{l+2} .. W_L
        A = Ws[i] @ A
    Hl = H1.copy()
    for i in range(0, l - 1):              # W_{l-1} ... W_1 H1
        Hl = Ws[i] @ Hl
    P = A.T @ A                       # d x d
    Q = (Hl @ Hl.T) / (K * n)        # d x d  (Av_ic of h h^T)
    return P, Q


def kron_hessian(Ws, H1, l, K, n):
    """Full d^2 x d^2 layer-wise Hessian via the Kronecker formula (for small d)."""
    P, Q = _factors_np(Ws, H1, l, K, n)
    return np.kron(P, Q)


def kron_factors(Ws, H1, l, K, n):
    """Return the two factors (cheap; spectrum obtained from their eigendecomps)."""
    return _factors_np(Ws, H1, l, K, n)


def autodiff_hessian(Ws_np, H1_np, l, Y_np):
    """Exact JAX Hessian of the MSE data loss w.r.t. flattened W_l."""
    d = Ws_np[l - 1].shape[0]
    K = Y_np.shape[0]
    n = Y_np.shape[1] // K

    Ws_j = [jnp.asarray(W) for W in Ws_np]
    H1_j = jnp.asarray(H1_np)
    Y_j = jnp.asarray(Y_np)

    def data_loss_wl(wl_flat):
        wl = wl_flat.reshape(d, d)
        Ws = list(Ws_j)
        Ws[l - 1] = wl
        F = H1_j
        for W in Ws:
            F = W @ F
        return 0.5 * jnp.sum((F - Y_j) ** 2) / (K * n)

    H = jax.hessian(data_loss_wl)(jnp.asarray(Ws_np[l - 1].reshape(-1)))
    return np.asarray(H)


def formula_vs_autodiff(Ws, H1, l, Y, atol=1e-9):
    """Max-abs difference between the Kronecker formula and the autodiff Hessian."""
    K, Kn = Y.shape
    n = Kn // K
    H_kron = kron_hessian(Ws, H1, l, K, n)
    H_auto = autodiff_hessian(Ws, H1, l, Y)
    return float(np.max(np.abs(H_kron - H_auto))), H_kron, H_auto
