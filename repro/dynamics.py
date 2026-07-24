"""Dynamics analysis for claims 5 (linear) and 6 (ReLU): layer-wise Hessian
spectrum, the f_{cc'} eigenvector-alignment metric, and gradient coefficients,
tracked over GD training checkpoints.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import ufm
import hessian as Hmod


def _factors(Ws, H1, l, K, n):
    return Hmod.kron_factors(Ws, H1, l, K, n)   # (P, Q) for the linear Hessian


def fcc_from_factors(P, Q, mus_lp1, mus_l, K):
    """f_{cc'} = |v^T H v|^2 / (||v||^2 ||Hv||^2), v = mu_c^{(l+1)} (x) mu_c'^{(l)}.
    Row-major identity: (P (x) Q) vec(M) = vec(P M Q^T),  M = mu_c mu_c'^T."""
    out = np.zeros((K, K))
    for c in range(K):
        for cp in range(K):
            M = np.outer(mus_lp1[:, c], mus_l[:, cp])
            PMQ = P @ M @ Q.T
            vtHv = float(np.sum(M * PMQ))
            nv2 = float(np.sum(M * M))
            nHv2 = float(np.sum(PMQ * PMQ))
            out[c, cp] = (vtHv ** 2) / (nv2 * nHv2) if nHv2 > 0 else 0.0
    return out


def spectrum_from_factors(P, Q):
    """All d^2 eigenvalues of the linear layer-wise Hessian = pairwise products of
    the eigenvalues of P and Q (Kronecker property)."""
    eP = np.linalg.eigvalsh(P)
    eQ = np.linalg.eigvalsh(Q)
    return np.sort(np.outer(eP, eQ).reshape(-1))[::-1]


def linear_dynamics(history, l, K, n, Y):
    """Per-checkpoint: DNC1 metric m, top-K^2 outliers vs bulk, f_cc alignment."""
    rows = []
    for step, Ws, H1 in history:
        P, Q = _factors(Ws, H1, l, K, n)
        eigs = spectrum_from_factors(P, Q)
        top9 = eigs[:K * K]
        bulk = eigs[K * K:]
        mus_l, _ = ufm.class_means(Ws, H1, K, n, l)
        mus_lp1, _ = ufm.class_means(Ws, H1, K, n, l + 1)
        fcc = fcc_from_factors(P, Q, mus_lp1, mus_l, K)
        # DNC1 metric m = ||Sigma_W Sigma_B^+||_F^2 (small => within-class collapse)
        Hl = H1
        for i in range(0, l - 1):
            Hl = Ws[i] @ Hl
        ddim = Hl.shape[0]
        Cn = np.eye(n) - np.ones((n, n)) / n   # centering matrix
        Sw = np.zeros((ddim, ddim))
        Sb = np.zeros((ddim, ddim))
        for c in range(K):
            blk = Hl[:, c * n:(c + 1) * n]              # d x n
            Sw += blk @ Cn @ blk.T / (K * n)
        muG = Hl.reshape(ddim, K, n).mean(axis=(1, 2))
        for c in range(K):
            blk = Hl[:, c * n:(c + 1) * n]
            muc = blk.mean(axis=1)
            Sb += np.outer(muc - muG, muc - muG) / K
        m = float(np.linalg.norm(Sw @ np.linalg.pinv(Sb), "fro") ** 2)
        rows.append(dict(step=int(step), top_eigs=top9.tolist(),
                         bulk_max=float(bulk.max()) if bulk.size else 0.0,
                         top_ratio=float(top9.max() / max(top9.min(), 1e-30)),
                         fcc_mean=float(fcc.mean()), fcc_min=float(fcc.min()),
                         fcc=fcc.tolist(), dnc1_m=m))
    return rows


def relu_hessian_topk(Ws_np, H1_np, l, Y_np, k):
    """Top-k eigenpairs of the ReLU layer-wise Hessian via dense JAX autodiff Hessian
    + scipy eigsh. (Piecewise-linear => E_l=0, so this is the GN term G_l.)"""
    from scipy.sparse.linalg import eigsh
    d = Ws_np[l - 1].shape[0]
    K, Kn = Y_np.shape
    n = Kn // K
    Ws_j = [jnp.asarray(W) for W in Ws_np]; H1_j = jnp.asarray(H1_np); Y_j = jnp.asarray(Y_np)

    def data_loss(wl_flat):
        wl = wl_flat.reshape(d, d)
        Ws = list(Ws_j); Ws[l - 1] = wl
        F = H1_j
        for i in range(len(Ws) - 1):
            F = jax.nn.relu(Ws[i] @ F)
        F = Ws[-1] @ F
        return 0.5 * jnp.sum((F - Y_j) ** 2) / (K * n)

    Hmat = np.asarray(jax.hessian(data_loss)(jnp.asarray(Ws_np[l - 1].reshape(-1))))  # d^2 x d^2
    kk = min(k, Hmat.shape[0] - 2)
    vals, vecs = eigsh(Hmat, k=kk, which="LA")
    idx = np.argsort(-vals)
    return vals[idx], vecs[:, idx]


def grad_data(Ws_np, H1_np, l, Y_np, activation):
    """g~^{(l)} = Av{(A^{(l+1)T} u) (x) h^{(l)}} (data-term gradient, flattened)."""
    d = Ws_np[l - 1].shape[0]
    K, Kn = Y_np.shape
    n = Kn // K
    Ws_j = [jnp.asarray(W) for W in Ws_np]; H1_j = jnp.asarray(H1_np); Y_j = jnp.asarray(Y_np)

    def dl(wl_flat):
        Ws = list(Ws_j); Ws[l - 1] = wl_flat.reshape(d, d)
        F = H1_j
        for i in range(len(Ws) - 1):
            F = jax.nn.relu(Ws[i] @ F) if activation == "relu" else Ws[i] @ F
        F = Ws[-1] @ F
        return 0.5 * jnp.sum((F - Y_j) ** 2) / (K * n)
    return np.asarray(jax.grad(dl)(jnp.asarray(Ws_np[l - 1].reshape(-1))))


def relu_gradient_coeffs(Ws, H1, l, K, n, Y):
    """Coefficients of g~ in the TRUE top-K^2 eigenbasis of the ReLU Hessian
    (Table 2): |<u_i, g>|^2 / ||g||^2 over the K^2 leading eigenvectors."""
    vals, vecs = relu_hessian_topk(Ws, H1, l, Y, K * K)
    g = grad_data(Ws, H1, l, Y, "relu")
    coeffs = (vecs.T @ g) ** 2 / float(g @ g)
    idx = np.argsort(-coeffs)
    return coeffs[idx].tolist(), vals[idx].tolist(), int(np.sum(np.array(coeffs) > 0.02 * coeffs.max()))
