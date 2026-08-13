"""Verifiers for Theorems 4.1-4.4 (deep linear UFM, arXiv 2404.06106).

Each function evaluates the EXACT claim at a constructed (machine-precision)
global optimum and returns structured evidence. Hessian/GN objects are formed
from the v-vectors {v_{icc'}} defined in Appendix B (faithful second moments);
ranks/eigenvalues come from the compact V V^T factorisation (no d^2 x d^2 matrix
needs to be materialised).
"""
from __future__ import annotations
import numpy as np
import ufm
import hessian as Hmod


def _rows_of_A(Ws, l):
    """Rows a_{c'}^{(l+1)} of A^{(l+1)} = W_L ... W_{l+1}  (K rows, each length d)."""
    A = Ws[l].copy()
    for i in range(l + 1, len(Ws)):
        A = Ws[i] @ A
    return A  # K x d


def _v_vectors(Ws, H1, l, K, n):
    """Return dict of the Appendix-B objects: v_{icc'}, v_{cc'}, v_c (each d^2 vectors)."""
    A = _rows_of_A(Ws, l)                 # K x d, rows a_{c'}^{(l+1)}
    Hl = H1.copy()
    for i in range(0, l - 1):
        Hl = Ws[i] @ Hl                   # h^{(l)} : d x Kn
    # v_{icc'} = a_{c'}^{(l+1)} (x) h_{ic}^{(l)}  in R^{d^2}
    v_icc = np.zeros((K, n, K, Hl.shape[0] * A.shape[1]))
    for c in range(K):
        hblock = Hl[:, c * n:(c + 1) * n]          # d x n  (samples of class c)
        for cp in range(K):
            a = A[cp]                               # d
            v_icc[c, :, cp, :] = np.kron(a[None, :], hblock.T)  # n x d^2
    v_cc = v_icc.mean(axis=1)                       # K x K x d^2  (mean over i)
    v_c = v_cc.mean(axis=1)                         # K x d^2      (mean over c')
    return dict(v_icc=v_icc, v_cc=v_cc, v_c=v_c, A=A, Hl=Hl)


# --------------------------------------------------------------------------------------
def claim1_hessian(Ws, H1, l, K, n, Y):
    """Theorem 4.1: rank(Hess_l)=K^2, all nonzero eigenvalues equal, eigenvectors
    mu_c^{(l+1)} (x) mu_c'^{(l)}; and kron formula == JAX autodiff Hessian."""
    P, Q = Hmod.kron_factors(Ws, H1, l, K, n)
    eigsP = np.linalg.eigvalsh(P); eigsQ = np.linalg.eigvalsh(Q)
    nzP = np.sort(eigsP[eigsP > 1e-7 * eigsP.max()])[::-1]
    nzQ = np.sort(eigsQ[eigsQ > 1e-7 * eigsQ.max()])[::-1]
    prods = np.sort(np.outer(nzP, nzQ).reshape(-1))[::-1]
    Hfull = Hmod.kron_hessian(Ws, H1, l, K, n)
    mus_l, _ = ufm.class_means(Ws, H1, K, n, l)
    mus_lp1, _ = ufm.class_means(Ws, H1, K, n, l + 1)
    fcc = np.zeros((K, K))
    for c in range(K):
        for cp in range(K):
            v = np.kron(mus_lp1[:, c], mus_l[:, cp])
            a = Hfull @ v
            fcc[c, cp] = (v @ a) ** 2 / ((v @ v) * (a @ a))
    diff, _, _ = Hmod.formula_vs_autodiff(Ws, H1, l, Y)
    return dict(rank_P=int(nzP.size), rank_Q=int(nzQ.size), n_nonzero=int(prods.size),
                eig_ratio=float(prods.max() / prods.min()) if prods.size else float("nan"),
                top_eigs=prods.tolist(), fcc_min=float(fcc.min()), fcc_max=float(fcc.max()),
                formula_vs_autodiff=diff)


def claim2_gn_decomp(Ws, H1, l, K, n):
    """Theorem 4.2: G_within rank 0, G_cross rank K(K-1), G_class rank K, equal
    nonzero eigenvalues, orthogonal images. Ranks via the compact V V^T factors."""
    V = _v_vectors(Ws, H1, l, K, n)
    v_icc, v_cc, v_c = V["v_icc"], V["v_cc"], V["v_c"]

    # G_class = sum_c v_c v_c^T  -> factor matrix Vc (d^2 x K)
    Vclass = v_c.T  # d^2 x K
    # G_cross,c = (1/K) sum_{c'} (v_{cc'} - v_c)(.)^T ; stack all differences
    diffs = (v_cc - v_c[:, None, :]).reshape(K * K, -1)  # (K*K) x d^2
    Vcross = diffs.T / np.sqrt(K)                         # d^2 x (K*K)
    # G_within,c,c' = (1/n) sum_i (v_{icc'} - v_{cc'})(.)^T
    within = (v_icc - v_cc[:, None, :, :]).reshape(K * n * K, -1)  # (K n K) x d^2
    Vwithin = within.T / np.sqrt(n)

    def rank_of(M):  # rank of M M^T via singular values of M, with absolute floor
        s = np.linalg.svd(M, compute_uv=False)
        smax = s.max() if s.size else 0.0
        thr = max(1e-7 * smax, 1e-9)        # absolute floor: a numerically-zero matrix -> rank 0
        return int(np.sum(s > thr)), s.tolist()

    rc, sc = rank_of(Vclass)
    rx, sx = rank_of(Vcross)
    rw, sw = rank_of(Vwithin)
    # orthogonality of images(G_class) and images(G_cross): project each cross
    # vector onto the class image and measure the relative energy (max cosine).
    Qc, _ = np.linalg.qr(Vclass)                       # d^2 x K, orthonormal class image
    norms = np.linalg.norm(Vcross, axis=0)
    nzcols = norms > 1e-12
    proj = Qc @ (Qc.T @ Vcross[:, nzcols])
    cos = np.linalg.norm(proj, axis=0) / norms[nzcols]
    ortho = float(cos.max() if cos.size else 0.0)
    # eigenvalues (nonzero) of G_class and G_cross via the compact factors
    eig_class = np.sort(np.linalg.eigvalsh(Vclass.T @ Vclass))[::-1]
    eig_cross = np.sort(np.linalg.eigvalsh(Vcross.T @ Vcross))[::-1]
    nz_c = eig_class[eig_class > 1e-7 * eig_class.max()]
    nz_x = eig_cross[eig_cross > 1e-7 * eig_cross.max()]
    return dict(rank_class=rc, rank_cross=rx, rank_within=rw,
                exp_class=K, exp_cross=K * (K - 1), exp_within=0,
                eig_class_ratio=float(nz_c.max() / nz_c.min()) if nz_c.size > 1 else 1.0,
                eig_cross_ratio=float(nz_x.max() / nz_x.min()) if nz_x.size > 1 else 1.0,
                class_cross_ortho=ortho,
                eig_class=nz_c.tolist(), eig_cross=nz_x.tolist())


def claim3_gradient(Ws, H1, l, K, n, Y):
    """Theorem 4.3: g~^{(l)} = (beta/K) sum_c mu_c^{(l+1)} (x) mu_c^{(l)}; exactly K
    equal non-zero coefficients (=1/K after normalisation) in the natural basis."""
    import jax, jax.numpy as jnp
    d = Ws[l - 1].shape[0]
    Wsj = [jnp.asarray(W) for W in Ws]; H1j = jnp.asarray(H1); Yj = jnp.asarray(Y)

    def data_loss(Wl_flat):
        Ws2 = list(Wsj); Ws2[l - 1] = Wl_flat.reshape(d, d)
        F = H1j
        for W in Ws2: F = W @ F
        return 0.5 * jnp.sum((F - Yj) ** 2) / (K * n)
    g = np.asarray(jax.grad(data_loss)(jnp.asarray(Ws[l - 1].reshape(-1))))  # d^2
    # g includes only the data term (we drop the reg part for g~)
    mus_l, _ = ufm.class_means(Ws, H1, K, n, l)
    mus_lp1, _ = ufm.class_means(Ws, H1, K, n, l + 1)
    # natural basis b_{c,c'} = mu_c^{(l+1)} (x) mu_c'^{(l)} ; normalise
    B = np.zeros((K, K, d * d))
    for c in range(K):
        for cp in range(K):
            v = np.kron(mus_lp1[:, c], mus_l[:, cp])
            B[c, cp] = v / np.linalg.norm(v)
    Bf = B.reshape(K * K, -1)
    # normalised alignment coefficients f~_{cc'} = |<b,g>|^2 / ||g||^2  (paper Eq.)
    coeffs = (np.abs(Bf @ g) ** 2) / float(g @ g)
    diag = np.array([coeffs[c * K + c] for c in range(K)])
    off = np.array([coeffs[c * K + cp] for c in range(K) for cp in range(K) if c != cp])
    return dict(coeff_matrix=coeffs.reshape(K, K).tolist(),
                diag_mean=float(diag.mean()), diag_ratio=float(diag.max() / diag.min()),
                offdiag_max=float(off.max()), n_nonzero=int(np.sum(coeffs > 0.02)),
                diag_target=1.0 / K)


def claim4_gram(Ws, H1, l, K, n, lam_W, lam_H):
    """Theorem 4.4: rank(W_l^T W_l)=K, nonzero eigenvalues = (lam_H/lam_Wl) n ||mu_c||^2,
    and recurrence lam_Wl W_l^T W_l = lam_W{l-1} W_{l-1} W_{l-1}^T."""
    Wl = Ws[l - 1]
    GtW = Wl.T @ Wl
    eigs = np.sort(np.linalg.eigvalsh(GtW))[::-1]
    rank = int(np.sum(eigs > 1e-7 * eigs[0]))
    mus, _ = ufm.class_means(Ws, H1, K, n, 1)   # mu_c (layer-1 / Hbar features)
    mu_norms_sq = np.sort((mus ** 2).sum(axis=0))[::-1]
    predicted = np.sort((lam_H / lam_W) * n * mu_norms_sq)[::-1]
    nz = eigs[:K]
    pred = predicted[:K]
    rel_err = float(np.max(np.abs(np.sort(nz) - np.sort(pred)) / np.abs(np.sort(pred).mean())))
    # recurrence check
    rec_err = None
    if l >= 2:
        lhs = lam_W * (Ws[l - 1].T @ Ws[l - 1])
        rhs = lam_W * (Ws[l - 2] @ Ws[l - 2].T)
        rec_err = float(np.max(np.abs(lhs - rhs)) / max(np.max(np.abs(lhs)), 1e-30))
    return dict(rank=rank, exp_rank=K, eigenvalues=nz.tolist(),
                predicted=pred.tolist(), eig_rel_err=rel_err, recurrence_err=rec_err,
                mu_norms_sq=mu_norms_sq.tolist())
