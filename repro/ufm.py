"""Deep linear / ReLU Unconstrained Features Model (UFM) for arXiv 2404.06106.

Faithful implementation of the model in Eq. (1):
    L(H1, W1..WL) = (1/2Kn)||W_L ... W_1 H1 - Y||_F^2
                    + sum_l (lam_Wl/2)||W_l||_F^2 + (lam_H1/2)||H1||_F^2
with Y = I_K (x) 1_n^T  (balanced K classes, n samples each), W_L in R^{Kxd},
W_{L-1}..W_1 in R^{dxd}, H1 in R^{dxKn}.

All quantities are plain numpy. JAX is used only for the *independent* autodiff
Hessian in hessian.py (to validate the Kronecker formula without circularity).
"""
from __future__ import annotations
import numpy as np


def make_Y(K: int, n: int) -> np.ndarray:
    """Label matrix Y = I_K (x) 1_n^T in R^{K x Kn} (column of class c is e_c)."""
    return np.kron(np.eye(K), np.ones((1, n)))


def forward(Ws, H1):
    """End-to-end linear map applied to features: W_L ... W_1 H1."""
    F = H1
    for W in Ws:
        F = W @ F
    return F


def data_loss(Ws, H1, Y):
    """Unregularised MSE term (1/2Kn)||W_L...W_1 H1 - Y||_F^2 (paper's Hessian defn)."""
    Kn = Y.shape[1]
    return 0.5 * np.sum((forward(Ws, H1) - Y) ** 2) / Kn


def reg_loss(Ws, H1, lam_W, lam_H):
    s = 0.0
    for W in Ws:
        s += 0.5 * lam_W * np.sum(W ** 2)
    s += 0.5 * lam_H * np.sum(H1 ** 2)
    return s


def total_loss(Ws, H1, Y, lam_W, lam_H):
    return data_loss(Ws, H1, Y) + reg_loss(Ws, H1, lam_W, lam_H)


def reg_condition(lam_W, lam_H, K, n, L):
    """Eq. (2): returns (lhs, rhs); condition holds iff lhs < rhs."""
    lhs = (K * n * (lam_W ** L) * lam_H) ** (1.0 / L)
    rhs = (1.0 / (K * L ** 2)) * (L - 1) ** ((L - 1) / L)
    return lhs, rhs


# --------------------------------------------------------------------------------------
# DNC quantities (section 4 definitions): A^{(l)}, h_{ic}^{(l)}, mu_c^{(l)}, mu_G^{(l)}
# --------------------------------------------------------------------------------------
def A_layer(Ws, l):
    """A^{(l)} = W_L W_{L-1} ... W_l  in R^{K x d}  (l in 1..L)."""
    A = Ws[l - 1]
    for i in range(l, len(Ws)):
        A = Ws[i] @ A
    return A


def features_at_layer(Ws, H1, l):
    """h^{(l)} = W_{l-1} ... W_1 H1  (features at the input of layer l). l in 1..L."""
    H = H1
    for i in range(0, l - 1):
        H = Ws[i] @ H
    return H


def class_means(Ws, H1, K, n, l):
    """mu_c^{(l)} = Av_i{h_{ic}^{(l)}} and global mean, for layer l. Returns (Hbar, muG).

    Hbar = [mu_1^{(l)},...,mu_K^{(l)}] in R^{d x K}; columns are the per-class means
    of the layer-l features.
    """
    H = features_at_layer(Ws, H1, l)  # d x Kn
    d = H.shape[0]
    mus = np.zeros((d, K))
    for c in range(K):
        cols = H[:, c * n:(c + 1) * n]
        mus[:, c] = cols.mean(axis=1)
    muG = mus.mean(axis=1)
    return mus, muG


# --------------------------------------------------------------------------------------
# Analytic global optimum of the deep linear UFM (DNC balanced solution).
# We restrict to the K-dim class-mean subspace (the global minimiser lives there) and
# solve the 3-variable problem for the balanced scaling (a,b,s) to machine precision.
# --------------------------------------------------------------------------------------
def analytic_optimum(d, K, n, L, lam_W, lam_H):
    """Construct the global optimum satisfying DNC1-3 (machine precision).

    The balanced DNC optimum has every separated layer sharing one singular
    value sigma (DNC4's recursion lam_W W_l^T W_l = lam_W W_{l-1} W_{l-1}^T
    forces equal singular values across layers). With class means mu_c = s e_c
    on an equal-norm orthogonal frame U = [e_1..e_K]:
        W_l = sigma (U U^T)   (l=1..L-1),   W_L = sigma U^T,   H1* = s U (x) 1_n^T,
        gamma = sigma^L s = end-to-end gain on the class subspace.
    The first-order conditions reduce to one equation in gamma in (0,1):
        gamma^{L-1} (1-gamma)^{L+1} = (lam_W K)^L (lam_H K n)
    whose root nearest 1 is the non-trivial global minimum (lowest data loss);
    sigma^2 = (1-gamma)gamma/(lam_W K), s^2 = (1-gamma)gamma/(lam_H K n).
    GD from a sufficient random init converges to this same point (validated).
    """
    from scipy.optimize import brentq
    U = np.zeros((d, K))
    U[:K, :] = np.eye(K)  # first K standard basis vectors

    lW_K = lam_W * K            # lambda_W * K
    lH_Kn = lam_H * K * n       # lambda_H * K * n
    rhs = (lW_K ** L) * lH_Kn

    def Feq(g):
        return (g ** (L - 1)) * ((1.0 - g) ** (L + 1)) - rhs

    gs = np.linspace(1e-5, 1 - 1e-5, 200000)
    vals = Feq(gs)
    brackets = [(gs[i], gs[i + 1]) for i in range(len(gs) - 1)
                if vals[i] == 0 or vals[i] * vals[i + 1] < 0]
    a_lo, a_hi = max(brackets, key=lambda t: t[1])     # root nearest 1 (global min)
    gamma = brentq(Feq, a_lo, a_hi, xtol=1e-15, rtol=1e-15, maxiter=200)
    sigma = np.sqrt((1 - gamma) * gamma / lW_K)
    s = np.sqrt((1 - gamma) * gamma / lH_Kn)
    a = b = sigma
    P = U @ U.T  # projector onto class-mean subspace
    Ws = [a * P.copy() for _ in range(L - 1)] + [b * U.T.copy()]
    # H1* = Hbar* (x) 1_n^T : block column c is s*mu_c repeated n times -> d x Kn
    H1 = np.kron(s * U, np.ones((1, n)))
    gamma_chk = b * a ** (L - 1) * s
    meta = dict(a=a, b=b, s=s, gamma=gamma_chk, fun=float(ufm_total_loss_chk(Ws, H1, K, n, lam_W, lam_H)),
                grad_inf=None, U=U, P=P)
    return Ws, H1, meta


def ufm_total_loss_chk(Ws, H1, K, n, lam_W, lam_H):
    Y = make_Y(K, n)
    return total_loss(Ws, H1, Y, lam_W, lam_H)
