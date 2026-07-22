"""Clean-room low-dimensional spectra from "Unifying Low Dimensional Spectra in Deep Learning"
(arXiv 2404.06106). numpy, CPU. Deep linear unconstrained features model (UFM).
c1: layer-wise Hessian rank K^2, equal eigenvalues. c4: Gram matrix rank K.
"""
from __future__ import annotations
import numpy as np


def ufm_weights(L, d, K, seed=0):
    """Generate L-layer deep linear UFM weights. W_l: d x d, rank K.
    Optimal weights: W_l = (optimal direction) with rank K."""
    rng = np.random.default_rng(seed)
    # optimal solution: W_l = U @ V^T where U,V have K columns
    U = rng.standard_normal((d, K)); V = rng.standard_normal((d, K))
    Ws = [U @ V.T / L] * L  # balanced across layers
    return Ws


def layer_hessian(W_l, X, Y):
    """Compute the Hessian of the loss w.r.t. layer weights W_l for the UFM.
    For linear model f = W_L ... W_1 x, loss = 0.5||Y - F X||^2."""
    d = W_l.shape[0]; n = X.shape[0]
    # Hessian = X^T X ⊗ I_d (Kronecker structure for linear networks)
    XtX = X.T @ X / n
    H = np.kron(XtX, np.eye(d))
    return H


def gram_matrix(W_l):
    """W_l^T W_l — the Gram matrix of layer weights."""
    return W_l.T @ W_l


def gradient_directions(Ws, X, Y):
    """Compute the K effective gradient directions (out of K^2 possible)."""
    L = len(Ws); d = Ws[0].shape[0]; K = min(Ws[0].shape)
    # forward pass
    F = np.eye(d)
    for W in Ws:
        F = F @ W
    # gradient at each layer
    residual = F @ X.T - Y.T  # d x n
    directions = []
    for l in range(L):
        # left factor: product of Ws before l
        left = np.eye(d)
        for i in range(l):
            left = left @ Ws[i]
        # right factor: product of Ws after l
        right = np.eye(d)
        for i in range(l + 1, L):
            right = right @ Ws[i]
        grad = left.T @ residual @ X @ right.T / X.shape[0]
        # SVD to get top-K directions
        U, S, Vt = np.linalg.svd(grad, full_matrices=False)
        for k in range(min(K, len(S))):
            if S[k] > 1e-8:
                directions.append((U[:, k], Vt[k], S[k]))
    return directions
