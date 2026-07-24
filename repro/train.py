"""Gradient-descent training of the deep (linear or ReLU) UFM (arXiv 2404.06106).

Full-batch GD on the regularised MSE loss of Eq. (1), all parameters
(W_1..W_L, H1) trained jointly. JAX + lax.scan for speed. Used for the
empirical dynamics claims 5 (linear) and 6 (ReLU).
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _act(x, activation):
    return jax.nn.relu(x) if activation == "relu" else x


def _forward(Ws, H1, activation):
    F = H1
    for l in range(len(Ws) - 1):
        F = _act(Ws[l] @ F, activation)
    return Ws[-1] @ F


def train_ufm(K, d, n, L, lam_W, lam_H, activation, steps, lr, seed,
              ckpts=None):
    """Train and return final params + checkpointed params for analysis."""
    rng = np.random.default_rng(seed)
    Y = np.kron(np.eye(K), np.ones((1, n)))               # K x Kn
    sc = 1.0 / np.sqrt(d)                                  # variance-preserving normal init
    W0 = [rng.standard_normal((d, d)) * sc for _ in range(L - 1)] + \
         [rng.standard_normal((K, d)) * sc]
    H0 = rng.standard_normal((d, K * n)) * sc
    Yj = jnp.asarray(Y)

    def loss(Ws, H1):
        F = _forward(Ws, H1, activation)
        dl = 0.5 * jnp.sum((F - Yj) ** 2) / (K * n)
        reg = sum(0.5 * lam_W * jnp.sum(W ** 2) for W in Ws) + 0.5 * lam_H * jnp.sum(H1 ** 2)
        return dl + reg

    gloss = jax.grad(loss, (0, 1))

    def step(carry, _):
        Ws, H1 = carry
        gW, gH = gloss(Ws, H1)
        Ws = jax.tree_util.tree_map(lambda w, g: w - lr * g, Ws, gW)
        H1 = H1 - lr * gH
        return (Ws, H1), None

    Ws_j = [jnp.asarray(W) for W in W0]
    H1_j = jnp.asarray(H0)
    (Wsf, H1f), _ = jax.lax.scan(jax.jit(step), (Ws_j, H1_j), None, length=steps)

    if ckpts is None:
        ckpts = sorted(set([1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000,
                            10000, 25000, 50000, 100000, steps]))
    ckpts = [c for c in ckpts if c <= steps]
    hist = []
    # re-run prefix segments to snapshot params at each checkpoint
    Wc, Hc = [jnp.asarray(W) for W in W0], jnp.asarray(H0)
    prev = 0
    step_jit = jax.jit(step)
    for c in ckpts:
        (Wc, Hc), _ = jax.lax.scan(step_jit, (Wc, Hc), None, length=c - prev)
        hist.append((c, [np.asarray(W) for W in Wc], np.asarray(Hc)))
        prev = c
    return dict(Y=Y, final=([np.asarray(W) for W in Wsf], np.asarray(H1f)),
                history=hist, activation=activation)
