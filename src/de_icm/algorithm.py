"""DE-ICM placement probabilities (arXiv:2610.06028, Algorithm 1)."""

from __future__ import annotations

import numpy as np

# Section C. c = 20, window [x_L, x_R) = [-3.75, 27.5).
C_NORM = 20.0
X_L = -3.75
X_R = 27.5


def step_size(n: int) -> float:
    """Quadrature step from the field size (Section C)."""
    if n <= 50:
        return 1.0 / 8.0
    if n <= 200:
        return 1.0 / 16.0
    if n <= 1000:
        return 1.0 / 32.0
    return 1.0 / 64.0


def phi(x: np.ndarray) -> np.ndarray:
    """phi(x) = exp(x - exp(-x))."""
    return np.exp(x - np.exp(-x))


def phi_prime(x: np.ndarray) -> np.ndarray:
    """phi'(x) = (1 + exp(-x)) * phi(x)."""
    return (1.0 + np.exp(-x)) * phi(x)


def _nodes(h: float) -> np.ndarray:
    n_steps = int(round((X_R - X_L) / h))
    xs = X_L + h * np.arange(n_steps, dtype=np.float64)
    return xs[xs < X_R]


def _full_product(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Coefficients of prod_j (q_j + p_j z). C[r] is degree r."""
    coeff = np.array([1.0], dtype=np.float64)
    for qj, pj in zip(q, p):
        coeff = np.convolve(coeff, np.array([qj, pj], dtype=np.float64))
    return coeff


def de_icm(stacks, prizes=None, h=None):
    """Placement matrix Q and optional equities V.

    Q[i, r] is the probability that player i finishes in place r+1.
    Stacks are positive. If prizes is None, m = n and V is None.
    """
    s = np.asarray(stacks, dtype=np.float64)
    if s.ndim != 1 or s.size < 2:
        raise ValueError("stacks must be a 1d array of length at least 2")
    if np.any(s <= 0):
        raise ValueError("stacks must be positive")
    n = int(s.size)
    if prizes is None:
        m = n
        prize = None
    else:
        prize = np.asarray(prizes, dtype=np.float64)
        if prize.ndim != 1 or prize.size < 1 or prize.size > n:
            raise ValueError("prizes length must be between 1 and n")
        m = int(prize.size)

    if h is None:
        h = step_size(n)
    h = float(h)

    s_max = float(np.max(s))
    s_tilde = s / (C_NORM * s_max)
    order = np.argsort(s_tilde, kind="stable")
    s_sorted = s_tilde[order]

    xs = _nodes(h)
    t = phi(xs)
    w = h * phi_prime(xs)

    Q = np.zeros((n, m), dtype=np.float64)
    for ti, wi in zip(t, w):
        q = np.exp(-s_sorted * ti)
        p = -np.expm1(-s_sorted * ti)
        coeff = _full_product(q, p)
        # q is non-increasing in the sorted stacks, so the forward group is a prefix.
        b = int(np.searchsorted(-q, -0.5, side="right"))
        if b:
            alpha = p[:b] / q[:b]
            carried = np.zeros(b, dtype=np.float64)
            for r in range(m):
                carried = np.clip(coeff[r] - alpha * carried, 0.0, coeff[r])
                Q[:b, r] += wi * carried
        if b < n:
            qb = q[b:]
            pb = p[b:]
            carried = np.zeros(n - b, dtype=np.float64)
            for r in range(n, 0, -1):
                carried = np.clip((qb / pb) * (coeff[r] - carried), 0.0, coeff[r - 1])
                if r <= m:
                    Q[b:, r - 1] += wi * carried

    Q *= s_sorted[:, None]
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)
    Q = Q[inv]
    if prize is None:
        return Q, None
    return Q, Q @ prize
