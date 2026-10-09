"""Expected-F0.5 decision rule.

For one S1 entity with candidate probabilities p_1 >= ... >= p_n (treated as
independent), predicting the top-k gives TP = A ~ PoissonBinomial(p_1..p_k)
and FN = B + M, B ~ PoissonBinomial(p_{k+1}..p_n), M = true matches outside the
candidate list (optional Poisson prior with mean `miss`). F0.5 = 1.25 A /
(0.25 (A + B + M) + k); predicting nothing scores 1 only if A + B + M = 0.
The rule predicts the k with the highest expected F0.5.
"""
import numpy as np
import pandas as pd


def _pb(P):
    """Poisson-binomial pmf of the row-wise sum; P: (G, m) -> (G, m + 1)."""
    G, m = P.shape
    d = np.zeros((G, m + 1))
    d[:, 0] = 1.0
    for j in range(m):
        p = P[:, j:j + 1]
        d[:, 1:] = d[:, 1:] * (1 - p) + d[:, :-1] * p
        d[:, 0] *= (1 - p[:, 0])
    return d


def best_k(P, miss=0.0, max_extra=3):
    """P: (G, n) probabilities sorted descending per row. Returns best k per row."""
    G, n = P.shape
    if miss > 0:
        mk = np.arange(max_extra + 1)
        pm = np.exp(-miss) * miss ** mk / np.array([np.math.factorial(int(x)) for x in mk])
        pm[-1] += 1 - pm.sum()
    else:
        pm = np.array([1.0])
    best = np.zeros(G, dtype=np.int64)
    best_val = np.full(G, -1.0)
    for k in range(n + 1):
        DA = _pb(P[:, :k])            # (G, k+1)
        DB = _pb(P[:, k:])            # (G, n-k+1)
        a = np.arange(k + 1)[:, None, None]
        b = np.arange(n - k + 1)[None, :, None]
        mm = np.arange(len(pm))[None, None, :]
        if k == 0:
            f = ((b + mm) == 0).astype(float) * np.ones((1, 1, 1))
        else:
            f = 1.25 * a / (0.25 * (a + b + mm) + k)
        f = np.broadcast_to(f, (k + 1, n - k + 1, len(pm)))
        w = np.einsum("ga,gb,abm,m->g", DA, DB, f, pm)
        upd = w > best_val
        best[upd], best_val[upd] = k, w[upd]
    return best


def expected_f05_keep(code, p, miss=0.0):
    """Keep mask choosing, for every S1 (`code`), the expected-F0.5-optimal top-k."""
    d = pd.DataFrame({"c": code, "p": p, "i": np.arange(len(p))})
    d = d.sort_values(["c", "p"], ascending=[True, False], kind="stable")
    d["r"] = d.groupby("c").cumcount()
    d["n"] = d.groupby("c")["p"].transform("size")
    keep = np.zeros(len(p), dtype=bool)
    for n, g in d.groupby("n"):
        codes = g["c"].values.reshape(-1, n)
        P = g["p"].values.reshape(-1, n)
        k = best_k(P, miss)
        sel = g["r"].values.reshape(-1, n) < k[:, None]
        keep[g["i"].values.reshape(-1, n)[sel]] = True
        del codes
    return keep
