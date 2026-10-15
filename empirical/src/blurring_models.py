from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import Ridge

EPS = 1e-12
REGIONS = ["Capsule 1", "Capsule 2", "Capsule 3", "Capsule 4"]


def closure(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    s = x.sum(axis=-1, keepdims=True)
    if x.ndim == 1:
        return x / s if float(s) > 0 else np.ones_like(x) / len(x)
    return np.divide(x, s, out=np.full_like(x, 1.0 / x.shape[1]), where=s > 0)


def replace_zeros(x: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Multiplicative zero replacement followed by closure.

    Each zero is replaced by a small fixed mass delta. Positive components are
    rescaled proportionally so the composition still sums to one. The chosen
    delta is a numerical assumption, not a biological abundance; sensitivity is
    evaluated in the workflow.
    """
    x = closure(np.asarray(x, dtype=float))
    if x.ndim == 1:
        z = x <= 0
        nz = int(z.sum())
        if nz == 0:
            return x
        total = nz * delta
        if total >= 1.0:
            raise ValueError("zero-replacement mass is too large")
        out = x.copy()
        out[z] = delta
        pos = ~z
        out[pos] *= (1.0 - total) / out[pos].sum()
        return closure(out)
    return np.vstack([replace_zeros(row, delta) for row in x])


def clr(x: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    y = replace_zeros(x, delta)
    logy = np.log(y)
    return logy - logy.mean(axis=-1, keepdims=True)


def inv_clr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return closure(e)


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(closure(a) - closure(b)).sum())


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(closure(a), closure(b), base=2.0))


def aitchison_distance(a: np.ndarray, b: np.ndarray, delta: float = 1e-6) -> float:
    return float(np.linalg.norm(clr(a, delta) - clr(b, delta)))


def softmax(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    e = np.exp(theta - np.max(theta))
    return e / e.sum()


def normalized_sample_weight(sample_weight: np.ndarray | None, n: int) -> np.ndarray:
    """Return nonnegative weights normalized to mean one."""
    if sample_weight is None:
        return np.ones(n, dtype=float)
    w = np.asarray(sample_weight, dtype=float)
    if w.shape != (n,) or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid sample weights")
    return w * (n / w.sum())


def _simplex_optimize(objective, n: int) -> np.ndarray:
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 5000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"simplex optimization failed: {res.message}")
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def arithmetic_reference(stool_train: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
    stool_train = np.asarray(stool_train, dtype=float)
    w = normalized_sample_weight(sample_weight, len(stool_train))
    return closure(np.average(stool_train, axis=0, weights=w))


def aitchison_reference(stool_train: np.ndarray, delta: float, sample_weight: np.ndarray | None = None) -> np.ndarray:
    stool_train = np.asarray(stool_train, dtype=float)
    w = normalized_sample_weight(sample_weight, len(stool_train))
    return inv_clr(np.average(clr(stool_train, delta), axis=0, weights=w))


def equal_arithmetic(regions: np.ndarray) -> np.ndarray:
    # regions shape: n x 4 x G
    return closure(np.mean(regions, axis=1))


def equal_aitchison(regions: np.ndarray, delta: float) -> np.ndarray:
    z = clr(regions.reshape(-1, regions.shape[-1]), delta).reshape(regions.shape)
    return inv_clr(np.mean(z, axis=1))


def fit_arithmetic_weights(regions: np.ndarray, stool: np.ndarray, reference: np.ndarray | None = None, sample_weight: np.ndarray | None = None) -> np.ndarray:
    regions = np.asarray(regions, dtype=float)
    stool = np.asarray(stool, dtype=float)
    event_weight = normalized_sample_weight(sample_weight, len(stool))
    components = [regions[:, j, :] for j in range(regions.shape[1])]
    if reference is not None:
        components.append(np.tile(reference, (len(stool), 1)))

    def objective(w: np.ndarray) -> float:
        pred = sum(w[j] * components[j] for j in range(len(components)))
        pred = closure(pred)
        event_error = np.sum((pred - stool) ** 2, axis=1)
        return float(np.average(event_error, weights=event_weight))

    return _simplex_optimize(objective, len(components))


def predict_arithmetic(regions: np.ndarray, weights: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    components = [regions[:, j, :] for j in range(regions.shape[1])]
    if reference is not None:
        components.append(np.tile(reference, (len(regions), 1)))
    pred = sum(weights[j] * components[j] for j in range(len(components)))
    return closure(pred)


def fit_aitchison_weights(regions: np.ndarray, stool: np.ndarray, delta: float, reference: np.ndarray | None = None, sample_weight: np.ndarray | None = None) -> np.ndarray:
    regions = np.asarray(regions, dtype=float)
    stool = np.asarray(stool, dtype=float)
    event_weight = normalized_sample_weight(sample_weight, len(stool))
    z_regions = clr(regions.reshape(-1, regions.shape[-1]), delta).reshape(regions.shape)
    z_stool = clr(stool, delta)
    components = [z_regions[:, j, :] for j in range(regions.shape[1])]
    if reference is not None:
        components.append(np.tile(clr(reference, delta), (len(stool), 1)))

    def objective(w: np.ndarray) -> float:
        zpred = sum(w[j] * components[j] for j in range(len(components)))
        event_error = np.sum((zpred - z_stool) ** 2, axis=1)
        return float(np.average(event_error, weights=event_weight))

    return _simplex_optimize(objective, len(components))


def predict_aitchison(regions: np.ndarray, weights: np.ndarray, delta: float, reference: np.ndarray | None = None) -> np.ndarray:
    z_regions = clr(regions.reshape(-1, regions.shape[-1]), delta).reshape(regions.shape)
    components = [z_regions[:, j, :] for j in range(regions.shape[1])]
    if reference is not None:
        components.append(np.tile(clr(reference, delta), (len(regions), 1)))
    zpred = sum(weights[j] * components[j] for j in range(len(components)))
    return inv_clr(zpred)


@dataclass
class RidgeFit:
    model: Ridge
    alpha: float
    delta: float


def region_clr_design(regions: np.ndarray, delta: float) -> np.ndarray:
    z = clr(regions.reshape(-1, regions.shape[-1]), delta).reshape(regions.shape)
    return z.reshape(len(regions), -1)


def fit_ridge(regions: np.ndarray, stool: np.ndarray, alpha: float, delta: float, sample_weight: np.ndarray | None = None) -> RidgeFit:
    X = region_clr_design(regions, delta)
    Y = clr(stool, delta)
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    event_weight = normalized_sample_weight(sample_weight, len(stool))
    model.fit(X, Y, sample_weight=event_weight)
    return RidgeFit(model=model, alpha=float(alpha), delta=float(delta))


def predict_ridge(fit: RidgeFit, regions: np.ndarray) -> np.ndarray:
    X = region_clr_design(regions, fit.delta)
    return inv_clr(fit.model.predict(X))


def select_ridge_alpha(
    regions: np.ndarray,
    stool: np.ndarray,
    subjects: np.ndarray,
    alphas: Sequence[float],
    delta: float,
) -> Tuple[float, List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    unique = np.unique(subjects)
    for alpha in alphas:
        subject_errors = []
        for held in unique:
            tr = subjects != held
            te = subjects == held
            train_subjects = np.asarray(subjects)[tr]
            train_weight = np.array([1.0 / np.sum(train_subjects == s) for s in train_subjects])
            fit = fit_ridge(regions[tr], stool[tr], alpha, delta, sample_weight=train_weight)
            pred = predict_ridge(fit, regions[te])
            errs = [aitchison_distance(y, p, delta) for y, p in zip(stool[te], pred)]
            subject_errors.append(float(np.mean(errs)))
        rows.append({"alpha": float(alpha), "inner_mean_subject_aitchison": float(np.mean(subject_errors))})
    best = min(rows, key=lambda r: (r["inner_mean_subject_aitchison"], -r["alpha"]))
    return float(best["alpha"]), rows


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d) & (np.abs(d) > 1e-15)]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(float(np.mean(d)))
    if n <= 20:
        vals = []
        for mask in range(1 << n):
            signs = np.array([1.0 if (mask >> j) & 1 else -1.0 for j in range(n)])
            vals.append(abs(float(np.mean(signs * d))))
        return float(np.mean(np.asarray(vals) >= obs - 1e-15))
    rng = np.random.default_rng(20260723)
    vals = []
    for _ in range(200000):
        vals.append(abs(float(np.mean(rng.choice([-1.0, 1.0], size=n) * d))))
    return float(np.mean(np.asarray(vals) >= obs - 1e-15))


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    out = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        out[idx] = min(running, 1.0)
    return out
