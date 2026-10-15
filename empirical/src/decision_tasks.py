from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from blurring_models import closure

EPS = 1e-12

TASKS: Dict[str, Dict[str, object]] = {
    "proximal_local_burden": {
        "label": "Proximal-local oxygen-tolerant/sulfidogenic state proxy",
        "guilds": ["G01_sulfidogenic_bile_proteolytic", "G02_oxygen_tolerant_proximal_oral"],
        "guilds_by_resolution": {
            "K11": ["G01_sulfidogenic_bile_proteolytic", "G02_oxygen_tolerant_proximal_oral"],
            "K6": ["K6_G1_sulfidogenic_proximal"],
        },
        "target": "Capsule 1",
        "summary": "local",
        "direction": "high",
        "interpretation": "A high-state proxy defined from the intended proximal capsule profile.",
    },
    "distal_local_butyrate_deficit": {
        "label": "Distal-local low butyrate-associated state proxy",
        "guilds": ["G07_ruminococcaceae_fiber_butyrate", "G09_lachnospiraceae_butyrate_crossfeeders"],
        "guilds_by_resolution": {
            "K11": ["G07_ruminococcaceae_fiber_butyrate", "G09_lachnospiraceae_butyrate_crossfeeders"],
            "K6": ["K6_G4_fiber_butyrate_crossfeeders"],
        },
        "target": "Capsule 4",
        "summary": "local",
        "direction": "low",
        "interpretation": "A low-state proxy defined from the intended distal capsule profile.",
    },
    "regional_hotspot_bacteroidales": {
        "label": "Regional Bacteroidales/Prevotella hotspot proxy",
        "guilds": ["G03_bacteroidales_generalist", "G04_prevotella_carbohydrate_propionate"],
        "guilds_by_resolution": {
            "K11": ["G03_bacteroidales_generalist", "G04_prevotella_carbohydrate_propionate"],
            "K6": ["K6_G2_bacteroidales_prevotella"],
        },
        "target": "all",
        "summary": "maximum",
        "direction": "high",
        "interpretation": "A high-state proxy defined by the maximum score across the four capsule regions.",
    },
}


def task_guilds(guild_cols: List[str], task: str) -> List[str]:
    """Resolve a task's equivalent guild score at the available resolution.

    The K6 task groups are exact amalgamations of the K11 members used by each
    proxy task.  Consequently, regional target scores and observed-stool scores
    are algebraically unchanged by K11-to-K6 aggregation; only the fitted
    reconstruction model changes.
    """
    available = set(guild_cols)
    spec = TASKS[task]
    for resolution in ["K11", "K6"]:
        members = list(spec["guilds_by_resolution"][resolution])
        if set(members).issubset(available):
            return members
    raise ValueError(f"Task {task} is not defined for guild columns: {guild_cols}")


def subject_equal_weights(subjects: Sequence[object]) -> np.ndarray:
    s = pd.Series(subjects)
    counts = s.map(s.value_counts()).to_numpy(float)
    return 1.0 / counts


def weighted_quantile(values: Sequence[float], q: float, weights: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, q * c[-1], side="left")])


def guild_score(profile: np.ndarray, guild_cols: List[str], members: List[str]) -> float:
    idx = [guild_cols.index(g) for g in members]
    return float(np.asarray(profile, dtype=float)[idx].sum())


def target_score(row: pd.Series, guild_cols: List[str], task: str) -> float:
    spec = TASKS[task]
    members = task_guilds(guild_cols, task)
    if spec["summary"] == "local":
        value = guild_score(row[spec["target"]], guild_cols, members)
    elif spec["summary"] == "maximum":
        value = max(guild_score(row[f"Capsule {r}"], guild_cols, members) for r in range(1, 5))
    else:
        raise ValueError(spec["summary"])
    return float(-value if spec["direction"] == "low" else value)


def observation_score(profile: np.ndarray, guild_cols: List[str], task: str) -> float:
    spec = TASKS[task]
    value = guild_score(profile, guild_cols, task_guilds(guild_cols, task))
    return float(-value if spec["direction"] == "low" else value)


def fit_target_threshold(train: pd.DataFrame, guild_cols: List[str], task: str, q: float = 0.5) -> float:
    values = np.array([target_score(row, guild_cols, task) for _, row in train.iterrows()])
    return weighted_quantile(values, q, subject_equal_weights(train["Subject"]))


def assign_target_state(df: pd.DataFrame, guild_cols: List[str], task: str, threshold: float) -> np.ndarray:
    values = np.array([target_score(row, guild_cols, task) for _, row in df.iterrows()])
    return (values >= threshold).astype(int)


def fit_observation_threshold(scores: Sequence[float], x: Sequence[int], subjects: Sequence[object]) -> float:
    """Select a training-only threshold for a continuous observation score.

    Candidate cutoffs are the midpoints between consecutive unique training
    scores plus the two constant-signal cutoffs.  The primary criterion is
    participant-weighted misclassification.  When several cutoffs have the
    same error, we first choose the cutoff whose weighted predicted high-state
    prevalence is closest to the weighted target-state prevalence.  Remaining
    ties are resolved by proximity to the participant-weighted median score.
    This avoids an arbitrary preference for thresholds near zero.
    """
    scores = np.asarray(scores, dtype=float)
    x = np.asarray(x, dtype=int)
    w = subject_equal_weights(subjects)
    unique = np.unique(scores)
    mids = [-np.inf]
    if len(unique) > 1:
        mids.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
    mids.append(np.inf)
    target_prevalence = float(np.sum(w * x) / np.sum(w))
    score_median = weighted_quantile(scores, 0.5, w)
    best = None
    for t in mids:
        pred = (scores >= t).astype(int)
        error = float(np.sum(w * (pred != x)) / np.sum(w))
        predicted_prevalence = float(np.sum(w * pred) / np.sum(w))
        prevalence_gap = abs(predicted_prevalence - target_prevalence)
        centrality = abs(float(t) - score_median) if np.isfinite(t) else np.inf
        candidate = (error, prevalence_gap, centrality, float(t))
        if best is None or candidate < best:
            best = candidate
    return float(best[3])


def weighted_channel(x: Sequence[int], o: Sequence[int], subjects: Sequence[object], prior: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=int)
    o = np.asarray(o, dtype=int)
    w = subject_equal_weights(subjects)
    counts = np.zeros((2, 2), dtype=float)
    for xi, oi, wi in zip(x, o, w):
        counts[xi, oi] += wi
    T = counts + prior
    T = T / T.sum(axis=1, keepdims=True)
    state_counts = np.array([w[x == 0].sum(), w[x == 1].sum()]) + prior
    phi = state_counts / state_counts.sum()
    return counts, T, phi


def decision_curve(T: np.ndarray, phi: np.ndarray, eta_grid: np.ndarray) -> pd.DataFrame:
    T = np.asarray(T, dtype=float)
    phi = closure(phi)
    m = phi @ T
    posterior = np.divide(phi[1] * T[1], m, out=np.zeros_like(m), where=m > EPS)
    rows = []
    for eta in eta_grid:
        action = (posterior >= eta).astype(int)
        value = float(np.sum(m * np.maximum(posterior - eta, 0.0)))
        perfect = float(phi[1] * (1.0 - eta))
        missed = phi[1] * float(np.sum(T[1, action == 0]))
        overtreat = phi[0] * float(np.sum(T[0, action == 1]))
        rows.append({
            "eta": float(eta),
            "value": value,
            "perfect_value": perfect,
            "midl": max(0.0, perfect - value),
            "missed_treatment_probability": float(missed),
            "overtreatment_probability": float(overtreat),
        })
    return pd.DataFrame(rows)


def summarize_curve(curve: pd.DataFrame) -> Dict[str, float]:
    eta = curve["eta"].to_numpy(float)
    return {
        "integrated_midl": float(np.trapz(curve["midl"], eta)),
        "max_midl": float(curve["midl"].max()),
        "integrated_missed_treatment": float(np.trapz(curve["missed_treatment_probability"], eta)),
        "integrated_overtreatment": float(np.trapz(curve["overtreatment_probability"], eta)),
    }


def curve_distance(a: pd.DataFrame, b: pd.DataFrame) -> float:
    eta = a["eta"].to_numpy(float)
    return float(np.trapz(np.abs(a["midl"].to_numpy() - b["midl"].to_numpy()), eta))
