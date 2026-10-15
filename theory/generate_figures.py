#!/usr/bin/env python3
"""Reproduce the five MIDL manuscript figures and their numerical analyses.

Run: python generate_figures.py --out figures

Outputs exactly figure_1.png through figure_5.png, CSV files containing the
plotted data, and analysis_summary.json. All parameterizations are synthetic.
Only NumPy, SciPy, and Matplotlib are required. No external figure libraries,
empirical datasets, user-specific paths, or import-time command-line parsing
are used. Figure 3(a,b) share K5_PRIOR; panel (c) uses its declared rank-prior
rule because its state counts differ.
"""
from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import linprog

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

CB = dict(orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", vermillion="#D55E00",
          pink="#CC79A7", black="#000000", grey="#999999",
          lgrey="#CCCCCC", darkgrey="#4D4D4D")
D_ECO = 1.0
TIE_ATOL = 1e-10
K5_BENEFIT = np.array([0., .25, .50, .75, 1.])
K5_BURDEN = np.array([0., .05, .12, .22, .35])
K5_PRIOR = np.array([.12, .20, .31, .23, .14])
MULTI_WEIGHTS = np.array([.10, .15, .25, .50])
MULTI_BLURS = np.array([.75, .55, .35, .15])
MULTI_COMPOSITIONS = np.array([
    [.55, .25, .15, .05], [.35, .35, .20, .10],
    [.18, .32, .30, .20], [.08, .22, .35, .35],
])
MULTI_REFERENCE = np.full(4, .25)
ECO_ALPHA, ECO_ETA, ECO_TAU = .60, .30, .228
R_GL = np.array([.55, .80])
A_GL = np.array([[.65, .30], [.20, .70]])
B_GL = np.array([-.50, .55])


def transition(K, beta, r=None):
    """Row-stochastic scalar affine experiment, with uniform reference by default."""
    if not 0. <= beta <= 1.:
        raise ValueError("beta must lie in [0,1]")
    r = np.full(K, 1. / K) if r is None else np.asarray(r, float)
    if r.shape != (K,) or np.any(r < 0) or not np.isclose(r.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError("r must be a probability vector of length K")
    return (1. - beta) * np.eye(K) + beta * np.tile(r, (K, 1))


@lru_cache(maxsize=None)
def monotone_strategies(K):
    """All deterministic nondecreasing K-signal, K-action policies."""
    return tuple(combinations_with_replacement(range(K), K))


def utility_tables(b, c, ell=None, delta=None):
    """Return (ecological, host) tables with rows=actions and columns=states.

    Figure 3(a) uses ell=delta=b, matching the manuscript's explicit SI choice.
    """
    b, c = np.asarray(b, float), np.asarray(c, float)
    if b.shape != c.shape or b.ndim != 1:
        raise ValueError("b and c must be same-length one-dimensional arrays")
    K = len(b)
    ell = b if ell is None else np.asarray(ell, float)
    delta = b if delta is None else np.asarray(delta, float)
    if ell.shape != (K,) or delta.shape != (K,):
        raise ValueError("ecological parameters must have length K")
    UH = b[:, None] * (np.arange(K)[None, :] >= np.arange(K)[:, None]) - c[:, None]
    UE = np.maximum(0., ell[None, :] - delta[:, None])
    return UE, UH


def outcomes(K, phi, b, beta, c, ell=None, delta=None):
    """Expected outcomes for every monotone policy under one scalar experiment."""
    phi = np.asarray(phi, float)
    if phi.shape != (K,) or np.any(phi < 0) or not np.isclose(phi.sum(), 1., rtol=0, atol=1e-12):
        raise ValueError("phi must be a probability vector of length K")
    UE, UH = utility_tables(b, c, ell, delta)
    joint = phi[:, None] * transition(K, beta)
    sigs = np.asarray(monotone_strategies(K), int)
    eco_values = np.sum((UE @ joint)[sigs, np.arange(K)[None, :]], axis=1)
    host_values = np.sum((UH @ joint)[sigs, np.arange(K)[None, :]], axis=1)
    return [(float(e), float(h), tuple(int(j) for j in sig)) for e, h, sig in zip(eco_values, host_values, sigs)]


def posterior_viability(beta, phi=K5_PRIOR, b=K5_BENEFIT, c=K5_BURDEN):
    """Signalwise Bayes tier viability, including all tied posterior maximizers.

    This is an unrestricted posterior diagnostic, not optimization over the
    monotone policy class. Zero-probability observations contribute no tiers.
    """
    phi = np.asarray(phi, float)
    _, UH = utility_tables(b, c)
    joint = phi[:, None] * transition(len(phi), beta)
    po = joint.sum(axis=0)
    positive = po > 0
    posterior = np.divide(joint, po[None, :], out=np.zeros_like(joint), where=positive[None, :])
    expected = UH @ posterior
    maximizers = np.isclose(expected, expected.max(axis=0)[None, :], rtol=0, atol=TIE_ATOL)
    maximizers[:, ~positive] = False
    return maximizers.any(axis=1), expected, posterior


def finite_envelope(phi, b, c):
    """Exact upper envelope of the finitely many affine host-policy values.

    Return interior slope-switch points for accurate piecewise-linear plotting.
    """
    left = outcomes(len(phi), phi, b, 0., c)
    right = outcomes(len(phi), phi, b, 1., c)
    lines = sorted((y[1] - x[1], x[1], i) for i, (x, y) in enumerate(zip(left, right)))
    unique = []
    for slope, intercept, index in lines:
        if unique and abs(slope - unique[-1][0]) <= 1e-12:
            if intercept > unique[-1][1]:
                unique[-1] = (slope, intercept, index)
        else:
            unique.append((slope, intercept, index))
    hull, starts = [], []
    for line in unique:
        start = -np.inf
        while hull:
            start = (hull[-1][1] - line[1]) / (line[0] - hull[-1][0])
            if start > starts[-1] + 1e-12:
                break
            hull.pop(); starts.pop()
        if not hull:
            start = -np.inf
        hull.append(line); starts.append(start)
    switches = np.array([x for x in starts if 0. < x < 1.])
    return switches


def finite_curve(K):
    b = np.linspace(0., 1., K)
    c = .35 * b ** 1.2
    phi = np.arange(1, K + 1, dtype=float); phi /= phi.sum()
    switches = finite_envelope(phi, b, c)
    betas = np.unique(np.r_[np.linspace(0., 1., 401), switches])
    host = np.array([max(p[1] for p in outcomes(K, phi, b, beta, c)) for beta in betas])
    return betas, host[0] - host, host, switches, phi, b, c


def ecological_frontiers(beta):
    points = np.array(binary_vertices(ECO_ALPHA, ECO_ETA, beta))
    clipped = clip_polygon_x(order_polygon_points(points), x_min=ECO_TAU)
    if len(clipped) == 0:
        raise ValueError("The expected ecological-margin constraint is infeasible")
    return float(points[:, 1].max()), float(clipped[:, 1].max())


def multisegment_values(gamma):
    w, betas, ref = MULTI_WEIGHTS, MULTI_BLURS, MULTI_REFERENCE
    z = (1. - gamma) * ref[None, :] + gamma * MULTI_COMPOSITIONS
    beta_eff, mean = float(w @ betas), w @ z
    total = np.sum(w[:, None] * ((1. - betas[:, None]) * z + betas[:, None] * ref), axis=0)
    effective = (1. - beta_eff) * mean + beta_eff * ref
    covariance = np.sum(w[:, None] * (betas - beta_eff)[:, None] * (z - mean), axis=0)
    bound = np.sqrt(np.sum(w * (betas - beta_eff) ** 2)) * np.sqrt(np.sum(w * np.sum((z - mean) ** 2, axis=1)))
    return total, effective, covariance, float(bound)


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(header); writer.writerows(rows)

def binary_D(alpha, eta):
    return alpha * (1.0 - eta) + eta * (1.0 - alpha)

def binary_switch(alpha, eta):
    """Blur at which the observation-dependent policy stops being optimal."""
    D = binary_D(alpha, eta)
    if abs(alpha - eta) < 1e-15:
        return 1.0
    return 2.0 * alpha * (1.0 - eta) / D if alpha < eta else 2.0 * eta * (1.0 - alpha) / D

def binary_vertices(alpha, eta, beta, d=D_ECO):
    """(A, B_beta, C): always-conservative, observation-dependent, always-aggressive."""
    D = binary_D(alpha, eta)
    A = np.array([alpha * d, 0.0])
    B = np.array([alpha * beta * d / 2.0, alpha * (1.0 - eta) - beta * D / 2.0])
    C = np.array([0.0, alpha - eta])
    return A, B, C

def binary_host_frontier(alpha, eta, beta):
    return max(0.0, alpha - eta, alpha * (1.0 - eta) - beta * binary_D(alpha, eta) / 2.0)

def binary_midl(alpha, eta, beta):
    return (binary_D(alpha, eta) / 2.0) * min(beta, binary_switch(alpha, eta))

def _set_limits(ax, allpts, pad=0.10):
    """Preserve negative host-utility coordinates; do not use a
    `max(0, ymin-pad*yr)` clamp. Under Assumption 2.1 the always-aggressive vertex
    C=(0, alpha-eta) is NEGATIVE whenever alpha < eta, and the library clamp would
    silently push it off the bottom of the axis."""
    pts = np.vstack(allpts)
    xmin, ymin = pts.min(axis=0); xmax, ymax = pts.max(axis=0)
    xr = max(xmax - xmin, 1e-6); yr = max(ymax - ymin, 1e-6)
    ax.set_xlim(xmin - pad*xr, xmax + pad*xr)
    ax.set_ylim(ymin - pad*yr, ymax + pad*yr)

def hull_xy(pts):
    arr = np.array([(x, y) for x, y, *_ in pts])
    if len(arr) < 3: return arr
    a = np.unique(np.round(arr, 10), axis=0)
    if len(a) < 3: return a
    try:
        xy = a[ConvexHull(a).vertices]
        ctr = xy.mean(axis=0)
        xy = xy[np.argsort(np.arctan2(xy[:, 1] - ctr[1], xy[:, 0] - ctr[0]))]
        return np.vstack([xy, xy[0]])
    except Exception:
        return a

def save(fig, fig_dir: Path, name: str) -> str:
    path = fig_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"✓ {path.name}")
    return str(path)

def panel_label(ax, letter: str, x=-0.12, y=1.05):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=12,
            fontweight="bold", ha="left", va="top")

def order_polygon_points(points):
    """Return polygon points in cyclic order around their centroid.

    This prevents matplotlib from drawing crossed/overlapping triangle fills
    when vertices are supplied in a non-geometric order.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] <= 2:
        return pts

    center = pts.mean(axis=0)
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )
    return pts[np.argsort(angles)]

def draw_triangle(ax, verts: Iterable[np.ndarray], color: str, label=None, alpha=0.18,
                  lw=2.0, ls="-", z=2):
    pts = np.vstack(list(verts))
    pts_closed = np.vstack([pts, pts[0]])
    ax.fill(pts[:,0], pts[:,1], color=color, alpha=alpha, zorder=z)
    ax.plot(pts_closed[:,0], pts_closed[:,1], color=color, lw=lw, ls=ls, label=label, zorder=z+1)
    return pts

def label_vertices(ax, labels, verts, color, offsets=None):
    if offsets is None:
        offsets = [(5,5)]*len(labels)
    for lab, v, off in zip(labels, verts, offsets):
        ax.scatter([v[0]], [v[1]], s=28, color=color, zorder=10)
        ax.annotate(lab, v, textcoords="offset points", xytext=off,
                    fontsize=9, color=color, fontweight="bold")

def clip_polygon_x(poly, x_min=None, x_max=None):
    """Clip a 2D polygon by a vertical half-plane.

    Use x_min=tau for U_E >= tau.
    Use x_max=tau for U_E <= tau.
    """
    pts = np.asarray(poly, dtype=float)
    if len(pts) == 0:
        return pts

    if (x_min is None) == (x_max is None):
        raise ValueError("Provide exactly one of x_min or x_max.")

    x_cut = x_min if x_min is not None else x_max

    def inside(p):
        return p[0] >= x_cut if x_min is not None else p[0] <= x_cut

    def intersect(p, q):
        denom = q[0] - p[0]
        if abs(denom) < 1e-12:
            return p.copy()
        t = (x_cut - p[0]) / denom
        return p + t * (q - p)

    out = []
    prev = pts[-1]
    prev_inside = inside(prev)

    for cur in pts:
        cur_inside = inside(cur)

        if cur_inside:
            if not prev_inside:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_inside:
            out.append(intersect(prev, cur))

        prev = cur
        prev_inside = cur_inside

    return np.asarray(out, dtype=float)

order_polygon_points_x = order_polygon_points

def interior_eq(u):
    """Formal interior equilibrium x*(u)=A^{-1}(r+bu)."""
    return np.linalg.solve(A_GL, R_GL + B_GL * u)

def glv_jacobian(x, u):
    """Full Jacobian for dx_i=x_i[r_i+b_i*u-(A x)_i]."""
    x = np.asarray(x, dtype=float)
    per_capita_growth = R_GL + B_GL * u - A_GL @ x

    return (
        np.diag(per_capita_growth)
        - np.diag(x) @ A_GL
    )

def exact_coexistence_boundary():
    """First value of u at which an interior component reaches zero."""
    intercept = np.linalg.solve(A_GL, R_GL)
    slope = np.linalg.solve(A_GL, B_GL)

    candidates = []
    for guild in range(len(intercept)):
        if slope[guild] < 0:
            crossing = -intercept[guild] / slope[guild]
            if 0.0 <= crossing <= 1.0:
                candidates.append((crossing, guild))

    if not candidates:
        raise RuntimeError(
            "No interior-equilibrium component reaches zero on u in [0,1]."
        )

    ucrit, extinct_guild = min(candidates)
    return float(ucrit), int(extinct_guild)

def boundary_eq(u):
    """Boundary equilibrium after loss of guild 0."""
    return np.array([
        0.0,
        (R_GL[1] + B_GL[1] * u) / A_GL[1, 1],
    ])

def figure_1_feasible_triangles(OUT):
        # ---------------- Figure 1: binary feasible sets ----------------
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6)); axes = axes.ravel()
        panels = [dict(alpha=0.65, eta=0.35, frac=0.45, tag="Low burden / low blur"),
                  # eta=0.65 mirrors panel (a) across alpha<->eta, so beta_c (hence beta) matches exactly
                  dict(alpha=0.35, eta=0.65, frac=0.45, tag="High burden / low blur"),
                  dict(alpha=0.65, eta=0.35, frac=0.90, tag="Low burden / higher blur"),
                  dict(alpha=0.65, eta=0.35, frac=0.00, tag="Perfect-observation baseline")]
        for ax, p in zip(axes, panels):
            a, e, fr = p["alpha"], p["eta"], p["frac"]
            bc = binary_switch(a, e); beta = fr * bc
            A0, B0, C0 = binary_vertices(a, e, 0.0)
            P0 = order_polygon_points([A0, B0, C0]); pts_all = [P0]
            ax.plot(np.r_[P0[:, 0], P0[0, 0]], np.r_[P0[:, 1], P0[0, 1]], ls="--", lw=1.8,
                    color=CB["vermillion"], label=r"$\mathcal{S}(0)$ perfect observation", zorder=2)
            if beta > 0:
                A, B, C = binary_vertices(a, e, beta)
                P = order_polygon_points([A, B, C]); pts_all.append(P)
                ax.fill(P[:, 0], P[:, 1], color=CB["blue"], alpha=0.25, zorder=3)
                ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], color=CB["blue"], lw=2.2,
                        label=rf"$\mathcal{{S}}(\beta={beta:.2f})$ stool-blurred", zorder=4)
                label_vertices(ax, ["A", "B", "C"], [A, B, C], CB["blue"], [(7, 7), (7, -14), (-18, 7)])
            else:
                label_vertices(ax, ["A", "B", "C"], [A0, B0, C0], CB["vermillion"], [(7, 7), (7, -14), (-18, 7)])
            ax.axhline(
        0,
        color=CB["darkgrey"],
        lw=0.9,
        ls=":",
        zorder=1,
    )
            ax.text(0.035, 0.05, rf"$\beta_c={bc:.2f}$", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec=CB["lgrey"], alpha=0.9))
            ax.set_title(f"{p['tag']}\n" + rf"$\alpha={a:.2f},\ \eta={e:.2f},\ \beta={beta:.2f}$",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel(r"$U_E$ (ecological margin)"); ax.set_ylabel(r"$U_H$ (net host utility)")
            _set_limits(ax, pts_all, pad=0.18)
            ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=7.7, loc="best")
        for ax, lab in zip(axes, "abcd"): panel_label(ax, lab)
        fig.tight_layout(); save(fig, OUT, "figure_1.png")



def figure_2_binary_midl(OUT):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # ------------------------------------------------------------------
    # Panel (a): MIDL versus beta
    # ------------------------------------------------------------------
    ax = axes[0]

    # Keep these numerical values and make the labels match them.
    configs = [
        (
            0.35,
            0.35,
            CB["blue"],
            r"$\alpha=0.35,\ \eta=0.35$",
        ),
        (
            0.45,
            0.35,
            CB["green"],
            r"$\alpha=0.45,\ \eta=0.35$",
        ),
        (
            0.45,
            0.65,
            CB["orange"],
            r"$\alpha=0.45,\ \eta=0.65$",
        ),
        (
            0.65,
            0.35,
            CB["vermillion"],
            r"$\alpha=0.65,\ \eta=0.35$",
        ),
    ]

    beta_grid = np.linspace(0.0, 1.0, 400)

    for alpha, eta, color, label in configs:
        values = np.array([
            binary_midl(alpha, eta, beta)
            for beta in beta_grid
        ])

        # MIDL must be nonnegative, nondecreasing, and concave.
        if np.min(values) < -1e-10:
            raise AssertionError("MIDL became negative.")

        if np.any(np.diff(values) < -1e-9):
            raise AssertionError("MIDL is not nondecreasing.")

        if np.any(np.diff(values, n=2) > 1e-8):
            raise AssertionError("MIDL is not concave.")

        ax.plot(
            beta_grid,
            values,
            color=color,
            linewidth=2,
            label=label,
        )

        beta_c = binary_switch(alpha, eta)

        # Mark interior policy switches. For alpha=eta, beta_c=1,
        # so the tie occurs only at the endpoint.
        if beta_c < 1.0 - 1e-10:
            ax.scatter(
                [beta_c],
                [binary_midl(alpha, eta, beta_c)],
                s=28,
                color=color,
                zorder=5,
            )

    ax.set_xlabel(r"Information-loss index $\beta$")
    ax.set_ylabel(
        r"MIDL $=U_H^{\max}(0)-U_H^{\max}(\beta)$"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.margins(x=0)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    panel_label(ax, "a")


    # ------------------------------------------------------------------
    # Panel (b): fixed-beta MIDL landscape
    # ------------------------------------------------------------------
    ax = axes[1]

    beta_fix = 0.45
    etas = np.linspace(0.05, 0.95, 200)
    alphas = np.linspace(0.05, 0.95, 200)

    Z = np.array([
        [
            binary_midl(alpha, eta, beta_fix)
            for eta in etas
        ]
        for alpha in alphas
    ])

    beta_c_surface = np.array([
        [
            binary_switch(alpha, eta)
            for eta in etas
        ]
        for alpha in alphas
    ])

    # Basic theory checks.
    if np.min(Z) < -1e-10:
        raise AssertionError("The MIDL landscape contains negative values.")

    # Because the grids are identical, the landscape should be symmetric
    # under alpha <-> eta.
    if not np.allclose(Z, Z.T, atol=1e-10):
        raise AssertionError(
            "The binary MIDL landscape is not symmetric in alpha and eta."
        )

    levels = np.linspace(0.0, float(np.max(Z)), 21)

    filled = ax.contourf(
        etas,
        alphas,
        Z,
        levels=levels,
        cmap="YlOrRd",
    )

    fig.colorbar(
        filled,
        ax=ax,
        label="MIDL (net host-utility units)",
    )

    # Actual policy-switch boundary at the beta displayed in the panel.
    switch_contour = ax.contour(
        etas,
        alphas,
        beta_c_surface,
        levels=[beta_fix],
        colors="black",
        linewidths=1.5,
    )

    ax.clabel(
        switch_contour,
        fmt={beta_fix: r"$\beta_c=\beta$"},
        inline=True,
        fontsize=8,
    )

    # Symmetry line; beta_c=1 along alpha=eta.
    ax.plot(
        etas,
        etas,
        color="white",
        linestyle="--",
        linewidth=1.4,
        label=r"$\alpha=\eta$ ($\beta_c=1$)",
    )

    ax.set_xlabel(
        r"Normalized intervention burden "
        r"$\eta=c_1/b_1$"
    )
    ax.set_ylabel(
        r"Prior high-state probability "
        r"$\alpha=P(X=1)$"
    )
    ax.set_title(
        rf"MIDL landscape at $\beta={beta_fix:.2f}$",
        fontsize=10,
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    panel_label(ax, "b")

    fig.tight_layout()
    save(fig, OUT, "figure_2.png")

def figure_3_finite_state(OUT):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    K, b, c, phi = 5, K5_BENEFIT, K5_BURDEN, K5_PRIOR
    for beta in (0., .35, .75):
        xy = hull_xy(outcomes(K, phi, b, beta, c, ell=b, delta=b))
        axes[0].plot(xy[:, 0], xy[:, 1], lw=2, label=rf"$\beta={beta:g}$")
    axes[0].set_xlabel(r"Ecological margin $U_E$")
    axes[0].set_ylabel(r"Net host benefit $U_H$")
    axes[0].set_title("(a)"); axes[0].legend(frameon=False)

    betas = np.linspace(0., 1., 501)
    viable = np.column_stack([posterior_viability(beta, phi, b, c)[0] for beta in betas])
    for j in range(K):
        axes[1].fill_between(betas, j, j + .8, where=viable[j], step="mid", alpha=.35)
    axes[1].set_yticks(np.arange(K) + .4)
    axes[1].set_yticklabels([f"tier {j}" for j in range(K)])
    axes[1].set_xlabel(r"Information-loss index $\beta$")
    axes[1].set_title("(b)"); axes[1].set_xlim(0, 1); axes[1].set_ylim(0, K)

    for K in (3, 5, 6):
        betas, midl, *_ = finite_curve(K)
        axes[2].plot(betas, midl, lw=2, label=f"K={K}")
    axes[2].set_xlabel(r"Information-loss index $\beta$")
    axes[2].set_ylabel("MIDL"); axes[2].set_title("(c)")
    axes[2].legend(frameon=False)
    fig.tight_layout(); save(fig, OUT, "figure_3.png")


def figure_4_multisegment(OUT):
    OUT = Path(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    S = 4
    G = 4

    # Segment contribution weights and blur levels
    w = MULTI_WEIGHTS
    beta_s = MULTI_BLURS

    # Common reference distribution
    reference = MULTI_REFERENCE
    base = np.ones(G, dtype=float) / G

    # Illustrative segment-specific functional-guild compositions
    z_s = MULTI_COMPOSITIONS

    # --------------------------------------------------------------
    # Input validation
    # --------------------------------------------------------------
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("Segment weights must sum to one.")

    if np.any(w < 0):
        raise ValueError("Segment weights must be nonnegative.")

    if np.any((beta_s < 0.0) | (beta_s > 1.0)):
        raise ValueError("Every segment blur level must lie in [0,1].")

    if z_s.shape != (S, G):
        raise ValueError("z_s must have shape (S,G).")

    if np.any(z_s < 0.0):
        raise ValueError("Guild compositions must be nonnegative.")

    if not np.allclose(z_s.sum(axis=1), 1.0):
        raise ValueError("Every segment composition must sum to one.")

    if not np.isclose(reference.sum(), 1.0):
        raise ValueError("The reference profile must sum to one.")

    beta_eff = float(np.dot(w, beta_s))

    # --------------------------------------------------------------
    # Plotting style
    # --------------------------------------------------------------
    color_map = {
        "weight": "#EEAEAE",
        "blur": "#CEB527",
        "gap": "#B03A48",
        "bound": "#357994",
    }

    guild_colors = [
        "#70C79E",
        "#72B7B2",
        "#F2CF5B",
        "#B279A2",
    ]

    for ax in axes:
        ax.set_facecolor("#FAFAFA")
        ax.grid(
            axis="y",
            color="#D0D0D0",
            alpha=0.45,
            linewidth=0.8,
        )
        ax.set_axisbelow(True)
        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.tick_params(labelsize=10, colors="#333333")

    segment_index = np.arange(S)

    # --------------------------------------------------------------
    # Panel (a): segment weights and blur levels
    # --------------------------------------------------------------
    ax = axes[0]

    ax.bar(
        segment_index - 0.18,
        w,
        width=0.35,
        label=r"Segment weight $w_s$",
        color=color_map["weight"],
        edgecolor="white",
        linewidth=1.2,
    )

    ax.bar(
        segment_index + 0.18,
        beta_s,
        width=0.35,
        label=r"Segment blur $\beta_s$",
        color=color_map["blur"],
        edgecolor="white",
        linewidth=1.2,
    )

    ax.set_xticks(segment_index)
    ax.set_xticklabels(
        [f"Seg. {index + 1}" for index in segment_index]
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dimensionless value", fontsize=11)
    ax.set_title(
        "(a)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="upper right",
    )

    # --------------------------------------------------------------
    # Panel (b): functional-guild compositions
    # --------------------------------------------------------------
    ax = axes[1]

    bottom = np.zeros(S)

    for guild in range(G):
        ax.bar(
            segment_index,
            z_s[:, guild],
            bottom=bottom,
            label=f"Guild {guild + 1}",
            color=guild_colors[guild],
            edgecolor="white",
            linewidth=1.0,
        )
        bottom += z_s[:, guild]

    ax.set_xticks(segment_index)
    ax.set_xticklabels(
        [f"Seg. {index + 1}" for index in segment_index]
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Functional-guild composition", fontsize=11)
    ax.set_title(
        "(b)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )

    # --------------------------------------------------------------
    # Panel (c): exact discrepancy and norm bound
    # --------------------------------------------------------------
    ax = axes[2]

    gamma_grid = np.linspace(0.0, 1.0, 101)
    discrepancies = []
    bounds = []

    beta_dispersion = np.sqrt(
        np.sum(
            w * (beta_s - beta_eff) ** 2
        )
    )

    for gamma in gamma_grid:
        compositions = (
            (1.0 - gamma) * base[None, :]
            + gamma * z_s
        )

        # Convex combinations of simplex vectors must remain in the simplex.
        if np.any(compositions < -1e-12):
            raise AssertionError(
                "Interpolated compositions became negative."
            )

        if not np.allclose(
            compositions.sum(axis=1),
            1.0,
            atol=1e-12,
        ):
            raise AssertionError(
                "Interpolated compositions do not sum to one."
            )

        mean_composition = np.sum(
            w[:, None] * compositions,
            axis=0,
        )

        total_profile = np.sum(
            w[:, None]
            * (
                (1.0 - beta_s[:, None]) * compositions
                + beta_s[:, None] * reference[None, :]
            ),
            axis=0,
        )

        effective_profile = (
            (1.0 - beta_eff) * mean_composition
            + beta_eff * reference
        )

        covariance_term = np.sum(
            w[:, None]
            * (beta_s - beta_eff)[:, None]
            * (compositions - mean_composition[None, :]),
            axis=0,
        )

        # Exact multi-segment identity
        if not np.allclose(
            total_profile,
            effective_profile - covariance_term,
            atol=1e-12,
        ):
            raise AssertionError(
                "The multi-segment covariance identity failed."
            )

        discrepancy = float(
            np.linalg.norm(
                total_profile - effective_profile,
                ord=2,
            )
        )

        composition_dispersion = np.sqrt(
            np.sum(
                w
                * np.linalg.norm(
                    compositions - mean_composition[None, :],
                    axis=1,
                ) ** 2
            )
        )

        upper_bound = float(
            beta_dispersion * composition_dispersion
        )

        if discrepancy > upper_bound + 1e-12:
            raise AssertionError(
                "The discrepancy exceeded the Cauchy--Schwarz bound."
            )

        discrepancies.append(discrepancy)
        bounds.append(upper_bound)

    discrepancies = np.asarray(discrepancies)
    bounds = np.asarray(bounds)

    ax.plot(
        gamma_grid,
        discrepancies,
        linewidth=2.8,
        color=color_map["gap"],
        label="Exact single-blur discrepancy",
    )

    ax.plot(
        gamma_grid,
        bounds,
        linewidth=2.8,
        linestyle="--",
        color=color_map["bound"],
        label="Cauchy--Schwarz bound",
    )

    ax.fill_between(
        gamma_grid,
        discrepancies,
        bounds,
        color=color_map["bound"],
        alpha=0.10,
        linewidth=0,
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(
        r"Heterogeneity scaling $\gamma$",
        fontsize=11,
    )
    ax.set_ylabel(
        r"Magnitude ($\ell_2$ norm)",
        fontsize=11,
    )
    ax.set_title(
        "(c)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="upper left",
    )

    fig.tight_layout()

    output_path = OUT / "figure_4.png"
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print(
        f"beta_eff={beta_eff:.4f}; "
        f"gap(gamma=1)={discrepancies[-1]:.6f}; "
        f"bound(gamma=1)={bounds[-1]:.6f}"
    )

def ecological_dynamics_panel(ax):
    """Locally stable equilibrium branches of the two-guild toy model."""
    ucrit, extinct_guild = exact_coexistence_boundary()

    if extinct_guild != 0:
        raise RuntimeError(
            "This plotting function assumes that guild 0 is lost first."
        )

    # Include ucrit explicitly so the two branches meet exactly.
    u_in = np.linspace(0.0, ucrit, 350)
    u_out = np.linspace(ucrit, 1.0, 450)

    coexistence = np.vstack([
        interior_eq(u_value)
        for u_value in u_in
    ])

    boundary = np.vstack([
        boundary_eq(u_value)
        for u_value in u_out
    ])

    # Verify feasibility.
    if np.any(coexistence < -1e-10):
        raise AssertionError(
            "The displayed coexistence branch contains a negative abundance."
        )

    if np.any(boundary < -1e-10):
        raise AssertionError(
            "The displayed boundary branch contains a negative abundance."
        )

    # Verify local stability away from the non-hyperbolic crossing itself.
    interior_stability = [
        np.max(np.real(np.linalg.eigvals(glv_jacobian(x, u_value))))
        for x, u_value in zip(coexistence[:-1], u_in[:-1])
    ]

    boundary_stability = [
        np.max(np.real(np.linalg.eigvals(glv_jacobian(x, u_value))))
        for x, u_value in zip(boundary[1:], u_out[1:])
    ]

    if max(interior_stability) >= 1e-8:
        raise AssertionError(
            "The displayed interior branch is not locally stable."
        )

    if max(boundary_stability) >= 1e-8:
        raise AssertionError(
            "The displayed boundary branch is not locally stable."
        )

    # Stable coexistence branch.
    ax.plot(
        u_in,
        coexistence[:, 0],
        color=CB["vermillion"],
        linewidth=2,
        label="Guild 1",
    )
    ax.plot(
        u_in,
        coexistence[:, 1],
        color=CB["blue"],
        linewidth=2,
        label="Guild 2",
    )

    # Stable boundary branch after the bifurcation.
    ax.plot(
        u_out,
        boundary[:, 0],
        color=CB["vermillion"],
        linewidth=2,
    )
    ax.plot(
        u_out,
        boundary[:, 1],
        color=CB["blue"],
        linewidth=2,
    )

    ax.axvspan(
        0.0,
        ucrit,
        color=CB["green"],
        alpha=0.12,
        label="Stable coexistence",
    )
    ax.axvline(
        ucrit,
        color=CB["green"],
        linewidth=2,
        linestyle="--",
    )
    ax.text(
    ucrit,                  # x-position
    0.95,                   # y-position in axes coordinates
    r"$u_{\mathrm{crit}}$",
    color=CB["green"],
    fontsize=14,
    rotation=90,
    va="top",
    ha="right",
    transform=ax.get_xaxis_transform(),
)
    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        linestyle=":",
    )

    ax.scatter(
        [ucrit],
        [0.0],
        s=32,
        color=CB["vermillion"],
        zorder=6,
        clip_on=False,
    )

    ax.annotate(
        "Transcritical bifurcation\n"
        "Guild 1 is lost",
        xy=(ucrit, 0.0),
        xytext=(ucrit + 0.06, 0.32),
        fontsize=8,
        arrowprops={
            "arrowstyle": "->",
            "color": CB["vermillion"],
        },
        color=CB["vermillion"],
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("Intervention intensity", fontsize=12)
    ax.set_ylabel("Equilibrium abundance", fontsize=12)
    ax.legend(
        frameon=False,
        fontsize=9,
        loc="center right",
    )
    panel_label(ax, "a")

def ecological_screening_panels(axes_bc):
    alpha_s, eta = ECO_ALPHA, ECO_ETA
    tau = ECO_TAU

    # (b)
    ax = axes_bc[0]
    beta_b = 0.30
    Ab, Bb, Cb = binary_vertices(alpha_s, eta, beta_b)
    pts = order_polygon_points_x([Ab, Bb, Cb])
    draw_triangle(ax, pts, CB["blue"], label=r"$\mathcal{S}(\beta)$", alpha=0.18, lw=2.0)
    eco = clip_polygon_x(pts, x_min=tau); excluded = clip_polygon_x(pts, x_max=tau)
    if len(eco) >= 3:
        ax.fill(eco[:, 0], eco[:, 1], color=CB["green"], alpha=0.40,
                label=r"$\mathcal{S}_{eco}(\beta;\tau)$", zorder=3)
        ax.plot(np.r_[eco[:, 0], eco[0, 0]], np.r_[eco[:, 1], eco[0, 1]], color=CB["green"], lw=1.5, zorder=4)
    if len(excluded) >= 3:
        ax.fill(excluded[:, 0], excluded[:, 1], color=CB["orange"], alpha=0.35, hatch="//",
                label="Excluded by ecological-margin floor", zorder=2)
    ax.axvline(tau, color=CB["green"], ls="--", lw=1.4)
    label_vertices(ax, ["A", "B", "C"], [Ab, Bb, Cb], CB["blue"], [(5, 5), (5, -16), (-18, 5)])
    ax.set_xlabel(r"$U_E$ (ecological margin)", fontsize=12)
    ax.set_ylabel(r"$U_H$ (net host benefit)", fontsize=12)
    _set_limits(ax, [pts], pad=0.15)
    ax.text(tau, ax.get_ylim()[1], r"$U_E=\tau$", color=CB["green"], fontsize=12,
            rotation=90, va="top", ha="right")
    ax.legend(frameon=False, fontsize=10, loc="best")
    panel_label(ax, "b")

    # (c)
    ax = axes_bc[1]
    betas = np.linspace(0, 0.55, 240)
    uh_un, uh_co = [], []
    for be in betas:
        P = order_polygon_points(list(binary_vertices(alpha_s, eta, be)))
        uh_un.append(float(np.max(P[:, 1])))
        ec = clip_polygon_x(P, x_min=tau)
        uh_co.append(float(np.max(ec[:, 1])) if len(ec) >= 1 else np.nan)
    uh_un = np.asarray(uh_un); uh_co = np.asarray(uh_co)
    bc = binary_switch(alpha_s, eta)
    ax.plot(betas, uh_un, color=CB["blue"], lw=2, label=r"Unconstrained $\max U_H$")
    ax.plot(betas, uh_co, color=CB["green"], lw=2, ls="--", label=r"Ecological-margin-constrained $\max U_H$")
    ax.fill_between(betas, uh_co, uh_un, where=uh_un > uh_co, color=CB["orange"], alpha=0.28, label="Ecological-screening gap")
    ax.axvline(bc, color=CB["grey"], lw=1.1, ls=":")
    ax.annotate(rf"$\beta_c={bc:.2f}$", xy=(bc, ax.get_ylim()[0]),
                xytext=(bc + 0.02, ax.get_ylim()[0] + 0.02), fontsize=8, color=CB["grey"])
    ax.set_xlabel(r"Information loss $\beta$", fontsize=12)
    ax.set_ylabel("Host-benefit frontier", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    panel_label(ax, "c")

def figure_5_ecological(OUT):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.3))
    ecological_dynamics_panel(axes[0])
    ecological_screening_panels(axes[1:])
    fig.tight_layout(rect=[0, 0, 1, .94])
    save(fig, OUT, "figure_5.png")


def validate_numerics():
    """Independent decision and numerical checks for the manuscript examples."""
    rng = np.random.default_rng(1946)
    binary_error = 0.
    for _ in range(500):
        alpha, eta = rng.uniform(.001, .999, 2)
        beta = rng.uniform()
        phi = np.array([1. - alpha, alpha])
        T = transition(2, beta)
        host = np.array([[0., 0.], [-eta, 1. - eta]])
        ecology = np.array([[0., D_ECO], [0., 0.]])
        policies = [(0, 0), (0, 1), (1, 1)]
        direct = np.array([
            [sum(phi[k] * T[k, o] * utility[sig[o], k]
                 for k in range(2) for o in range(2))
             for utility in (ecology, host)] for sig in policies
        ])
        expected = np.array(binary_vertices(alpha, eta, beta))
        binary_error = max(binary_error, float(np.max(np.abs(direct - expected))))
        ideal = max(sum(phi[k] * host[sig[k], k] for k in range(2)) for sig in policies)
        if not np.allclose(direct, expected, rtol=0, atol=1e-12):
            raise AssertionError("Binary vertices disagree with direct joint-distribution calculation")
        if abs(ideal - direct[:, 1].max() - binary_midl(alpha, eta, beta)) > 1e-12:
            raise AssertionError("Binary MIDL disagrees with independently optimized host benefit")

    garbling_error = 0.
    for _ in range(100):
        K = int(rng.integers(2, 8)); r = rng.dirichlet(np.ones(K))
        beta1, beta2 = sorted(rng.uniform(0., 1., 2))
        lam = (1. - beta2) / (1. - beta1)
        kernel = lam * np.eye(K) + (1. - lam) * np.tile(r, (K, 1))
        error = float(np.max(np.abs(transition(K, beta1, r) @ kernel - transition(K, beta2, r))))
        garbling_error = max(garbling_error, error)
        if error > 1e-12 or np.min(kernel) < -1e-12:
            raise AssertionError("Scalar Blackwell factorization failed")

    finite_results = {}
    for K in (3, 5, 6):
        betas, loss, host, switches, phi, b, c = finite_curve(K)
        if np.min(loss) < -1e-12 or np.max(np.diff(host)) > 1e-12:
            raise AssertionError("The finite-state frontier violates scalar monotonicity")
        slopes = np.diff(host) / np.diff(betas)
        if np.min(np.diff(slopes)) < -1e-8:
            raise AssertionError("The finite-state host frontier is not convex")
        for beta in (0., .37, 1.):
            values = outcomes(K, phi, b, beta, c)
            T = transition(K, beta)
            for index in (0, len(values) // 2, len(values) - 1):
                UE, UH, sig = values[index]
                h_direct = sum(phi[k] * T[k, o] * (b[sig[o]] * (k >= sig[o]) - c[sig[o]])
                               for k in range(K) for o in range(K))
                e_direct = sum(phi[k] * T[k, o] * max(0., b[k] - b[sig[o]])
                               for k in range(K) for o in range(K))
                if abs(UH - h_direct) > 1e-12 or abs(UE - e_direct) > 1e-12:
                    raise AssertionError("Vectorized policy outcomes disagree with direct enumeration")
        boundaries = np.r_[0., switches, 1.]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            midpoint = (left + right) / 2.
            Hmid = max(p[1] for p in outcomes(K, phi, b, midpoint, c))
            Hleft = max(p[1] for p in outcomes(K, phi, b, left, c))
            Hright = max(p[1] for p in outcomes(K, phi, b, right, c))
            if abs(Hmid - (Hleft + Hright) / 2.) > 1e-10:
                raise AssertionError("Exact finite-policy envelope missed a switch point")
        finite_results[str(K)] = {
            "policy_count": len(monotone_strategies(K)),
            "H_identity": float(host[0]), "H_uninformative": float(host[-1]),
            "MIDL_uninformative": float(loss[-1]),
            "interior_slope_switches": switches.tolist(),
            "prior": phi.tolist(), "benefit": b.tolist(), "burden": c.tolist(),
        }

    _, expected, _ = posterior_viability(.8)
    policy = np.argmax(expected, axis=0)
    joint = K5_PRIOR[:, None] * transition(5, .8)
    _, UH = utility_tables(K5_BENEFIT, K5_BURDEN)
    unrestricted = float(np.max(UH @ joint, axis=0).sum())
    monotone = max(outcomes(5, K5_PRIOR, K5_BENEFIT, .8, K5_BURDEN), key=lambda p: p[1])
    if policy.tolist() != [2, 1, 2, 2, 2] or abs(unrestricted - .2248) > 1e-12 or abs(monotone[1] - .22) > 1e-12:
        raise AssertionError("Shared-prior posterior/monotone diagnostic changed unexpectedly")

    clip_error = 0.
    for beta in np.linspace(0., .55, 241):
        points = np.array(binary_vertices(ECO_ALPHA, ECO_ETA, beta))
        H, Htau = ecological_frontiers(beta)
        lp = linprog(-points[:, 1], A_ub=-points[:, 0][None, :], b_ub=[-ECO_TAU],
                     A_eq=np.ones((1, 3)), b_eq=[1.], bounds=(0., None), method="highs")
        if not lp.success:
            raise AssertionError("The independent ecological linear program failed")
        clip_error = max(clip_error, abs(Htau + lp.fun))
        if Htau > H + 1e-12 or abs(Htau + lp.fun) > 1e-10:
            raise AssertionError("Polygon clipping disagrees with independent constrained optimization")
        analytical = .372 * (.42 - .27 * beta) / (.60 - .30 * beta)
        if abs(Htau - analytical) > 1e-12:
            raise AssertionError("The analytical ecological-gap example disagrees with clipping")

    ucrit, extinct = exact_coexistence_boundary()
    if extinct != 0:
        raise AssertionError("The selected gLV parameters no longer lose Guild 1 first")
    eigen_at_crossing = np.linalg.eigvals(glv_jacobian(boundary_eq(ucrit), ucrit))
    if abs(np.max(np.real(eigen_at_crossing))) > 1e-10:
        raise AssertionError("The coexistence crossing is not non-hyperbolic")
    for u in np.r_[np.linspace(0., ucrit, 101)[:-1], np.linspace(ucrit, 1., 101)[1:]]:
        x = interior_eq(u) if u < ucrit else boundary_eq(u)
        if np.any(x < -1e-12) or np.max(np.real(np.linalg.eigvals(glv_jacobian(x, u)))) >= 0:
            raise AssertionError("The displayed realized gLV equilibrium is infeasible or unstable")

    identity_error = 0.
    for gamma in np.linspace(0., 1., 101):
        total, effective, covariance, bound = multisegment_values(gamma)
        identity_error = max(identity_error, float(np.max(np.abs(total - effective + covariance))))
        if np.linalg.norm(total - effective) > bound + 1e-12:
            raise AssertionError("The compositional discrepancy exceeds its norm bound")
    if identity_error > 1e-12:
        raise AssertionError("The multi-segment covariance identity failed")
    total, effective, _, bound = multisegment_values(1.)

    beta = np.array([.067, .666, .630]); beta_prime = np.array([.290, .959, .651])
    T = np.diag(1. - beta) + beta[:, None] / 3.
    Tp = np.diag(1. - beta_prime) + beta_prime[:, None] / 3.
    forward, reverse = np.linalg.solve(T, Tp), np.linalg.solve(Tp, T)
    if forward.min() >= 0 or reverse.min() >= 0:
        raise AssertionError("The state-specific incomparability check failed")

    return {
        "checks_passed": True,
        "binary_direct_checks": 500, "binary_max_vertex_error": binary_error,
        "scalar_garbling_checks": 100, "scalar_garbling_max_error": garbling_error,
        "finite_state_systems": finite_results,
        "figure_3b_prior": K5_PRIOR.tolist(),
        "figure_3b_definition": "Signalwise posterior-optimal tiers, all ties, no monotonicity constraint",
        "figure_3b_tie_absolute_tolerance": TIE_ATOL,
        "figure_3b_beta_0_8_diagnostic": {
            "Bayes_policy": policy.tolist(), "unrestricted_host_benefit": unrestricted,
            "monotone_host_benefit": monotone[1], "monotone_policy": list(monotone[2]),
        },
        "ecological_clip_LP_checks": 241, "ecological_clip_max_error": clip_error,
        "gLV_ucrit": ucrit,
        "gLV_crossing_eigenvalues": np.real(eigen_at_crossing).tolist(),
        "multi_segment_identity_max_error": identity_error,
        "multi_segment_beta_eff": float(MULTI_WEIGHTS @ MULTI_BLURS),
        "multi_segment_gap_gamma_1": float(np.linalg.norm(total - effective)),
        "multi_segment_bound_gamma_1": bound,
        "state_specific_forward_kernel_min": float(forward.min()),
        "state_specific_reverse_kernel_min": float(reverse.min()),
        "software": {"numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    }


def export_analysis(OUT, summary):
    data = OUT / "data"; data.mkdir(parents=True, exist_ok=True)
    triangle_rows = []
    for panel, alpha, eta, fraction in [("a", .65, .35, .45), ("b", .35, .65, .45),
                                        ("c", .65, .35, .90), ("d", .65, .35, 0.)]:
        beta_c = binary_switch(alpha, eta); beta = fraction * beta_c
        for label, point in zip("ABC", binary_vertices(alpha, eta, beta)):
            triangle_rows.append([panel, alpha, eta, beta, beta_c, D_ECO, label, *point])
    write_csv(data / "figure_1_vertices.csv", ["panel", "alpha", "eta", "beta", "beta_c", "d", "policy", "U_E", "U_H"], triangle_rows)

    rows = [[alpha, eta, beta, binary_switch(alpha, eta), binary_host_frontier(alpha, eta, beta), binary_midl(alpha, eta, beta)]
            for alpha, eta in [(.35, .35), (.45, .35), (.45, .65), (.65, .35)]
            for beta in np.linspace(0., 1., 400)]
    write_csv(data / "figure_2_midl.csv", ["alpha", "eta", "beta", "beta_c", "H", "MIDL"], rows)
    grid = np.linspace(.05, .95, 200)
    rows = [[alpha, eta, .45, binary_switch(alpha, eta), binary_midl(alpha, eta, .45)] for alpha in grid for eta in grid]
    write_csv(data / "figure_2_landscape.csv", ["alpha", "eta", "beta", "beta_c", "MIDL"], rows)

    rows = [[beta, *sig, UE, UH] for beta in (0., .35, .75)
            for UE, UH, sig in outcomes(5, K5_PRIOR, K5_BENEFIT, beta, K5_BURDEN)]
    write_csv(data / "figure_3_feasible_points.csv", ["beta", "action_o0", "action_o1", "action_o2", "action_o3", "action_o4", "U_E", "U_H"], rows)
    rows = [[beta, j, int(flag)] for beta in np.linspace(0., 1., 501)
            for j, flag in enumerate(posterior_viability(beta)[0])]
    write_csv(data / "figure_3_viability.csv", ["beta", "tier", "posterior_optimal_for_some_signal"], rows)
    rows = []
    for K in (3, 5, 6):
        betas, losses, hosts, *_ = finite_curve(K)
        rows.extend([K, beta, host, loss] for beta, host, loss in zip(betas, hosts, losses))
    write_csv(data / "figure_3_midl.csv", ["K", "beta", "H", "MIDL"], rows)

    rows = []
    for gamma in np.linspace(0., 1., 101):
        total, effective, covariance, bound = multisegment_values(gamma)
        rows.append([gamma, np.linalg.norm(total - effective), bound, *covariance])
    write_csv(data / "figure_4_covariance.csv", ["gamma", "L2_discrepancy", "CS_bound", "covariance_guild1", "covariance_guild2", "covariance_guild3", "covariance_guild4"], rows)

    ucrit, _ = exact_coexistence_boundary()
    rows = []
    for u in np.unique(np.r_[np.linspace(0., ucrit, 350), np.linspace(ucrit, 1., 450)]):
        x = interior_eq(u) if u <= ucrit else boundary_eq(u)
        leading = float(np.max(np.real(np.linalg.eigvals(glv_jacobian(x, u)))))
        rows.append([u, *x, leading, "coexistence" if u < ucrit else "boundary"])
    write_csv(data / "figure_5_equilibria.csv", ["u", "guild1_abundance", "guild2_abundance", "leading_J_eigenvalue", "branch"], rows)
    rows = []
    for beta in np.linspace(0., .55, 240):
        H, Htau = ecological_frontiers(beta); rows.append([beta, ECO_TAU, H, Htau, H - Htau])
    write_csv(data / "figure_5_frontiers.csv", ["beta", "tau", "H", "H_tau", "screening_gap"], rows)

    summary["figure_files"] = [f"figure_{i}.png" for i in range(1, 6)]
    summary["parameters"] = {
        "binary_ecological_reserve": [0., D_ECO], "binary_ecological_burden": [0., D_ECO],
        "figure_3a_b": {"prior": K5_PRIOR.tolist(), "benefit": K5_BENEFIT.tolist(),
                        "burden": K5_BURDEN.tolist(), "ecological_reserve": K5_BENEFIT.tolist(),
                        "ecological_burden": K5_BENEFIT.tolist()},
        "figure_4": {"weights": MULTI_WEIGHTS.tolist(), "blurs": MULTI_BLURS.tolist(),
                     "compositions": MULTI_COMPOSITIONS.tolist(), "reference": MULTI_REFERENCE.tolist()},
        "figure_5": {"alpha": ECO_ALPHA, "eta": ECO_ETA, "tau": ECO_TAU,
                     "rho": R_GL.tolist(), "growth_shift": B_GL.tolist(), "A": A_GL.tolist()},
    }
    summary["ecological_gap_examples"] = {}
    for beta in (0., .30, binary_switch(ECO_ALPHA, ECO_ETA), .55):
        H, Htau = ecological_frontiers(beta)
        summary["ecological_gap_examples"][str(beta)] = {"H": H, "H_tau": Htau, "gap": H - Htau}
    with (OUT / "analysis_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2); handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "figures",
                        help="Output directory for the five manuscript figures and numerical data")
    args = parser.parse_args(argv)
    OUT = args.out.resolve(); OUT.mkdir(parents=True, exist_ok=True)
    summary = validate_numerics()
    for draw in (figure_1_feasible_triangles, figure_2_binary_midl, figure_3_finite_state,
                 figure_4_multisegment, figure_5_ecological):
        draw(OUT)
    missing = [f"figure_{i}.png" for i in range(1, 6) if not (OUT / f"figure_{i}.png").is_file()]
    if missing:
        raise RuntimeError(f"Missing manuscript figures: {missing}")
    export_analysis(OUT, summary)
    print(f"Validated and generated all 5 manuscript figures in {OUT}")


if __name__ == "__main__":
    main()
