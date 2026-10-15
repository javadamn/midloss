"""Reusable primary analysis functions.

This module contains only the analyses retained in the manuscript: complete
four-region event construction, nested leave-one-participant-out reconstruction,
cross-fitted decision assignments, and subject-level performance summaries.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from data_pipeline import load_and_prepare, aggregate_resolution, build_event_cohort
from blurring_models import (
    REGIONS, closure, arithmetic_reference, aitchison_reference,
    equal_arithmetic, equal_aitchison, fit_arithmetic_weights, predict_arithmetic,
    fit_aitchison_weights, predict_aitchison, select_ridge_alpha, fit_ridge,
    predict_ridge, tv_distance, js_distance, aitchison_distance,
)
from decision_tasks import (
    TASKS, observation_score, fit_target_threshold, assign_target_state,
    fit_observation_threshold, weighted_channel, decision_curve,
    summarize_curve, curve_distance,
)


def build_events(data_dir: Path, resolution: str = "K11"):
    prepared = load_and_prepare(data_dir)
    gm, guild_cols = aggregate_resolution(prepared.sample_metadata_guild_k11, resolution)
    events = build_event_cohort(gm, guild_cols, ["Stool"] + REGIONS)
    if events.empty:
        raise RuntimeError("No complete stool plus four-capsule events were reconstructed.")
    return prepared, events, guild_cols


def fit_predict_models(train: pd.DataFrame, test: pd.DataFrame, delta: float, ridge_alphas: List[float], region_names: List[str] = REGIONS):
    tr_regions = np.stack([np.vstack(train[r].to_numpy()) for r in region_names], axis=1)
    te_regions = np.stack([np.vstack(test[r].to_numpy()) for r in region_names], axis=1)
    tr_stool = np.vstack(train["Stool"].to_numpy())
    train_subjects = train["Subject"].to_numpy()
    train_weight = np.array([1.0 / np.sum(train_subjects == s) for s in train_subjects])
    ref_a = arithmetic_reference(tr_stool, sample_weight=train_weight)
    ref_g = aitchison_reference(tr_stool, delta, sample_weight=train_weight)

    train_pred: Dict[str, np.ndarray] = {}
    test_pred: Dict[str, np.ndarray] = {}
    params: List[Dict[str, object]] = []
    inner_rows: List[Dict[str, object]] = []

    # Geometry-matched no-regional-information baselines.  The arithmetic
    # center is the natural comparator for probability-scale models, whereas
    # the Aitchison center minimizes squared Aitchison distance in clr space.
    train_pred["B0_population_arithmetic_center"] = np.tile(ref_a, (len(train), 1))
    test_pred["B0_population_arithmetic_center"] = np.tile(ref_a, (len(test), 1))
    train_pred["B0_population_aitchison_center"] = np.tile(ref_g, (len(train), 1))
    test_pred["B0_population_aitchison_center"] = np.tile(ref_g, (len(test), 1))
    if region_names == REGIONS:
        for r in REGIONS:
            key = f"B_{r.replace(' ', '_').lower()}"
            train_pred[key] = np.vstack(train[r].to_numpy())
            test_pred[key] = np.vstack(test[r].to_numpy())
        train_pred["B_distal_arithmetic_C3_C4"] = closure((np.vstack(train["Capsule 3"]) + np.vstack(train["Capsule 4"])) / 2)
        test_pred["B_distal_arithmetic_C3_C4"] = closure((np.vstack(test["Capsule 3"]) + np.vstack(test["Capsule 4"])) / 2)

    train_pred["M1_equal_arithmetic_regions"] = equal_arithmetic(tr_regions)
    test_pred["M1_equal_arithmetic_regions"] = equal_arithmetic(te_regions)

    w = fit_arithmetic_weights(tr_regions, tr_stool, None, sample_weight=train_weight)
    train_pred["M2_learned_arithmetic_regions"] = predict_arithmetic(tr_regions, w, None)
    test_pred["M2_learned_arithmetic_regions"] = predict_arithmetic(te_regions, w, None)
    params.extend({"model":"M2_learned_arithmetic_regions","parameter":f"weight_{name}","value":val} for name,val in zip(region_names,w))

    w = fit_arithmetic_weights(tr_regions, tr_stool, ref_a, sample_weight=train_weight)
    train_pred["M3_arithmetic_regions_plus_reference"] = predict_arithmetic(tr_regions, w, ref_a)
    test_pred["M3_arithmetic_regions_plus_reference"] = predict_arithmetic(te_regions, w, ref_a)
    params.extend({"model":"M3_arithmetic_regions_plus_reference","parameter":f"weight_{name}","value":val} for name,val in zip(region_names+["population_stool_reference"],w))

    train_pred["M4_equal_aitchison_regions"] = equal_aitchison(tr_regions, delta)
    test_pred["M4_equal_aitchison_regions"] = equal_aitchison(te_regions, delta)

    w = fit_aitchison_weights(tr_regions, tr_stool, delta, None, sample_weight=train_weight)
    train_pred["M5_learned_aitchison_regions"] = predict_aitchison(tr_regions, w, delta, None)
    test_pred["M5_learned_aitchison_regions"] = predict_aitchison(te_regions, w, delta, None)
    params.extend({"model":"M5_learned_aitchison_regions","parameter":f"weight_{name}","value":val} for name,val in zip(region_names,w))

    w = fit_aitchison_weights(tr_regions, tr_stool, delta, ref_g, sample_weight=train_weight)
    train_pred["M6_aitchison_regions_plus_reference"] = predict_aitchison(tr_regions, w, delta, ref_g)
    test_pred["M6_aitchison_regions_plus_reference"] = predict_aitchison(te_regions, w, delta, ref_g)
    params.extend({"model":"M6_aitchison_regions_plus_reference","parameter":f"weight_{name}","value":val} for name,val in zip(region_names+["population_stool_reference"],w))

    alpha, path = select_ridge_alpha(tr_regions, tr_stool, train["Subject"].to_numpy(), ridge_alphas, delta)
    ridge = fit_ridge(tr_regions, tr_stool, alpha, delta, sample_weight=train_weight)
    train_pred["M7_multiregion_clr_ridge"] = predict_ridge(ridge, tr_regions)
    test_pred["M7_multiregion_clr_ridge"] = predict_ridge(ridge, te_regions)
    params.append({"model":"M7_multiregion_clr_ridge","parameter":"ridge_alpha","value":alpha})
    inner_rows.extend(path)
    return train_pred, test_pred, params, inner_rows


def run_outer_cv(events: pd.DataFrame, guild_cols: List[str], delta: float, ridge_alphas: List[float], analysis_label: str = "primary"):
    pred_rows, param_rows, inner_rows, decision_rows = [], [], [], []
    held_seen = []
    for held in sorted(events["Subject"].unique()):
        train = events[events["Subject"].ne(held)].copy().reset_index(drop=True)
        test = events[events["Subject"].eq(held)].copy().reset_index(drop=True)
        if train["Subject"].eq(held).any():
            raise AssertionError("held participant leaked into training data")
        held_seen.append(held)
        tr_pred, te_pred, pars, inner = fit_predict_models(train, test, delta, ridge_alphas)
        param_rows.extend({"analysis":analysis_label,"held_subject":held,**row} for row in pars)
        inner_rows.extend({"analysis":analysis_label,"held_subject":held,**row} for row in inner)

        y_test = np.vstack(test["Stool"].to_numpy())
        for model, pred in te_pred.items():
            if not np.allclose(pred.sum(axis=1), 1.0, atol=1e-9):
                raise AssertionError(f"prediction does not close for {model}")
            for i,p in enumerate(pred):
                pred_rows.append({
                    "analysis":analysis_label,"held_subject":held,"event_id":test.loc[i,"event_id"],"Subject":test.loc[i,"Subject"],"model":model,
                    "TV":tv_distance(y_test[i],p),"JS":js_distance(y_test[i],p),"Aitchison":aitchison_distance(y_test[i],p,delta),
                    **{f"pred__{g}":p[j] for j,g in enumerate(guild_cols)},
                })

        for task in TASKS:
            target_threshold = fit_target_threshold(train, guild_cols, task, q=0.5)
            x_train = assign_target_state(train, guild_cols, task, target_threshold)
            x_test = assign_target_state(test, guild_cols, task, target_threshold)

            actual_train_scores = [observation_score(p, guild_cols, task) for p in train["Stool"]]
            actual_test_scores = [observation_score(p, guild_cols, task) for p in test["Stool"]]
            actual_threshold = fit_observation_threshold(actual_train_scores, x_train, train["Subject"])
            actual_o_test = (np.asarray(actual_test_scores) >= actual_threshold).astype(int)
            for i in range(len(test)):
                decision_rows.append({
                    "analysis":analysis_label,"held_subject":held,"event_id":test.loc[i,"event_id"],"Subject":test.loc[i,"Subject"],
                    "task":task,"model":"Observed_stool","X":int(x_test[i]),"O":int(actual_o_test[i]),
                    "target_threshold":target_threshold,"observation_threshold":actual_threshold,
                })

            for model in te_pred:
                tr_scores = [observation_score(p, guild_cols, task) for p in tr_pred[model]]
                te_scores = [observation_score(p, guild_cols, task) for p in te_pred[model]]
                obs_threshold = fit_observation_threshold(tr_scores, x_train, train["Subject"])
                o_test = (np.asarray(te_scores) >= obs_threshold).astype(int)
                for i in range(len(test)):
                    decision_rows.append({
                        "analysis":analysis_label,"held_subject":held,"event_id":test.loc[i,"event_id"],"Subject":test.loc[i,"Subject"],
                        "task":task,"model":model,"X":int(x_test[i]),"O":int(o_test[i]),
                        "target_threshold":target_threshold,"observation_threshold":obs_threshold,
                    })

    if set(held_seen) != set(events["Subject"].unique()):
        raise AssertionError("outer folds did not cover every participant")
    return pd.DataFrame(pred_rows), pd.DataFrame(param_rows), pd.DataFrame(inner_rows), pd.DataFrame(decision_rows)


def subject_performance(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.groupby(["analysis","Subject","model"],as_index=False).agg(
        mean_TV=("TV","mean"),mean_JS=("JS","mean"),mean_Aitchison=("Aitchison","mean"),n_events=("event_id","nunique")
    )


def performance_summary(subject_perf: pd.DataFrame, reps: int, seed: int):
    rng=np.random.default_rng(seed); rows=[]; boot=[]
    for (analysis,model),grp in subject_perf.groupby(["analysis","model"]):
        for metric in ["mean_TV","mean_JS","mean_Aitchison"]:
            vals=grp[metric].to_numpy(float)
            draws=np.array([np.mean(rng.choice(vals,size=len(vals),replace=True)) for _ in range(reps)])
            rows.append({"analysis":analysis,"model":model,"metric":metric,"mean":float(np.mean(vals)),"median":float(np.median(vals)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"n_subjects":len(vals)})
            boot.extend({"analysis":analysis,"model":model,"metric":metric,"replicate":j,"value":v} for j,v in enumerate(draws))
    return pd.DataFrame(rows),pd.DataFrame(boot)


def decision_point_outputs(assignments: pd.DataFrame, eta_grid: np.ndarray, prior: float):
    """Construct cross-fitted channels, MIDL curves, and stool-calibration errors."""
    curves, summaries, calibration = [], [], []
    for task, gt in assignments.groupby("task", sort=True):
        by_model = {}
        for model, g in gt.groupby("model", sort=True):
            counts, channel, phi = weighted_channel(g["X"], g["O"], g["Subject"], prior)
            curve = decision_curve(channel, phi, eta_grid)
            curve["task"] = task
            curve["model"] = model
            curves.append(curve)
            by_model[model] = curve
            summaries.append({
                "task": task,
                "model": model,
                **summarize_curve(curve),
                "phi1": float(phi[1]),
                "state_agreement": float(np.mean(g["X"].to_numpy() == g["O"].to_numpy())),
                "count_00": float(counts[0, 0]),
                "count_01": float(counts[0, 1]),
                "count_10": float(counts[1, 0]),
                "count_11": float(counts[1, 1]),
                "T_00": float(channel[0, 0]),
                "T_01": float(channel[0, 1]),
                "T_10": float(channel[1, 0]),
                "T_11": float(channel[1, 1]),
            })
        observed = by_model["Observed_stool"]
        for model, curve in by_model.items():
            if model != "Observed_stool":
                calibration.append({
                    "task": task,
                    "model": model,
                    "midl_curve_error_vs_observed_stool": curve_distance(curve, observed),
                })
    return pd.concat(curves, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(calibration)
