#!/usr/bin/env python3
"""Sensitivity analyses, regional ablation, and scientific validation checks.

All summaries used for interpretation are participant-level. Event-level files
are retained only as auditable intermediate outputs. This distinction matters
because four participants contribute repeated complete recovery events.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import (
    load_and_prepare,
    aggregate_resolution,
    build_event_cohort,
    subject_mean_cohort,
    K_GROUPS,
)
from analysis_pipeline import run_outer_cv, decision_point_outputs, fit_predict_models
from blurring_models import (
    REGIONS,
    arithmetic_reference,
    aitchison_reference,
    equal_arithmetic,
    equal_aitchison,
    fit_arithmetic_weights,
    predict_arithmetic,
    fit_aitchison_weights,
    predict_aitchison,
    tv_distance,
    js_distance,
    aitchison_distance,
    exact_sign_flip_pvalue,
    holm_adjust,
)
from decision_tasks import (
    TASKS,
    target_score,
    observation_score,
    fit_target_threshold,
    assign_target_state,
    fit_observation_threshold,
)

TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
CFG = json.loads((ROOT / "config.json").read_text())
DELTA = float(CFG["clr_zero_replacement"])
SEED = int(CFG["seed"])
BOOT_REPS = int(CFG["performance_bootstrap_replicates"])


def event_arrays(df: pd.DataFrame, region_names=REGIONS):
    regions = np.stack([np.vstack(df[r].to_numpy()) for r in region_names], axis=1)
    stool = np.vstack(df["Stool"].to_numpy())
    return regions, stool


def participant_weights(subjects: np.ndarray) -> np.ndarray:
    return np.array([1.0 / np.sum(subjects == s) for s in subjects], dtype=float)


def core_cv(events: pd.DataFrame, delta: float, label: str):
    """Reduced nested-LOPO workflow used in retained sensitivity analyses."""
    rows, weights = [], []
    for held in sorted(events["Subject"].unique()):
        train = events[events["Subject"].ne(held)].reset_index(drop=True)
        test = events[events["Subject"].eq(held)].reset_index(drop=True)
        r_train, s_train = event_arrays(train)
        r_test, s_test = event_arrays(test)
        subjects = train["Subject"].to_numpy()
        w_event = participant_weights(subjects)

        ref_a = arithmetic_reference(s_train, sample_weight=w_event)
        ref_g = aitchison_reference(s_train, delta, sample_weight=w_event)
        w_a = fit_arithmetic_weights(r_train, s_train, ref_a, sample_weight=w_event)
        w_g = fit_aitchison_weights(r_train, s_train, delta, ref_g, sample_weight=w_event)

        models = {
            "B0_population_arithmetic_center": np.tile(ref_a, (len(test), 1)),
            "B0_population_aitchison_center": np.tile(ref_g, (len(test), 1)),
            "M1_equal_arithmetic_regions": equal_arithmetic(r_test),
            "M3_arithmetic_regions_plus_reference": predict_arithmetic(r_test, w_a, ref_a),
            "M4_equal_aitchison_regions": equal_aitchison(r_test, delta),
            "M6_aitchison_regions_plus_reference": predict_aitchison(r_test, w_g, delta, ref_g),
        }

        for name, value in zip(REGIONS + ["population_stool_reference"], w_a):
            weights.append({
                "analysis": label,
                "held_subject": held,
                "model": "M3_arithmetic_regions_plus_reference",
                "parameter": name,
                "value": value,
            })
        for name, value in zip(REGIONS + ["population_stool_reference"], w_g):
            weights.append({
                "analysis": label,
                "held_subject": held,
                "model": "M6_aitchison_regions_plus_reference",
                "parameter": name,
                "value": value,
            })

        for model, predictions in models.items():
            for i, prediction in enumerate(predictions):
                rows.append({
                    "analysis": label,
                    "Subject": test.loc[i, "Subject"],
                    "event_id": test.loc[i, "event_id"],
                    "model": model,
                    "TV": tv_distance(s_test[i], prediction),
                    "JS": js_distance(s_test[i], prediction),
                    "Aitchison": aitchison_distance(s_test[i], prediction, delta),
                })
    return pd.DataFrame(rows), pd.DataFrame(weights)


def participant_model_summary(event_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = event_predictions.groupby(["analysis", "Subject", "model"], as_index=False).agg(
        mean_TV=("TV", "mean"),
        mean_JS=("JS", "mean"),
        mean_Aitchison=("Aitchison", "mean"),
        n_events=("event_id", "nunique"),
    )
    summary = subject.groupby(["analysis", "model"], as_index=False).agg(
        mean_TV=("mean_TV", "mean"),
        mean_JS=("mean_JS", "mean"),
        mean_Aitchison=("mean_Aitchison", "mean"),
        n_subjects=("Subject", "nunique"),
    )
    return subject, summary


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    draws = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(BOOT_REPS)
    ])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


# ---------------------------------------------------------------------------
# Reconstruct the primary event cohort.
# ---------------------------------------------------------------------------
prepared = load_and_prepare(ROOT / "data")
guild_meta, guild_cols = aggregate_resolution(prepared.sample_metadata_guild_k11, "K11")
events = build_event_cohort(guild_meta, guild_cols, ["Stool"] + REGIONS)


# ---------------------------------------------------------------------------
# Leave-one-region-out ablation.
# ---------------------------------------------------------------------------
ablation_rows = []
for omitted in REGIONS:
    retained = [r for r in REGIONS if r != omitted]
    for held in sorted(events["Subject"].unique()):
        train = events[events["Subject"].ne(held)].reset_index(drop=True)
        test = events[events["Subject"].eq(held)].reset_index(drop=True)
        r_train, s_train = event_arrays(train, retained)
        r_test, s_test = event_arrays(test, retained)
        subjects = train["Subject"].to_numpy()
        w_event = participant_weights(subjects)
        ref_a = arithmetic_reference(s_train, sample_weight=w_event)
        ref_g = aitchison_reference(s_train, DELTA, sample_weight=w_event)
        w_a = fit_arithmetic_weights(r_train, s_train, ref_a, sample_weight=w_event)
        w_g = fit_aitchison_weights(r_train, s_train, DELTA, ref_g, sample_weight=w_event)
        models = {
            "M3_arithmetic_regions_plus_reference": predict_arithmetic(r_test, w_a, ref_a),
            "M6_aitchison_regions_plus_reference": predict_aitchison(r_test, w_g, DELTA, ref_g),
        }
        for model, predictions in models.items():
            for i, prediction in enumerate(predictions):
                ablation_rows.append({
                    "omitted_region": omitted,
                    "Subject": test.loc[i, "Subject"],
                    "event_id": test.loc[i, "event_id"],
                    "model": model,
                    "TV": tv_distance(s_test[i], prediction),
                    "JS": js_distance(s_test[i], prediction),
                    "Aitchison": aitchison_distance(s_test[i], prediction, DELTA),
                })

ablation = pd.DataFrame(ablation_rows)
ablation.to_csv(TABLES / "region_ablation_predictions.csv", index=False)
ablation_subject = ablation.groupby(["omitted_region", "model", "Subject"], as_index=False).agg(
    mean_TV=("TV", "mean"),
    mean_JS=("JS", "mean"),
    mean_Aitchison=("Aitchison", "mean"),
    n_events=("event_id", "nunique"),
)
ablation_subject.to_csv(TABLES / "region_ablation_subject_performance.csv", index=False)

full_subject = pd.read_csv(TABLES / "subject_level_model_performance.csv")
rng = np.random.default_rng(SEED + 17)
ablation_summary_rows = []
for model, metric in [
    ("M3_arithmetic_regions_plus_reference", "mean_TV"),
    ("M6_aitchison_regions_plus_reference", "mean_Aitchison"),
]:
    full = full_subject[full_subject["model"].eq(model)].set_index("Subject")[metric]
    family_rows = []
    for omitted in REGIONS:
        omitted_values = ablation_subject[
            (ablation_subject["model"].eq(model))
            & (ablation_subject["omitted_region"].eq(omitted))
        ].set_index("Subject")[metric]
        common = full.index.intersection(omitted_values.index)
        differences = (omitted_values.loc[common] - full.loc[common]).to_numpy(float)
        ci_low, ci_high = bootstrap_mean_ci(differences, rng)
        family_rows.append({
            "model": model,
            "metric": metric,
            "omitted_region": omitted,
            "n_subjects": len(common),
            "full_model_mean": float(full.loc[common].mean()),
            "omitted_model_mean": float(omitted_values.loc[common].mean()),
            "mean_change_omitted_minus_full": float(differences.mean()),
            "median_change_omitted_minus_full": float(np.median(differences)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_exact_two_sided": exact_sign_flip_pvalue(differences),
        })
    fam = pd.DataFrame(family_rows)
    fam["p_holm"] = holm_adjust(fam["p_exact_two_sided"])
    ablation_summary_rows.append(fam)

ablation_summary = pd.concat(ablation_summary_rows, ignore_index=True)
ablation_summary.to_csv(TABLES / "region_ablation_summary.csv", index=False)


# ---------------------------------------------------------------------------
# One-profile-per-participant sensitivity.
# ---------------------------------------------------------------------------
subject_mean_events = subject_mean_cohort(events, ["Stool"] + REGIONS)
subject_mean_events["Recover_date_time"] = "subject mean"
subject_mean_events["n_capsule_sets_event"] = np.nan
subject_mean_predictions, _ = core_cv(subject_mean_events, DELTA, "subject_mean")
subject_mean_predictions.to_csv(TABLES / "subject_mean_sensitivity_performance.csv", index=False)
subject_mean_subject, subject_mean_summary = participant_model_summary(subject_mean_predictions)
subject_mean_subject.to_csv(TABLES / "subject_mean_sensitivity_subject_performance.csv", index=False)
subject_mean_summary.to_csv(TABLES / "subject_mean_sensitivity_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Joint guild-resolution by zero-replacement sensitivity.
#
# Amalgamating guilds changes both the biological contrasts retained in the
# composition and its zero pattern.  Treating guild resolution and zero
# replacement as a crossed sensitivity avoids attributing a ranking change to
# biological representation alone when the two choices interact in CLR space.
# ---------------------------------------------------------------------------
resolution_events = {"K11": events}
guild_meta_k6, guild_cols_k6 = aggregate_resolution(prepared.sample_metadata_guild_k11, "K6")
events_k6 = build_event_cohort(guild_meta_k6, guild_cols_k6, ["Stool"] + REGIONS)
resolution_events["K6"] = events_k6

joint_event_frames = []
for resolution, resolution_event_table in resolution_events.items():
    for delta in [1e-8, 1e-6, 1e-4]:
        label = f"{resolution}_delta_{delta:.0e}"
        frame, _ = core_cv(resolution_event_table, delta, label)
        frame["resolution"] = resolution
        frame["delta"] = delta
        joint_event_frames.append(frame)

joint_predictions = pd.concat(joint_event_frames, ignore_index=True)
joint_predictions.to_csv(TABLES / "resolution_zero_sensitivity_predictions.csv", index=False)
joint_subject = joint_predictions.groupby(
    ["resolution", "delta", "Subject", "model"], as_index=False
).agg(
    mean_TV=("TV", "mean"),
    mean_JS=("JS", "mean"),
    mean_Aitchison=("Aitchison", "mean"),
    n_events=("event_id", "nunique"),
)
joint_subject.to_csv(TABLES / "resolution_zero_sensitivity_subject_performance.csv", index=False)
joint_summary = joint_subject.groupby(["resolution", "delta", "model"], as_index=False).agg(
    mean_TV=("mean_TV", "mean"),
    mean_JS=("mean_JS", "mean"),
    mean_Aitchison=("mean_Aitchison", "mean"),
    n_subjects=("Subject", "nunique"),
)
joint_summary.to_csv(TABLES / "resolution_zero_sensitivity_summary.csv", index=False)

rng_joint = np.random.default_rng(SEED + 31)
joint_contrast_rows = []
for (resolution, delta), grp in joint_subject.groupby(["resolution", "delta"], sort=True):
    wide = grp.pivot(index="Subject", columns="model", values="mean_Aitchison")
    center = wide["B0_population_aitchison_center"]
    regional = wide["M6_aitchison_regions_plus_reference"]
    difference = (center - regional).to_numpy(float)  # positive favors regional model
    ci_low, ci_high = bootstrap_mean_ci(difference, rng_joint)
    joint_contrast_rows.append({
        "resolution": resolution,
        "delta": float(delta),
        "n_subjects": len(difference),
        "population_center_mean_Aitchison": float(center.mean()),
        "regional_plus_reference_mean_Aitchison": float(regional.mean()),
        "mean_difference_center_minus_regional": float(difference.mean()),
        "median_difference_center_minus_regional": float(np.median(difference)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "relative_error_reduction_percent": float(100.0 * (1.0 - regional.mean() / center.mean())),
        "p_exact_two_sided": exact_sign_flip_pvalue(difference),
        "n_participants_regional_better": int(np.sum(difference > 0)),
        "n_participants_center_better": int(np.sum(difference < 0)),
    })
joint_contrasts = pd.DataFrame(joint_contrast_rows)
joint_contrasts.to_csv(TABLES / "resolution_zero_sensitivity_paired_contrasts.csv", index=False)

# Backward-compatible K6 and K11 zero-replacement exports used by downstream
# readers.  All are derived from the crossed analysis above.
k6_predictions = joint_predictions[
    joint_predictions["resolution"].eq("K6") & np.isclose(joint_predictions["delta"], DELTA)
].copy()
k6_subject = joint_subject[
    joint_subject["resolution"].eq("K6") & np.isclose(joint_subject["delta"], DELTA)
].copy()
k6_summary = joint_summary[
    joint_summary["resolution"].eq("K6") & np.isclose(joint_summary["delta"], DELTA)
].copy()
k6_predictions.to_csv(TABLES / "K6_sensitivity_performance.csv", index=False)
k6_subject.to_csv(TABLES / "K6_sensitivity_subject_performance.csv", index=False)
k6_summary.to_csv(TABLES / "K6_sensitivity_summary.csv", index=False)

zero_predictions = joint_predictions[joint_predictions["resolution"].eq("K11")].copy()
zero_subject = joint_subject[joint_subject["resolution"].eq("K11")].copy()
zero_summary = joint_summary[joint_summary["resolution"].eq("K11")].copy()
zero_predictions.to_csv(TABLES / "clr_zero_replacement_sensitivity.csv", index=False)
zero_subject.to_csv(TABLES / "clr_zero_replacement_sensitivity_subject_performance.csv", index=False)
zero_summary.to_csv(TABLES / "clr_zero_replacement_sensitivity_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Decision-layer sensitivity at K6.
#
# Each K6 task group is the exact sum of the corresponding K11 task members.
# We verify this equality before refitting the models, then evaluate whether the
# representation used for reconstruction changes stool-MIDL calibration.
# ---------------------------------------------------------------------------
event_k11_by_id = events.set_index("event_id").loc[events_k6["event_id"]].reset_index()
k6_task_equivalence_max_difference = 0.0
for task in TASKS:
    score_k11 = np.array([target_score(row, guild_cols, task) for _, row in event_k11_by_id.iterrows()])
    score_k6 = np.array([target_score(row, guild_cols_k6, task) for _, row in events_k6.iterrows()])
    if not np.allclose(score_k11, score_k6, atol=1e-12):
        raise AssertionError(f"K6 target score is not an exact amalgamation for {task}")
    stool_k11 = np.array([observation_score(x, guild_cols, task) for x in event_k11_by_id["Stool"]])
    stool_k6 = np.array([observation_score(x, guild_cols_k6, task) for x in events_k6["Stool"]])
    k6_task_equivalence_max_difference = max(
        k6_task_equivalence_max_difference,
        float(np.max(np.abs(score_k11 - score_k6))),
        float(np.max(np.abs(stool_k11 - stool_k6))),
    )
    if not np.allclose(stool_k11, stool_k6, atol=1e-12):
        raise AssertionError(f"K6 stool score is not an exact amalgamation for {task}")

k6_full_pred, k6_full_pars, k6_full_inner, k6_assignments = run_outer_cv(
    events_k6, guild_cols_k6, DELTA, [float(x) for x in CFG["ridge_alphas"]],
    analysis_label="K6_decision_sensitivity",
)
k6_assignments.to_csv(TABLES / "K6_crossfitted_decision_assignments.csv", index=False)
eta_grid = np.linspace(0, 1, int(CFG["eta_grid_n"]))
k6_curves, k6_decision_summary, k6_calibration = decision_point_outputs(
    k6_assignments, eta_grid, float(CFG["channel_prior"])
)
k6_curves.to_csv(TABLES / "K6_decision_value_curves.csv", index=False)
k6_decision_summary.to_csv(TABLES / "K6_decision_value_summary.csv", index=False)
k6_calibration.to_csv(TABLES / "K6_decision_calibration_error.csv", index=False)


# ---------------------------------------------------------------------------
# Target-region-omission decision sensitivity.
#
# For each local task, X remains defined from the prespecified target region,
# but that region is excluded from reconstruction model fitting and prediction.
# This distinguishes direct reuse of the state-defining profile from signal
# available through the other three regions.  The hotspot task has no single
# fixed target region and is therefore not included.
# ---------------------------------------------------------------------------
target_omission_rows = []
local_omissions = {
    "proximal_local_burden": "Capsule 1",
    "distal_local_butyrate_deficit": "Capsule 4",
}
for task, omitted_region in local_omissions.items():
    retained_regions = [region for region in REGIONS if region != omitted_region]
    for held in sorted(events["Subject"].unique()):
        train = events[events["Subject"].ne(held)].reset_index(drop=True)
        test = events[events["Subject"].eq(held)].reset_index(drop=True)
        train_predictions, test_predictions, _, _ = fit_predict_models(
            train,
            test,
            DELTA,
            [float(x) for x in CFG["ridge_alphas"]],
            region_names=retained_regions,
        )

        target_threshold = fit_target_threshold(train, guild_cols, task, q=0.5)
        x_train = assign_target_state(train, guild_cols, task, target_threshold)
        x_test = assign_target_state(test, guild_cols, task, target_threshold)

        selected = {
            "Observed_stool": (
                list(train["Stool"]),
                list(test["Stool"]),
            ),
            "M3_arithmetic_regions_plus_reference": (
                list(train_predictions["M3_arithmetic_regions_plus_reference"]),
                list(test_predictions["M3_arithmetic_regions_plus_reference"]),
            ),
            "M6_aitchison_regions_plus_reference": (
                list(train_predictions["M6_aitchison_regions_plus_reference"]),
                list(test_predictions["M6_aitchison_regions_plus_reference"]),
            ),
        }
        for model, (training_profiles, test_profiles) in selected.items():
            train_scores = [observation_score(profile, guild_cols, task) for profile in training_profiles]
            test_scores = [observation_score(profile, guild_cols, task) for profile in test_profiles]
            observation_threshold = fit_observation_threshold(
                train_scores, x_train, train["Subject"]
            )
            observations = (np.asarray(test_scores) >= observation_threshold).astype(int)
            for i in range(len(test)):
                target_omission_rows.append({
                    "analysis": "target_region_omission",
                    "held_subject": held,
                    "event_id": test.loc[i, "event_id"],
                    "Subject": test.loc[i, "Subject"],
                    "task": task,
                    "omitted_region": omitted_region,
                    "model": model,
                    "X": int(x_test[i]),
                    "O": int(observations[i]),
                    "target_threshold": target_threshold,
                    "observation_threshold": observation_threshold,
                })

target_omission_assignments = pd.DataFrame(target_omission_rows)
target_omission_assignments.to_csv(
    TABLES / "target_region_omission_assignments.csv", index=False
)
target_omission_curves, target_omission_summary, target_omission_calibration = decision_point_outputs(
    target_omission_assignments, eta_grid, float(CFG["channel_prior"])
)
target_omission_summary = target_omission_summary.merge(
    target_omission_assignments[["task", "model", "omitted_region"]].drop_duplicates(),
    on=["task", "model"],
    how="left",
    validate="one_to_one",
)
target_omission_curves.to_csv(
    TABLES / "target_region_omission_value_curves.csv", index=False
)
target_omission_summary.to_csv(
    TABLES / "target_region_omission_value_summary.csv", index=False
)
target_omission_calibration.to_csv(
    TABLES / "target_region_omission_calibration_error.csv", index=False
)


# ---------------------------------------------------------------------------
# Consolidated primary parameters and decision-layer contrasts.
# ---------------------------------------------------------------------------
parameters = pd.read_csv(TABLES / "nested_loso_parameters_long.csv")
parameter_summary = (
    parameters.groupby(["model", "parameter"], as_index=False)["value"]
    .agg(["mean", "median", "std", "min", "max"])
    .reset_index()
)
parameter_summary.to_csv(TABLES / "primary_weight_summary.csv", index=False)

boot = pd.read_csv(TABLES / "decision_bootstrap_long.csv")
contrast_models = [
    "B0_population_arithmetic_center",
    "B0_population_aitchison_center",
    "M3_arithmetic_regions_plus_reference",
    "M6_aitchison_regions_plus_reference",
    "M7_multiregion_clr_ridge",
]
contrast_rows = []
for task, task_boot in boot.groupby("task"):
    observed = task_boot[task_boot["model"].eq("Observed_stool")][
        ["replicate", "integrated_midl"]
    ].rename(columns={"integrated_midl": "observed"})
    for model in contrast_models:
        model_boot = task_boot[task_boot["model"].eq(model)][
            ["replicate", "integrated_midl"]
        ].rename(columns={"integrated_midl": "model_value"})
        paired = observed.merge(model_boot, on="replicate", how="inner")
        difference = paired["model_value"] - paired["observed"]
        contrast_rows.append({
            "task": task,
            "model": model,
            "contrast_model_minus_observed_stool_median": float(difference.median()),
            "ci_low": float(difference.quantile(0.025)),
            "ci_high": float(difference.quantile(0.975)),
            "probability_model_lower_loss": float((difference < 0).mean()),
        })
pd.DataFrame(contrast_rows).to_csv(TABLES / "decision_bootstrap_contrasts.csv", index=False)


# ---------------------------------------------------------------------------
# Automated validation checks.
# ---------------------------------------------------------------------------
predictions = pd.read_csv(TABLES / "nested_loso_predictions_long.csv")
assignments = pd.read_csv(TABLES / "crossfitted_decision_assignments.csv")
curves = pd.read_csv(TABLES / "decision_value_curves.csv")
decision_boot = pd.read_csv(TABLES / "decision_bootstrap_long.csv")
checks = []


def check(name: str, condition: bool, detail):
    checks.append({"check": name, "passed": bool(condition), "detail": str(detail)})
    if not condition:
        raise AssertionError(f"{name}: {detail}")


check("QC sample count", len(prepared.metadata_qc) == 297, len(prepared.metadata_qc))
check("complete events", len(events) == 16, len(events))
check("complete subjects", events["Subject"].nunique() == 11, events["Subject"].nunique())
check("guilds", len(guild_cols) == 11, len(guild_cols))
check(
    "prediction closure",
    np.allclose(predictions.filter(like="pred__").sum(axis=1), 1, atol=1e-8),
    "closed",
)
check(
    "prediction uniqueness",
    predictions.groupby(["event_id", "model"]).size().eq(1).all(),
    "one prediction per event and model",
)
check(
    "decision states supported",
    assignments.groupby("task")["X"].nunique().eq(2).all(),
    assignments.groupby("task")["X"].nunique().to_dict(),
)
check("MIDL nonnegative", (curves["midl"] >= -1e-10).all(), curves["midl"].min())
check(
    "MIDL endpoints",
    curves[curves["eta"].isin([0.0, 1.0])]["midl"].abs().max() < 1e-8,
    curves[curves["eta"].isin([0.0, 1.0])]["midl"].abs().max(),
)
check(
    "decision bootstrap replicates",
    decision_boot["replicate"].nunique() == int(CFG["decision_bootstrap_replicates"]),
    decision_boot["replicate"].nunique(),
)
check("four ablations", ablation["omitted_region"].nunique() == 4, ablation["omitted_region"].unique())
check(
    "ablation subject counts",
    ablation_subject.groupby(["omitted_region", "model"])["Subject"].nunique().eq(11).all(),
    ablation_subject.groupby(["omitted_region", "model"])["Subject"].nunique().to_dict(),
)
check(
    "K6 sensitivity subject counts",
    k6_subject.groupby("model")["Subject"].nunique().eq(11).all(),
    k6_subject.groupby("model")["Subject"].nunique().to_dict(),
)
check(
    "zero-replacement sensitivity subject counts",
    zero_subject.groupby(["delta", "model"])["Subject"].nunique().eq(11).all(),
    zero_subject.groupby(["delta", "model"])["Subject"].nunique().to_dict(),
)
check(
    "crossed sensitivity combinations",
    joint_contrasts.groupby(["resolution", "delta"]).size().eq(1).all()
    and len(joint_contrasts) == 6,
    joint_contrasts[["resolution", "delta"]].to_dict("records"),
)
check(
    "crossed sensitivity participant counts",
    joint_contrasts["n_subjects"].eq(11).all(),
    joint_contrasts.set_index(["resolution", "delta"])["n_subjects"].to_dict(),
)
check(
    "K6 decision tasks supported",
    k6_assignments.groupby("task")["X"].nunique().eq(2).all(),
    k6_assignments.groupby("task")["X"].nunique().to_dict(),
)
check(
    "K6 task-score exact amalgamation",
    k6_task_equivalence_max_difference < 1e-12,
    k6_task_equivalence_max_difference,
)
check(
    "K6 MIDL nonnegative",
    (k6_curves["midl"] >= -1e-10).all(),
    k6_curves["midl"].min(),
)
check(
    "target-region omission tasks supported",
    target_omission_assignments.groupby("task")["X"].nunique().eq(2).all(),
    target_omission_assignments.groupby("task")["X"].nunique().to_dict(),
)
check(
    "target-region omission MIDL nonnegative",
    (target_omission_curves["midl"] >= -1e-10).all(),
    target_omission_curves["midl"].min(),
)
ridge_alpha = parameters[parameters["parameter"].eq("ridge_alpha")]["value"]
check("ridge optimum inside grid", ridge_alpha.max() < max(CFG["ridge_alphas"]), ridge_alpha.max())

pd.DataFrame(checks).to_csv(TABLES / "workflow_validation_checks.csv", index=False)
print(f"completed {len(checks)} checks")
