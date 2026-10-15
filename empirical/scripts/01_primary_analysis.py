#!/usr/bin/env python3
"""Primary four-region-to-stool blurring and decision-preservation analysis.

This is the first stage used by ``run_all.py``. It starts from the four raw
Shalon/CapScan exports in ``data/`` and performs:

1. published-data quality control and curated K=11 guild aggregation;
2. construction of complete recovery events with stool and Capsules 1--4;
3. nested leave-one-participant-out fitting of all compositional blurring
   candidates, with every reference and hyperparameter estimated from training
   participants only;
4. subject-level reconstruction summaries and exact paired comparisons; and
5. cross-fitted proxy-state assignments and point-estimate decision-value
   curves.

Bootstrap uncertainty and manuscript sensitivities are deliberately separated
into later scripts so the workflow is restartable and intermediate results are
never confused with completed results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

from data_pipeline import input_manifest, save_prepared_data
from analysis_pipeline import (
    build_events, run_outer_cv, subject_performance, performance_summary,
    decision_point_outputs,
)
from blurring_models import REGIONS
from blurring_models import exact_sign_flip_pvalue, holm_adjust


def tests_by_metric(subject_perf: pd.DataFrame) -> pd.DataFrame:
    # Prespecified contrasts use a geometry-matched population baseline.
    # Probability-scale TV comparisons use the arithmetic stool center;
    # Aitchison comparisons use the training-fold Aitchison stool center.
    comparisons_by_metric = {
        "mean_TV": [
            ("M3_arithmetic_regions_plus_reference", "B0_population_arithmetic_center"),
            ("M6_aitchison_regions_plus_reference", "B0_population_arithmetic_center"),
            ("M7_multiregion_clr_ridge", "B0_population_arithmetic_center"),
            ("M3_arithmetic_regions_plus_reference", "M1_equal_arithmetic_regions"),
            ("M6_aitchison_regions_plus_reference", "M4_equal_aitchison_regions"),
        ],
        "mean_Aitchison": [
            ("M3_arithmetic_regions_plus_reference", "B0_population_aitchison_center"),
            ("M6_aitchison_regions_plus_reference", "B0_population_aitchison_center"),
            ("M7_multiregion_clr_ridge", "B0_population_aitchison_center"),
            ("M3_arithmetic_regions_plus_reference", "M1_equal_arithmetic_regions"),
            ("M6_aitchison_regions_plus_reference", "M4_equal_aitchison_regions"),
        ],
    }
    rows = []
    for metric in ["mean_TV", "mean_Aitchison"]:
        wide = subject_perf[subject_perf["analysis"].eq("primary")].pivot(
            index="Subject", columns="model", values=metric
        )
        family = []
        for a, b in comparisons_by_metric[metric]:
            d = (wide[b] - wide[a]).dropna()  # positive means model a has lower error
            family.append({
                "metric": metric,
                "model_a": a,
                "model_b": b,
                "n_subjects": int(len(d)),
                "mean_improvement_b_minus_a": float(d.mean()),
                "median_improvement_b_minus_a": float(d.median()),
                "p_exact_two_sided": exact_sign_flip_pvalue(d),
            })
        fam = pd.DataFrame(family)
        fam["p_holm"] = holm_adjust(fam["p_exact_two_sided"])
        rows.append(fam)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text())
    seed = int(cfg["seed"])
    delta = float(cfg["clr_zero_replacement"])
    alphas = [float(x) for x in cfg["ridge_alphas"]]

    prepared, events, guild_cols = build_events(ROOT / "data", "K11")

    # Auditable input and preprocessing exports.
    input_manifest(ROOT / "data").to_csv(TABLES / "input_manifest.csv", index=False)
    save_prepared_data(prepared, TABLES)
    pd.DataFrame({"guild": guild_cols}).to_csv(TABLES / "guild_order_K11.csv", index=False)

    raw_meta = pd.read_csv(ROOT / "data" / "phyloseq_alphaDiv_metadata_SAFE.csv")
    cohort_flow = pd.DataFrame([
        {"stage": "raw metadata records", "n": len(raw_meta), "subjects": raw_meta["Subject"].nunique()},
        {"stage": "QC 16S samples", "n": len(prepared.metadata_qc), "subjects": prepared.metadata_qc["Subject"].nunique()},
        {"stage": "complete stool plus four-region events", "n": len(events), "subjects": events["Subject"].nunique()},
    ])
    cohort_flow.to_csv(TABLES / "cohort_flow.csv", index=False)

    wide = events[["event_id", "Subject", "Recover_date_time", "n_capsule_sets_event"]].copy()
    for measurement in ["Stool"] + REGIONS:
        values = np.vstack(events[measurement])
        for j, guild in enumerate(guild_cols):
            wide[f"{measurement}__{guild}"] = values[:, j]
    wide.to_csv(TABLES / "complete_event_guild_compositions.csv", index=False)

    pred, pars, inner, assignments = run_outer_cv(
        events, guild_cols, delta, alphas, analysis_label="primary"
    )
    pred.to_csv(TABLES / "nested_loso_predictions_long.csv", index=False)
    pars.to_csv(TABLES / "nested_loso_parameters_long.csv", index=False)
    inner.to_csv(TABLES / "ridge_inner_cv_paths.csv", index=False)
    assignments.to_csv(TABLES / "crossfitted_decision_assignments.csv", index=False)

    subj = subject_performance(pred)
    subj.to_csv(TABLES / "subject_level_model_performance.csv", index=False)
    perf, perf_boot = performance_summary(
        subj, int(cfg["performance_bootstrap_replicates"]), seed
    )
    perf.to_csv(TABLES / "model_performance_summary.csv", index=False)
    perf_boot.to_csv(TABLES / "model_performance_bootstrap_long.csv", index=False)

    by_metric = tests_by_metric(subj)
    by_metric.to_csv(TABLES / "predefined_exact_model_tests_by_metric.csv", index=False)
    by_metric[by_metric["metric"].eq("mean_Aitchison")].drop(columns="metric").to_csv(
        TABLES / "predefined_exact_model_tests.csv", index=False
    )

    eta = np.linspace(0, 1, int(cfg["eta_grid_n"]))
    curves, summaries, calibration = decision_point_outputs(
        assignments, eta, float(cfg["channel_prior"])
    )
    curves.to_csv(TABLES / "decision_value_curves.csv", index=False)
    summaries.to_csv(TABLES / "decision_value_summary.csv", index=False)
    calibration.to_csv(TABLES / "decision_calibration_error.csv", index=False)

    pmain = perf[(perf["analysis"].eq("primary")) & (perf["metric"].eq("mean_Aitchison"))][
        ["model", "mean", "ci_low", "ci_high"]
    ].rename(columns={
        "mean": "mean_subject_Aitchison",
        "ci_low": "Aitchison_ci_low",
        "ci_high": "Aitchison_ci_high",
    })
    cmean = calibration.groupby("model", as_index=False)["midl_curve_error_vs_observed_stool"].mean().rename(
        columns={"midl_curve_error_vs_observed_stool": "mean_midl_curve_calibration_error"}
    )
    select = pmain.merge(cmean, on="model", how="left")
    flags = []
    for _, r in select.iterrows():
        dominated = (
            (select["mean_subject_Aitchison"] <= r["mean_subject_Aitchison"] + 1e-12)
            & (select["mean_midl_curve_calibration_error"] <= r["mean_midl_curve_calibration_error"] + 1e-12)
            & (
                (select["mean_subject_Aitchison"] < r["mean_subject_Aitchison"] - 1e-12)
                | (select["mean_midl_curve_calibration_error"] < r["mean_midl_curve_calibration_error"] - 1e-12)
            )
        ).any()
        flags.append(not dominated)
    select["pareto_frontier"] = flags
    select.sort_values("mean_subject_Aitchison").to_csv(
        TABLES / "reconstruction_decision_model_selection.csv", index=False
    )

    print(cohort_flow.to_string(index=False))
    print("\nPrimary nested-LOSO analysis complete.")
    print(select.sort_values("mean_subject_Aitchison").to_string(index=False))


if __name__ == "__main__":
    main()
