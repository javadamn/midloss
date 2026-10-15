#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

COL = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "gray": "#7A7A7A", "black": "#222222", "yellow": "#F0E442",
}
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    png_path = FIGURES / f"{name}.png"
    pdf_path = FIGURES / f"{name}.pdf"
    png_tmp = FIGURES / f".{name}.tmp.png"
    pdf_tmp = FIGURES / f".{name}.tmp.pdf"
    fig.savefig(png_tmp, format="png", dpi=350, bbox_inches="tight")
    fig.savefig(pdf_tmp, format="pdf", bbox_inches="tight")
    if png_tmp.stat().st_size == 0 or pdf_tmp.stat().st_size == 0:
        raise RuntimeError(f"Figure export failed for {name}")
    png_tmp.replace(png_path)
    pdf_tmp.replace(pdf_path)
    plt.close(fig)


LABELS = {
    "B0_population_arithmetic_center": "Population arithmetic center",
    "B0_population_aitchison_center": "Population Aitchison center",
    "B_capsule_1": "Capsule 1", "B_capsule_2": "Capsule 2",
    "B_capsule_3": "Capsule 3", "B_capsule_4": "Capsule 4",
    "B_distal_arithmetic_C3_C4": "Capsule 3/4 mean",
    "M1_equal_arithmetic_regions": "Equal arithmetic mean",
    "M2_learned_arithmetic_regions": "Learned arithmetic mixture",
    "M3_arithmetic_regions_plus_reference": "Arithmetic + reference",
    "M4_equal_aitchison_regions": "Equal Aitchison center",
    "M5_learned_aitchison_regions": "Learned Aitchison mixture",
    "M6_aitchison_regions_plus_reference": "Aitchison + reference",
    "M7_multiregion_clr_ridge": "Multi-region CLR ridge",
}




# Figure 2: performance and composition-versus-calibration frontier.
perf = pd.read_csv(TABLES / "model_performance_summary.csv")
selection = pd.read_csv(TABLES / "reconstruction_decision_model_selection.csv")
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
for ax, metric, title in zip(
    axes[:2], ["mean_Aitchison", "mean_TV"],
    ["Aitchison reconstruction error", "Total-variation reconstruction error"],
):
    d = perf.query("analysis=='primary' and metric==@metric").sort_values("mean")
    y = np.arange(len(d))
    colors = [COL["green"] if m in ["M6_aitchison_regions_plus_reference", "M7_multiregion_clr_ridge"]
              else COL["gray"] for m in d.model]
    ax.errorbar(d["mean"], y, xerr=[d["mean"] - d["ci_low"], d["ci_high"] - d["mean"]],
                fmt="none", ecolor=COL["gray"], capsize=2, lw=1)
    ax.scatter(d["mean"], y, c=colors, s=35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(m, m) for m in d.model])
    ax.invert_yaxis()
    ax.set_xlabel("Mean participant-level error (95% bootstrap CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.2)
ax = axes[2]
for _, row in selection.iterrows():
    color = COL["green"] if row.pareto_frontier else COL["gray"]
    x = row.mean_subject_Aitchison
    y = row.mean_midl_curve_calibration_error
    ax.scatter(x, y, c=color, s=45)
    if row.pareto_frontier or row.model in ["B0_population_arithmetic_center", "B0_population_aitchison_center",
                                            "M3_arithmetic_regions_plus_reference"]:
        dx = -92 if row.model in ["B0_population_arithmetic_center", "B0_population_aitchison_center"] else (
            4 if row.model != "M3_arithmetic_regions_plus_reference" else -105
        )
        ax.annotate(LABELS.get(row.model, row.model), (x, y), xytext=(dx, 4),
                    textcoords="offset points", fontsize=7)
ax.set_xlabel("Mean participant-level Aitchison error")
ax.set_ylabel("Mean MIDL-curve calibration error\nrelative to observed stool")
ax.set_title("Reconstruction vs decision calibration")
ax.grid(alpha=.2)
for i, ax in enumerate(axes):
    ax.text(-.13, 1.04, chr(65 + i), transform=ax.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "Figure2_model_performance")


# Figure 3: model weights and participant-level ablation.
parameters = pd.read_csv(TABLES / "nested_loso_parameters_long.csv")
ablation = pd.read_csv(TABLES / "region_ablation_summary.csv")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, model, title in [
    (axes[0], "M3_arithmetic_regions_plus_reference", "Arithmetic mixture + reference"),
    (axes[1], "M6_aitchison_regions_plus_reference", "Aitchison mixture + reference"),
]:
    d = parameters[parameters.model.eq(model)].copy()
    d["component"] = d.parameter.str.replace("weight_", "", regex=False)
    order = ["Capsule 1", "Capsule 2", "Capsule 3", "Capsule 4", "population_stool_reference"]
    bp = ax.boxplot([d[d.component.eq(x)].value for x in order],
                    tick_labels=["C1", "C2", "C3", "C4", "Reference"],
                    showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#D9EAF4")
        patch.set_edgecolor(COL["blue"])
    ax.set_ylim(-.02, 1.02)
    ax.set_ylabel("Outer-fold simplex weight")
    ax.set_title(title)
    ax.grid(axis="y", alpha=.2)
region_order = ["Capsule 1", "Capsule 2", "Capsule 3", "Capsule 4"]
d = ablation[(ablation.model.eq("M6_aitchison_regions_plus_reference")) &
             (ablation.metric.eq("mean_Aitchison"))].set_index("omitted_region").loc[region_order]
y = d["mean_change_omitted_minus_full"].to_numpy(float)
lo = y - d["ci_low"].to_numpy(float)
hi = d["ci_high"].to_numpy(float) - y
axes[2].bar(np.arange(4), y, color=[COL["sky"], COL["blue"], COL["purple"], COL["orange"]])
axes[2].errorbar(np.arange(4), y, yerr=[lo, hi], fmt="none", ecolor=COL["black"], capsize=3, lw=1)
axes[2].set_xticks(np.arange(4))
axes[2].set_xticklabels(["C1", "C2", "C3", "C4"])
axes[2].axhline(0, color=COL["black"], lw=.8)
axes[2].set_ylabel("Participant-level change in mean\nAitchison error after omission")
axes[2].set_title("Regional ablation of Aitchison model")
axes[2].grid(axis="y", alpha=.2)
for i, ax in enumerate(axes):
    ax.text(-.13, 1.04, chr(65 + i), transform=ax.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "Figure3_weights_ablation")


# Figure 4: threshold-specific MIDL curves.
curves = pd.read_csv(TABLES / "decision_value_curves.csv")
tasks = ["proximal_local_burden", "distal_local_butyrate_deficit", "regional_hotspot_bacteroidales"]
# task_titles = ["Proximal-local oxygen-tolerant/sulfidogenic proxy",
#                "Distal-local low butyrate-associated proxy",
#                "Regional Bacteroidales/Prevotella hotspot proxy"]
task_titles = ["Proximal-local proxy",
               "Distal-local proxy",
               "Regional hotspot proxy"]
models = ["Observed_stool", "B0_population_arithmetic_center",
          "M6_aitchison_regions_plus_reference", "M7_multiregion_clr_ridge"]
colors = {"Observed_stool": COL["black"], "B0_population_arithmetic_center": COL["gray"],
          "M6_aitchison_regions_plus_reference": COL["green"], "M7_multiregion_clr_ridge": COL["blue"]}
styles = {"Observed_stool": "-", "B0_population_arithmetic_center": "--",
          "M6_aitchison_regions_plus_reference": "-", "M7_multiregion_clr_ridge": "-."}
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
for ax, task, title in zip(axes, tasks, task_titles):
    for model in models:
        d = curves[(curves.task.eq(task)) & (curves.model.eq(model))]
        label = "Observed stool" if model == "Observed_stool" else LABELS.get(model, model).replace("Multi-region", "CLR ridge")
        ax.plot(d.eta, d.midl, color=colors[model], ls=styles[model], lw=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Intervention threshold $\\eta$")
    ax.grid(alpha=.2)
axes[0].set_ylabel("Threshold-specific MIDL")
axes[-1].legend(frameon=False, loc="lower center")
for i, ax in enumerate(axes):
    ax.text(-.12, 1.04, chr(65 + i), transform=ax.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "Figure4_decision_curves")


# Figure 5: integrated MIDL and calibration to observed stool.
bootstrap = pd.read_csv(TABLES / "decision_bootstrap_summary.csv")
calibration = pd.read_csv(TABLES / "decision_calibration_error.csv")
keep = ["Observed_stool", "B0_population_arithmetic_center", "M3_arithmetic_regions_plus_reference",
        "M6_aitchison_regions_plus_reference", "M7_multiregion_clr_ridge"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
x = np.arange(len(tasks))
width = .15
for j, model in enumerate(keep):
    d = bootstrap[bootstrap.model.eq(model)].set_index("task").reindex(tasks)
    y = d.integrated_midl_median.to_numpy()
    lo = y - d.integrated_midl_ci_low.to_numpy()
    hi = d.integrated_midl_ci_high.to_numpy() - y
    color = [COL["black"], COL["gray"], COL["orange"], COL["green"], COL["blue"]][j]
    label = "Observed stool" if model == "Observed_stool" else LABELS.get(model, model)
    axes[0].errorbar(x + (j - 2) * width, y, yerr=[lo, hi], fmt="o", color=color,
                     capsize=2, label=label, ms=5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Proximal local", "Distal low state", "Regional hotspot"])
axes[0].set_ylabel("Integrated MIDL across $\\eta$")
# axes[0].set_title("Decision preservation with participant bootstrap intervals")
axes[0].grid(axis="y", alpha=.2)
axes[0].legend(frameon=False, fontsize=7)
d = calibration[calibration.model.isin(keep[1:])].groupby("model").midl_curve_error_vs_observed_stool.mean().sort_values()
axes[1].barh([LABELS.get(m, m) for m in d.index], d.values,
             color=[COL["blue"] if m == "M7_multiregion_clr_ridge" else COL["gray"] for m in d.index])
axes[1].set_xlabel("Mean integrated absolute MIDL-curve error")
# axes[1].set_title("Calibration to observed-stool MIDL curves")
axes[1].grid(axis="x", alpha=.2)
for i, ax in enumerate(axes):
    ax.text(-.1, 1.04, chr(65 + i), transform=ax.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "Figure5_decision_summary")


# Supplementary Figure S1: participant-level model errors.
subject_perf = pd.read_csv(TABLES / "subject_level_model_performance.csv")
mods = ["B0_population_aitchison_center", "M1_equal_arithmetic_regions",
        "M3_arithmetic_regions_plus_reference", "M6_aitchison_regions_plus_reference",
        "M7_multiregion_clr_ridge"]
fig, ax = plt.subplots(figsize=(9, 5))
subject_order = sorted(subject_perf.Subject.unique())
for j, model in enumerate(mods):
    d = subject_perf[subject_perf.model.eq(model)].set_index("Subject").reindex(subject_order)
    ax.plot(np.arange(len(subject_order)) + j * .04, d.mean_Aitchison, "o-", lw=.7, ms=4,
            label=LABELS.get(model, model))
ax.set_xticks(range(len(subject_order)))
ax.set_xticklabels([str(x) for x in subject_order])
ax.set_xlabel("Participant")
ax.set_ylabel("Mean held-participant Aitchison error")
ax.legend(frameon=False, fontsize=7, ncol=2)
ax.grid(alpha=.2)
fig.tight_layout()
save(fig, "FigureS1_subject_errors")


# Supplementary Figure S2: representation and numerical sensitivities.
subject_mean = pd.read_csv(TABLES / "subject_mean_sensitivity_summary.csv")
joint = pd.read_csv(TABLES / "resolution_zero_sensitivity_paired_contrasts.csv")
fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.3))
q = subject_mean.set_index("model")["mean_Aitchison"].sort_values()
axes[0].barh([LABELS.get(m, m) for m in q.index], q.values, color=COL["sky"])
axes[0].set_xlabel("Mean participant-level Aitchison error")
axes[0].set_title("One profile per participant")
axes[0].grid(axis="x", alpha=.2)
primary_delta = joint[np.isclose(joint.delta, 1e-6)].set_index("resolution").loc[["K11", "K6"]]
y = primary_delta.mean_difference_center_minus_regional.to_numpy()
lo = y - primary_delta.ci_low.to_numpy()
hi = primary_delta.ci_high.to_numpy() - y
axes[1].errorbar([0, 1], y, yerr=[lo, hi], fmt="o", color=COL["purple"], capsize=4, ms=6)
axes[1].axhline(0, color=COL["black"], lw=.9)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(["11 guilds", "6 guilds"])
axes[1].set_ylabel("Center minus regional-model\nAitchison error (95% bootstrap CI)")
axes[1].set_title("Paired guild-resolution contrast")
axes[1].grid(axis="y", alpha=.2)
for resolution, color in [("K11", COL["blue"]), ("K6", COL["orange"])]:
    d = joint[joint.resolution.eq(resolution)].sort_values("delta")
    axes[2].plot(d.delta, d.relative_error_reduction_percent, "o-", color=color,
                 label=resolution.replace("K", "K="))
axes[2].axhline(0, color=COL["black"], lw=.9)
axes[2].set_xscale("log")
axes[2].set_xlabel("Zero-replacement value")
axes[2].set_ylabel("Regional-model error reduction\nrelative to population center (%)")
axes[2].set_title("Resolution by zero-replacement sensitivity")
axes[2].legend(frameon=False)
axes[2].grid(alpha=.2)
for i, ax in enumerate(axes):
    ax.text(-.12, 1.04, chr(65 + i), transform=ax.transAxes, fontweight="bold", fontsize=13)
fig.tight_layout()
save(fig, "FigureS2_sensitivities")


# Supplementary Figure S3: inner-validation ridge paths.
ridge_paths = pd.read_csv(TABLES / "ridge_inner_cv_paths.csv")
fig, ax = plt.subplots(figsize=(8.2, 5.0))
for _, d in ridge_paths.groupby("held_subject"):
    d = d.sort_values("alpha")
    ax.plot(d.alpha, d.inner_mean_subject_aitchison, color=COL["gray"], alpha=.42, lw=.9)
mean_path = ridge_paths.groupby("alpha", as_index=False).inner_mean_subject_aitchison.mean().sort_values("alpha")
ax.plot(mean_path.alpha, mean_path.inner_mean_subject_aitchison, "o-", color=COL["blue"], lw=2.2,
        label="Mean across outer folds")
selected = mean_path.loc[mean_path.inner_mean_subject_aitchison.idxmin()]
ax.axvline(selected.alpha, color=COL["green"], ls="--", lw=1.2, label=f"Selected $\\lambda={selected.alpha:g}$")
ax.set_xscale("log")
ax.set_xlabel("Ridge penalty $\\lambda$")
ax.set_ylabel("Inner-validation mean participant Aitchison error")
ax.set_title("Nested ridge-penalty selection")
ax.legend(frameon=False)
ax.grid(alpha=.2)
fig.tight_layout()
save(fig, "FigureS3_ridge_inner_cv_paths")

print("figures generated")
