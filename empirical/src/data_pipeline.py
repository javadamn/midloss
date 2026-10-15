from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import json

import numpy as np
import pandas as pd

from aim3_common import (
    CURATED_GUILD_ORDER,
    CURATED_GUILD_RUBRIC,
    map_taxon_to_curated_guild,
    normalize_rows,
)

CAPS = ["Capsule 1", "Capsule 2", "Capsule 3", "Capsule 4"]

# Revised biologically coherent coarsenings. These are fixed before model fitting.
K_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "K11": {g: [g] for g in CURATED_GUILD_ORDER},
    "K6": {
        "K6_G1_sulfidogenic_proximal": [
            "G01_sulfidogenic_bile_proteolytic",
            "G02_oxygen_tolerant_proximal_oral",
        ],
        "K6_G2_bacteroidales_prevotella": [
            "G03_bacteroidales_generalist",
            "G04_prevotella_carbohydrate_propionate",
        ],
        "K6_G3_lactate_acetate_propionate": [
            "G05_lactate_utilizers_propionate",
            "G06_bifidobacteria_lactate_acetate",
        ],
        "K6_G4_fiber_butyrate_crossfeeders": [
            "G07_ruminococcaceae_fiber_butyrate",
            "G09_lachnospiraceae_butyrate_crossfeeders",
        ],
        "K6_G5_mucin_barrier": ["G08_mucin_glycan_barrier"],
        "K6_G6_other_uncertain": [
            "G10_other_anaerobic_fermenters_uncertain",
            "G11_unassigned_low_information",
        ],
    },
    "K4": {
        "K4_G1_sulfidogenic_proximal": [
            "G01_sulfidogenic_bile_proteolytic",
            "G02_oxygen_tolerant_proximal_oral",
        ],
        "K4_G2_generalist_propionate": [
            "G03_bacteroidales_generalist",
            "G04_prevotella_carbohydrate_propionate",
            "G05_lactate_utilizers_propionate",
            "G06_bifidobacteria_lactate_acetate",
        ],
        "K4_G3_butyrate_mucin_barrier": [
            "G07_ruminococcaceae_fiber_butyrate",
            "G08_mucin_glycan_barrier",
            "G09_lachnospiraceae_butyrate_crossfeeders",
        ],
        "K4_G4_other_uncertain": [
            "G10_other_anaerobic_fermenters_uncertain",
            "G11_unassigned_low_information",
        ],
    },
}


@dataclass
class PreparedData:
    metadata_qc: pd.DataFrame
    taxonomy_mapping: pd.DataFrame
    sample_guild_counts_k11: pd.DataFrame
    sample_guild_rel_k11: pd.DataFrame
    sample_metadata_guild_k11: pd.DataFrame


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_manifest(data_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(data_dir.iterdir()):
        if p.is_file():
            rows.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return pd.DataFrame(rows)


def _bool_false(series: pd.Series) -> pd.Series:
    return ~series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_and_prepare(
    data_dir: Path,
    min_mean_asv_relative_abundance: float = 0.0,
) -> PreparedData:
    meta_path = data_dir / "phyloseq_alphaDiv_metadata_SAFE.csv"
    otu_path = data_dir / "phyloseq_alphaDiv_otu_table.csv"
    tax_path = data_dir / "phyloseq_alphaDiv_taxonomy.csv"
    meta = pd.read_csv(meta_path)
    otu = pd.read_csv(otu_path)
    tax = pd.read_csv(tax_path)

    if otu.columns[0] != "sample_id":
        otu = otu.rename(columns={otu.columns[0]: "sample_id"})
    if "ASV" not in tax.columns:
        tax = tax.rename(columns={tax.columns[0]: "ASV"})

    # Primary published-data QC rule.
    qc = meta.copy()
    qc = qc[qc["experiment"].astype(str).str.lower().eq("main")]
    qc = qc[_bool_false(qc["drop_16s"])]
    qc = qc.drop_duplicates("sample_id")

    # Alignment checks are strict because silent sample loss could bias pairing.
    if qc["sample_id"].duplicated().any():
        raise ValueError("QC metadata contains duplicated sample_id values.")
    otu = otu[otu["sample_id"].isin(qc["sample_id"])].copy()
    missing_otu = sorted(set(qc["sample_id"]) - set(otu["sample_id"]))
    if missing_otu:
        raise ValueError(f"QC samples missing from OTU table: {missing_otu[:10]}")
    otu = otu.set_index("sample_id").loc[qc["sample_id"]]

    asv_cols = [c for c in otu.columns if c in set(tax["ASV"])]
    if not asv_cols:
        raise ValueError("No OTU columns align to taxonomy ASV identifiers.")
    counts = otu[asv_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (counts < 0).any().any():
        raise ValueError("Negative ASV counts detected.")

    rel_asv = normalize_rows(counts)
    mean_rel = rel_asv.mean(axis=0)
    keep = mean_rel >= float(min_mean_asv_relative_abundance)
    kept_asvs = list(mean_rel.index[keep])
    if len(kept_asvs) == 0:
        raise ValueError("ASV filter removed every feature.")
    counts = counts[kept_asvs]

    tax_idx = tax.set_index("ASV").reindex(kept_asvs)
    mapping_rows = []
    for asv, row in tax_idx.iterrows():
        guild, score, rationale = map_taxon_to_curated_guild(row)
        mapping_rows.append({
            "ASV": asv,
            "curated_guild": guild,
            "asv_benefit_score": score,
            "mapping_rationale": rationale,
            **{c: row.get(c, "") for c in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"] if c in tax.columns},
        })
    tax_map = pd.DataFrame(mapping_rows)
    asv_to_guild = tax_map.set_index("ASV")["curated_guild"]

    guild_counts = pd.DataFrame(0.0, index=counts.index, columns=CURATED_GUILD_ORDER)
    for guild in CURATED_GUILD_ORDER:
        cols = asv_to_guild[asv_to_guild.eq(guild)].index.intersection(counts.columns)
        if len(cols):
            guild_counts[guild] = counts[list(cols)].sum(axis=1)
    guild_rel = normalize_rows(guild_counts)

    # Scientific/technical invariants.
    if not np.allclose(guild_rel.sum(axis=1).to_numpy(), 1.0, atol=1e-10):
        raise AssertionError("Guild relative abundances do not close to one.")
    if (guild_rel.to_numpy() < -1e-12).any():
        raise AssertionError("Negative guild relative abundance detected.")

    guild_counts.index.name = "sample_id"
    guild_rel.index.name = "sample_id"
    gm = qc.set_index("sample_id").join(guild_rel, how="inner").reset_index()
    gm["Type"] = gm["Type"].astype(str).str.replace("Device", "Capsule", regex=False)
    gm["event_id"] = gm["Subject"].astype(str) + "|" + gm["Recover_date_time"].astype(str)

    return PreparedData(
        metadata_qc=qc.reset_index(drop=True),
        taxonomy_mapping=tax_map,
        sample_guild_counts_k11=guild_counts,
        sample_guild_rel_k11=guild_rel,
        sample_metadata_guild_k11=gm,
    )


def aggregate_resolution(df: pd.DataFrame, resolution: str) -> Tuple[pd.DataFrame, List[str]]:
    groups = K_GROUPS[resolution]
    out = df.copy()
    cols = list(groups)
    for new, members in groups.items():
        out[new] = df[members].sum(axis=1)
    arr = out[cols].to_numpy(float)
    sums = arr.sum(axis=1, keepdims=True)
    arr = np.divide(arr, sums, out=np.full_like(arr, 1.0 / len(cols)), where=sums > 0)
    out[cols] = arr
    return out, cols


def aggregate_counts_resolution(counts_k11: pd.DataFrame, resolution: str) -> pd.DataFrame:
    groups = K_GROUPS[resolution]
    out = pd.DataFrame(index=counts_k11.index)
    for new, members in groups.items():
        out[new] = counts_k11[members].sum(axis=1)
    return out


def build_event_cohort(
    guild_meta: pd.DataFrame,
    guild_cols: List[str],
    require_types: List[str],
) -> pd.DataFrame:
    rows = []
    for event_id, grp in guild_meta.groupby("event_id", sort=True):
        available = set(grp["Type"].astype(str))
        if not set(require_types).issubset(available):
            continue
        subject = grp["Subject"].dropna()
        if subject.empty:
            continue
        rec = {
            "event_id": event_id,
            "Subject": subject.iloc[0],
            "Recover_date_time": grp["Recover_date_time"].dropna().iloc[0],
            "n_samples_event": int(len(grp)),
            "n_capsule_sets_event": int(grp.loc[grp["Type"].isin(CAPS), "Set"].astype(str).nunique()),
        }
        for tp in require_types:
            sub = grp[grp["Type"].eq(tp)]
            vec = sub[guild_cols].mean(axis=0).to_numpy(float)
            total = vec.sum()
            rec[tp] = vec / total if total > 0 else np.ones(len(guild_cols)) / len(guild_cols)
            rec[f"n_{tp.replace(' ', '_').lower()}_samples"] = int(len(sub))
            rec[f"sets_{tp.replace(' ', '_').lower()}"] = ",".join(sorted(sub["Set"].dropna().astype(str).unique()))
        rows.append(rec)
    events = pd.DataFrame(rows).sort_values(["Subject", "Recover_date_time", "event_id"]).reset_index(drop=True)
    if not events.empty and events["event_id"].duplicated().any():
        raise AssertionError("Event table contains duplicated event IDs.")
    return events


def subject_mean_cohort(events: pd.DataFrame, profile_cols: List[str]) -> pd.DataFrame:
    rows = []
    for subject, grp in events.groupby("Subject", sort=True):
        rec = {"Subject": subject, "event_id": f"subject_mean_{subject}", "n_events_subject": len(grp)}
        for c in profile_cols:
            vec = np.vstack(grp[c].to_numpy()).mean(axis=0)
            rec[c] = vec / vec.sum()
        rows.append(rec)
    return pd.DataFrame(rows)


def save_prepared_data(prepared: PreparedData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.metadata_qc.to_csv(output_dir / "qc_metadata.csv", index=False)
    prepared.taxonomy_mapping.to_csv(output_dir / "asv_to_curated_guild.csv", index=False)
    prepared.sample_guild_counts_k11.to_csv(output_dir / "sample_guild_counts_K11.csv")
    prepared.sample_guild_rel_k11.to_csv(output_dir / "sample_guild_relative_abundance_K11.csv")
    pd.DataFrame(CURATED_GUILD_RUBRIC).to_csv(output_dir / "curated_guild_rubric_K11.csv", index=False)
    with (output_dir / "resolution_groupings.json").open("w") as f:
        json.dump(K_GROUPS, f, indent=2)
