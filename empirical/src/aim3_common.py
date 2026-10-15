#!/usr/bin/env python3
"""
Common utilities for Aim 3 stool-blurring / feasible-set analyses.

The functions are intentionally self-contained and deterministic so that all
scripts can be run from uploaded CSV/XLSX files without R/phyloseq.
"""
from __future__ import annotations

from pathlib import Path
from itertools import combinations_with_replacement
import json
import math
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import jensenshannon

# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------

FILENAME_PATTERNS = {
    "metadata_alpha": ["phyloseq_alphaDiv_metadata_SAFE*.csv"],
    "otu_alpha": ["phyloseq_alphaDiv_otu_table*.csv"],
    "tax_alpha": ["phyloseq_alphaDiv_taxonomy*.csv"],
    "sample_data": ["sample_data*.csv"],
    "canberra": ["canberra_pairwise_16s_distance_matrix*.csv"],
    "bile_salts": ["bile_salts_wAAconjugates*.csv"],
    "data_dictionary": ["0-data-dictionary-samples*.xlsx"],
    "matlab_metadata": ["sample_metadata_matlab_export*.xlsx"],
    "food_questionnaire": ["FoodQuestionnaire*.xlsx"],
    "aa_conjugated_ba": ["AA_conjugated_BAs_15S*.xlsx"],
}


def find_file(data_dir: str | Path, key: str) -> Path:
    data_dir = Path(data_dir)
    for pat in FILENAME_PATTERNS[key]:
        matches = sorted(data_dir.glob(pat))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {key} using patterns {FILENAME_PATTERNS[key]} in {data_dir}")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: str | Path):
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, default=str))

# -----------------------------------------------------------------------------
# Curated guild definitions
# -----------------------------------------------------------------------------

CURATED_GUILD_RUBRIC: List[Dict[str, object]] = [
    {
        "curated_guild": "G01_sulfidogenic_bile_proteolytic",
        "benefit_score": 0.12,
        "interpretation": "Sulfidogenic, bile-associated, proteolytic, or pathobiont-enriched taxa.",
        "examples": "Desulfovibrionaceae, Bilophila, Desulfovibrio, Fusobacterium, Campylobacter",
        "claim_level": "Taxonomic-functional proxy; not enzyme-level annotation.",
    },
    {
        "curated_guild": "G02_oxygen_tolerant_proximal_oral",
        "benefit_score": 0.22,
        "interpretation": "Facultative, oxygen-tolerant, oral/proximal taxa often underrepresented in distal stool.",
        "examples": "Enterobacteriaceae, Streptococcaceae, Enterococcaceae, Lactobacillaceae",
        "claim_level": "Proximal/oral signal proxy, not necessarily harmful.",
    },
    {
        "curated_guild": "G03_bacteroidales_generalist",
        "benefit_score": 0.45,
        "interpretation": "Bacteroidales generalists with broad polysaccharide/protein versatility.",
        "examples": "Bacteroides, Parabacteroides, Alistipes, Odoribacter",
        "claim_level": "Broad taxonomic-functional proxy.",
    },
    {
        "curated_guild": "G04_prevotella_carbohydrate_propionate",
        "benefit_score": 0.55,
        "interpretation": "Prevotella/Prevotellaceae carbohydrate-associated propionate proxy.",
        "examples": "Prevotella, Alloprevotella, Paraprevotella",
        "claim_level": "Diet-associated taxonomic proxy.",
    },
    {
        "curated_guild": "G05_lactate_utilizers_propionate",
        "benefit_score": 0.50,
        "interpretation": "Lactate/succinate-utilizing and propionate-associated anaerobes.",
        "examples": "Veillonella, Dialister, Megasphaera, Phascolarctobacterium",
        "claim_level": "Taxonomic-functional proxy.",
    },
    {
        "curated_guild": "G06_bifidobacteria_lactate_acetate",
        "benefit_score": 0.68,
        "interpretation": "Bifidobacteria acetate/lactate producers and cross-feeding partners.",
        "examples": "Bifidobacterium",
        "claim_level": "Genus-level proxy.",
    },
    {
        "curated_guild": "G07_ruminococcaceae_fiber_butyrate",
        "benefit_score": 0.70,
        "interpretation": "Ruminococcaceae/Faecalibacterium-like fiber and butyrate-associated taxa.",
        "examples": "Faecalibacterium, Subdoligranulum, Ruminococcaceae UCG groups",
        "claim_level": "Family/genus-level proxy.",
    },
    {
        "curated_guild": "G08_mucin_glycan_barrier",
        "benefit_score": 0.72,
        "interpretation": "Mucin-glycan and mucus/barrier-associated taxa.",
        "examples": "Akkermansia, Ruminococcus gnavus/torques group",
        "claim_level": "Context-dependent taxonomic proxy.",
    },
    {
        "curated_guild": "G09_lachnospiraceae_butyrate_crossfeeders",
        "benefit_score": 0.65,
        "interpretation": "Lachnospiraceae-type butyrate/cross-feeding anaerobes.",
        "examples": "Roseburia, Eubacterium rectale group, Anaerostipes, Coprococcus",
        "claim_level": "Family/genus-level proxy.",
    },
    {
        "curated_guild": "G10_other_anaerobic_fermenters_uncertain",
        "benefit_score": 0.42,
        "interpretation": "Other anaerobic fermenters not specific enough for higher-confidence assignment.",
        "examples": "Erysipelotrichaceae, Christensenellaceae, unresolved Clostridia, methanogens",
        "claim_level": "Uncertain/heterogeneous.",
    },
    {
        "curated_guild": "G11_unassigned_low_information",
        "benefit_score": 0.35,
        "interpretation": "Unclassified or low-information taxa retained for compositional closure.",
        "examples": "Unresolved ASVs",
        "claim_level": "Not mechanistically interpreted.",
    },
]

CURATED_GUILD_ORDER = [d["curated_guild"] for d in CURATED_GUILD_RUBRIC]
BENEFIT_SCORE = {d["curated_guild"]: float(d["benefit_score"]) for d in CURATED_GUILD_RUBRIC}
BENEFIT_ORDERED_GUILDS = sorted(CURATED_GUILD_ORDER, key=lambda g: BENEFIT_SCORE[g])
BENEFIT_VECTOR_ORDERED = np.array([BENEFIT_SCORE[g] for g in BENEFIT_ORDERED_GUILDS], dtype=float)


def normalize_1d(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    x = np.asarray(x, dtype=float) + eps
    s = float(x.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x, dtype=float) / len(x)
    return x / s


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(sums, axis=0).fillna(0.0)


def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_1d(np.asarray(a, dtype=float))
    b = normalize_1d(np.asarray(b, dtype=float))
    return 0.5 * float(np.abs(a - b).sum())


def js_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_1d(np.asarray(a, dtype=float), eps=1e-12)
    b = normalize_1d(np.asarray(b, dtype=float), eps=1e-12)
    return float(jensenshannon(a, b, base=2.0))


def clr(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = normalize_1d(np.asarray(x, dtype=float), eps=eps)
    lx = np.log(x)
    return lx - lx.mean()


def inv_clr(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z))
    return e / e.sum()


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def clean_tax(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "na", "none", "unclassified", "uncultured", "unknown"}:
        return ""
    return s


def map_taxon_to_curated_guild(row: pd.Series) -> Tuple[str, float, str]:
    ph, cl, order, fam, genus, species = [
        clean_tax(row.get(x)) for x in ["Phylum", "Class", "Order", "Family", "Genus", "Species"]
    ]
    fam_l, genus_l, species_l = fam.lower(), genus.lower(), species.lower()

    if genus_l.startswith("akkermansia") or fam == "Akkermansiaceae":
        return "G08_mucin_glycan_barrier", 0.72, "Akkermansia/Akkermansiaceae mucin-glycan specialist"
    if "gnavus" in genus_l or "gnavus" in species_l or "torques" in genus_l or "torques" in species_l:
        return "G08_mucin_glycan_barrier", 0.55, "Ruminococcus gnavus/torques group; mucin-glycan, context-dependent"

    if fam == "Desulfovibrionaceae" or genus in {"Bilophila", "Desulfovibrio"}:
        return "G01_sulfidogenic_bile_proteolytic", 0.12, "sulfate/sulfite-reducing or bile-associated anaerobe"
    if fam in {"Fusobacteriaceae", "Campylobacteraceae", "Helicobacteraceae"} or genus in {"Fusobacterium", "Campylobacter", "Helicobacter"}:
        return "G01_sulfidogenic_bile_proteolytic", 0.12, "oral/pathobiont or protein-fermenting taxon"
    if fam == "Eggerthellaceae" or genus in {"Eggerthella", "Slackia", "Gordonibacter"}:
        return "G01_sulfidogenic_bile_proteolytic", 0.18, "bile/phenolic metabolism-enriched anaerobe"

    if fam in {
        "Enterobacteriaceae", "Streptococcaceae", "Enterococcaceae", "Lactobacillaceae",
        "Pasteurellaceae", "Neisseriaceae", "Micrococcaceae", "Staphylococcaceae",
        "Corynebacteriaceae", "Actinomycetaceae", "Leptotrichiaceae", "Carnobacteriaceae",
        "Leuconostocaceae",
    } or order in {"Enterobacteriales", "Lactobacillales", "Pasteurellales", "Neisseriales"}:
        return "G02_oxygen_tolerant_proximal_oral", 0.22, "facultative/oxygen-tolerant or oral/proximal taxon"

    if fam == "Bifidobacteriaceae" or genus == "Bifidobacterium":
        return "G06_bifidobacteria_lactate_acetate", 0.68, "Bifidobacteria acetate/lactate producer and cross-feeding partner"
    if fam in {"Veillonellaceae", "Acidaminococcaceae"} or genus in {"Veillonella", "Dialister", "Megasphaera", "Megamonas", "Phascolarctobacterium", "Selenomonas"}:
        return "G05_lactate_utilizers_propionate", 0.50, "lactate/succinate-utilizing or propionate-associated anaerobe"
    if genus_l.startswith("prevotella") or fam == "Prevotellaceae" or genus in {"Alloprevotella", "Paraprevotella"}:
        return "G04_prevotella_carbohydrate_propionate", 0.55, "Prevotella/Prevotellaceae carbohydrate-associated propionate proxy"
    if genus in {"Bacteroides", "Parabacteroides", "Alistipes", "Odoribacter", "Barnesiella"} or fam in {"Bacteroidaceae", "Tannerellaceae", "Rikenellaceae", "Barnesiellaceae", "Porphyromonadaceae"}:
        return "G03_bacteroidales_generalist", 0.45, "Bacteroidales generalist"

    butyrate_genera = {
        "Faecalibacterium", "Roseburia", "Eubacterium", "Agathobacter", "Anaerostipes",
        "Butyricicoccus", "Coprococcus", "Coprococcus_1", "Coprococcus_2", "Coprococcus_3",
        "Fusicatenibacter", "Subdoligranulum", "Butyrivibrio", "Pseudobutyrivibrio",
        "Anaerobutyricum", "Eubacterium_eligens_group", "Eubacterium_rectale_group",
        "Lachnospira", "Ruminococcus_2", "Gemmiger",
    }
    if genus in butyrate_genera or genus.startswith("Ruminococcaceae_UCG"):
        if fam == "Ruminococcaceae" or "ruminococcaceae" in genus_l or genus in {"Faecalibacterium", "Subdoligranulum", "Ruminococcus_2", "Gemmiger"}:
            return "G07_ruminococcaceae_fiber_butyrate", 0.82, "Ruminococcaceae/Faecalibacterium/Subdoligranulum-type fiber/butyrate producer"
        return "G09_lachnospiraceae_butyrate_crossfeeders", 0.78, "Lachnospiraceae-type butyrate/cross-feeding taxon"
    if fam == "Ruminococcaceae":
        return "G07_ruminococcaceae_fiber_butyrate", 0.70, "Ruminococcaceae unresolved; fiber-associated anaerobic guild with uncertainty"
    if fam == "Lachnospiraceae":
        return "G09_lachnospiraceae_butyrate_crossfeeders", 0.65, "Lachnospiraceae unresolved; butyrate/cross-feeding guild with uncertainty"

    if fam in {"Erysipelotrichaceae", "Christensenellaceae", "Clostridiaceae_1", "Family_XI", "Family_XIII", "Peptostreptococcaceae", "Peptococcaceae", "Oscillospiraceae"} or cl in {"Clostridia", "Erysipelotrichia", "Negativicutes"}:
        return "G10_other_anaerobic_fermenters_uncertain", 0.42, "other anaerobic fermenter; unresolved function from 16S"
    if ph == "Euryarchaeota" or fam == "Methanobacteriaceae" or genus == "Methanobrevibacter":
        return "G10_other_anaerobic_fermenters_uncertain", 0.38, "methanogen/archaeon; fermentation ecology but not SCFA-benefit guild"

    return "G11_unassigned_low_information", 0.35, "taxonomy insufficient; retained for compositional closure"

# -----------------------------------------------------------------------------
# Shalon 16S loading, QC, guild aggregation, event tables
# -----------------------------------------------------------------------------


def load_shalon_16s(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load exported phyloseq CSVs and apply the core QC mask.

    Returns taxonomy, raw OTU table aligned to QC samples, and QC metadata.
    """
    data_dir = Path(data_dir)
    meta = pd.read_csv(find_file(data_dir, "metadata_alpha"))
    otu = pd.read_csv(find_file(data_dir, "otu_alpha"))
    tax = pd.read_csv(find_file(data_dir, "tax_alpha"))

    # OTU table convention: first column is sample_id; rows are samples.
    if otu.columns[0] != "sample_id":
        otu = otu.rename(columns={otu.columns[0]: "sample_id"})
    if tax.columns[0] not in {"ASV", "asv", "taxon", "Unnamed: 0"}:
        pass
    if "ASV" not in tax.columns:
        tax = tax.rename(columns={tax.columns[0]: "ASV"})

    qc = meta.copy()
    if "experiment" in qc.columns:
        qc = qc[qc["experiment"].astype(str).str.lower().eq("main")]
    if "drop_16s" in qc.columns:
        # Handles bools and strings.
        qc = qc[~qc["drop_16s"].astype(str).str.lower().isin(["true", "1", "yes"])]
    if "sample_id" not in qc.columns:
        raise ValueError("Metadata is missing sample_id.")
    qc = qc.drop_duplicates("sample_id")

    otu = otu[otu["sample_id"].isin(qc["sample_id"])].copy()
    qc = qc[qc["sample_id"].isin(otu["sample_id"])].copy()
    otu = otu.set_index("sample_id").loc[qc["sample_id"]].reset_index()

    return tax, otu, qc


def build_curated_guild_abundance(tax: pd.DataFrame, otu: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Map ASVs to curated guilds and aggregate sample-level guild abundances."""
    tax = tax.copy()
    if "ASV" not in tax.columns:
        tax = tax.rename(columns={tax.columns[0]: "ASV"})
    mapping_rows = []
    for _, row in tax.iterrows():
        guild, score, rationale = map_taxon_to_curated_guild(row)
        mapping_rows.append({
            "ASV": row["ASV"],
            "curated_guild": guild,
            "asv_benefit_score": score,
            "mapping_rationale": rationale,
            **{col: row.get(col, "") for col in ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"] if col in tax.columns},
        })
    tax_map = pd.DataFrame(mapping_rows)

    if otu.columns[0] != "sample_id":
        otu = otu.rename(columns={otu.columns[0]: "sample_id"})
    asv_cols = [c for c in otu.columns if c in set(tax_map["ASV"])]
    otu_numeric = otu[["sample_id"] + asv_cols].copy()
    otu_numeric[asv_cols] = otu_numeric[asv_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    guild_mat = pd.DataFrame(0.0, index=otu_numeric["sample_id"], columns=CURATED_GUILD_ORDER)
    map_series = tax_map.set_index("ASV")["curated_guild"]
    for guild, cols in map_series[asv_cols].groupby(map_series[asv_cols]).groups.items():
        cols = list(cols)
        guild_mat[guild] = otu_numeric[cols].sum(axis=1).to_numpy()
    guild_rel = normalize_rows(guild_mat)
    guild_rel.index.name = "sample_id"

    guild_meta = meta.set_index("sample_id").join(guild_rel, how="inner").reset_index()
    if "Type" in guild_meta.columns:
        # Normalize Type labels used in scripts.
        guild_meta["Type"] = guild_meta["Type"].astype(str).str.replace("Device", "Capsule", regex=False)
    return tax_map, guild_rel, guild_meta


def make_event_table(guild_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Make one-row-per-subject-event table with stool and capsule vectors.

    Returns
    -------
    all_events : events with stool and any capsule(s)
    complete_events : subset with stool and Capsule 1-4
    """
    gm = guild_meta.copy()
    required = ["Subject", "Recover_date_time", "Type"]
    for col in required:
        if col not in gm.columns:
            raise ValueError(f"Required metadata column {col} not found.")
    gm["event_id"] = gm["Subject"].astype(str) + "|" + gm["Recover_date_time"].astype(str)
    caps = ["Capsule 1", "Capsule 2", "Capsule 3", "Capsule 4"]
    rows = []
    for eid, grp in gm.groupby("event_id"):
        types = set(grp["Type"].astype(str))
        if "Stool" not in types:
            continue
        if not any(c in types for c in caps):
            continue
        rec = {
            "event_id": eid,
            "Subject": grp["Subject"].dropna().iloc[0],
            "Recover_date_time": grp["Recover_date_time"].dropna().iloc[0],
            "n_samples_event": len(grp),
        }
        stool = grp[grp["Type"].eq("Stool")][CURATED_GUILD_ORDER].mean(axis=0).to_numpy(float)
        rec["Stool"] = normalize_1d(stool)
        for cap in caps:
            sub = grp[grp["Type"].eq(cap)]
            if len(sub):
                vec = sub[CURATED_GUILD_ORDER].mean(axis=0).to_numpy(float)
                rec[cap] = normalize_1d(vec)
            else:
                rec[cap] = None
        rows.append(rec)
    events = pd.DataFrame(rows)
    if len(events) == 0:
        return events, events.copy()
    complete_mask = events[caps].map(lambda x: x is not None).all(axis=1)
    complete = events[complete_mask].copy().reset_index(drop=True)
    return events.reset_index(drop=True), complete


def vector_columns_to_wide(df: pd.DataFrame, vector_col: str, prefix: str, guilds: List[str] = None) -> pd.DataFrame:
    guilds = guilds or CURATED_GUILD_ORDER
    arr = np.vstack(df[vector_col].to_numpy()) if len(df) else np.empty((0, len(guilds)))
    out = pd.DataFrame(arr, columns=[f"{prefix}_{g}" for g in guilds], index=df.index)
    return pd.concat([df.drop(columns=[vector_col]), out], axis=1)


def reorder_to_benefit(v: np.ndarray | pd.Series) -> np.ndarray:
    if isinstance(v, pd.Series):
        return v[BENEFIT_ORDERED_GUILDS].to_numpy(float)
    # Assumes input in CURATED_GUILD_ORDER.
    idx = [CURATED_GUILD_ORDER.index(g) for g in BENEFIT_ORDERED_GUILDS]
    return np.asarray(v, dtype=float)[idx]

# -----------------------------------------------------------------------------
# K-guild feasible set functions
# -----------------------------------------------------------------------------


def monotone_strategies(K: int) -> np.ndarray:
    """All monotone non-decreasing strategies sigma: observed guild -> intervention tier."""
    return np.array(list(combinations_with_replacement(range(K), K)), dtype=np.int16)


def transition_identity(K: int) -> np.ndarray:
    return np.eye(K, dtype=float)


def transition_one_beta(beta: float, ref: np.ndarray) -> np.ndarray:
    ref = normalize_1d(ref)
    K = len(ref)
    beta = float(np.clip(beta, 0.0, 1.0))
    return (1.0 - beta) * np.eye(K) + beta * np.tile(ref, (K, 1))


def transition_guild_specific(beta_g: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = normalize_1d(ref)
    beta_g = np.asarray(beta_g, dtype=float)
    K = len(ref)
    T = np.zeros((K, K), dtype=float)
    for k in range(K):
        b = float(np.clip(beta_g[k], 0.0, 1.0))
        T[k, :] = b * ref
        T[k, k] += (1.0 - b)
    return T / T.sum(axis=1, keepdims=True)


def feasible_points(phi: np.ndarray, h: np.ndarray, T: np.ndarray, strategies: np.ndarray | None = None) -> pd.DataFrame:
    """Compute all strategy outcomes under a transition operator.

    Columns are ecosystem stability UE, host benefit UH, and strategy summaries.
    """
    phi = normalize_1d(phi)
    h = np.asarray(h, dtype=float)
    K = len(h)
    T = np.asarray(T, dtype=float)
    if strategies is None:
        strategies = monotone_strategies(K)
    # contribution matrices [observed o, intervention j]
    UH_contrib = np.zeros((K, K), dtype=float)
    UE_contrib = np.zeros((K, K), dtype=float)
    for o in range(K):
        for j in range(K):
            for k in range(K):
                p = phi[k] * T[k, o]
                UH_contrib[o, j] += p * (h[j] if k >= j else 0.0)
                UE_contrib[o, j] += p * max(0.0, h[k] - h[j])
    n = strategies.shape[0]
    UH = np.zeros(n, dtype=float)
    UE = np.zeros(n, dtype=float)
    for o in range(K):
        js = strategies[:, o]
        UH += UH_contrib[o, js]
        UE += UE_contrib[o, js]
    mean_tier = strategies.mean(axis=1) / max(1, K - 1)
    mean_h = h[strategies].mean(axis=1)
    out = pd.DataFrame({"UE": UE, "UH": UH, "mean_intervention_tier": mean_tier, "mean_intervention_score": mean_h})
    out["strategy_id"] = np.arange(n)
    return out


def convex_hull_area(points_xy: np.ndarray) -> float:
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) < 3:
        return 0.0
    # Drop duplicates to stabilize hull.
    pts = np.unique(np.round(pts, 4), axis=0)
    if len(pts) < 3:
        return 0.0
    try:
        hull = ConvexHull(pts)
        return float(hull.volume)  # area in 2D
    except Exception:
        return 0.0


def hull_vertices(points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    pts = np.unique(np.round(pts, 4), axis=0)
    if len(pts) < 3:
        return pts
    try:
        hull = ConvexHull(pts)
        verts = pts[hull.vertices]
        ctr = verts.mean(axis=0)
        ang = np.arctan2(verts[:, 1] - ctr[1], verts[:, 0] - ctr[0])
        return verts[np.argsort(ang)]
    except Exception:
        return pts

# -----------------------------------------------------------------------------
# Compatibility aliases for sensitivity scripts 07--09
# -----------------------------------------------------------------------------
DEFAULT_GUILD_BENEFIT_SCORE = BENEFIT_SCORE

def order_guilds_for_intervention_axis(guilds: list[str], scores: dict[str, float] | None = None):
    if scores is None:
        scores = DEFAULT_GUILD_BENEFIT_SCORE
    h = np.asarray([scores[g] for g in guilds], dtype=float)
    order = np.argsort(h)
    return [guilds[i] for i in order], h[order]

def reorder_vector(vec: np.ndarray, from_guilds: list[str], to_guilds: list[str]) -> np.ndarray:
    s = pd.Series(np.asarray(vec, dtype=float), index=from_guilds)
    return s.reindex(to_guilds).fillna(0.0).to_numpy(dtype=float)

def _score_tables_scalar_beta(phi: np.ndarray, beta: float, h_vec: np.ndarray):
    K = len(h_vec)
    phi = normalize_1d(phi)
    p_obs_given_true = np.full((K, K), beta / K, dtype=float)
    p_obs_given_true[np.arange(K), np.arange(K)] += (1.0 - beta)
    UH = np.zeros((K, K), dtype=float)
    UE = np.zeros((K, K), dtype=float)
    for k in range(K):
        for j in range(K):
            UH[k, j] = h_vec[j] if k >= j else 0.0
            UE[k, j] = max(0.0, h_vec[k] - h_vec[j])
    weights = phi[:, None] * p_obs_given_true
    score_UH = weights.T @ UH
    score_UE = weights.T @ UE
    return score_UH, score_UE

def outcomes_for_strategies(phi: np.ndarray, beta: float, h_vec: np.ndarray, strategies: np.ndarray | None = None) -> np.ndarray:
    K = len(h_vec)
    if strategies is None:
        strategies = monotone_strategies(K)
    score_UH, score_UE = _score_tables_scalar_beta(phi, beta, h_vec)
    rows = np.arange(K)
    UH_vals = score_UH[rows[None, :], strategies].sum(axis=1)
    UE_vals = score_UE[rows[None, :], strategies].sum(axis=1)
    return np.column_stack([UE_vals, UH_vals])

def polygon_area(points: np.ndarray) -> float:
    return convex_hull_area(points)

# Replace uncached strategy enumeration with a cached version for K=11 reproducibility/speed.
from functools import lru_cache as _lru_cache
@_lru_cache(maxsize=None)
def monotone_strategies(K: int) -> np.ndarray:  # noqa: F811
    return np.array(list(combinations_with_replacement(range(K), K)), dtype=np.int16)
