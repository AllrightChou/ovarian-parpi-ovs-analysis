# -*- coding: utf-8 -*-

# =========================================================
# Main causal inference analysis for OVS-associated
# treatment heterogeneity in PARP inhibitor maintenance
# therapy among ovarian cancer patients.
#
# This script implements the primary analytical framework
# described in the Methods section of the manuscript,
# including:
#
#   1. Multiple Imputation (MI) for incomplete baseline
#      covariates using IterativeImputer.BRCA mutation status 
#       and HRD status are NEVER imputed.
#
#   2. Stabilized Inverse Probability of Treatment
#      Weighting (IPTW) using propensity scores
#
#   3. IPTW-weighted Cox proportional hazards models
#      with treatment-by-OVS interaction terms
#
#   4. Rubin's Rules pooling across multiply imputed
#      datasets
#
#   5. G-computation-based counterfactual survival
#      standardization
#
#   6. Covariate balance diagnostics and positivity
#      assessment
#
# ---------------------------------------------------------
# Treatment coding
# ---------------------------------------------------------
# maintenance_drug:
#   0 = Olaparib
#   1 = Niraparib
#
# Hazard ratios are ultimately reported as:
#
#   Olaparib versus Niraparib
#
# Therefore, coefficients estimated from Cox models are
# directionally transformed when necessary to maintain
# consistent clinical interpretation throughout the
# manuscript.
#
# ---------------------------------------------------------
# Organ Vulnerability Score (OVS)
# ---------------------------------------------------------
# OVS is analyzed using the prespecified binary grouping:
#
#   Low vulnerability  = OVS 0–1
#   High vulnerability = OVS 2–3
#
# The primary estimand is the treatment-by-vulnerability
# interaction effect on progression-free survival (PFS).
#
# ---------------------------------------------------------
# Missing data strategy
# ---------------------------------------------------------
# Only baseline clinical covariates with incomplete data
# are multiply imputed:
#
#   - age
#   - stage
#   - treatment_line
#
# BRCA and HRD variables are NOT multiply imputed.
# Missing molecular data are handled using an explicit
# "Unknown" category strategy to preserve real-world
# applicability and avoid unverifiable assumptions
# regarding missing molecular status.
#
# ---------------------------------------------------------
# Ordinal covariate handling
# ---------------------------------------------------------
# FIGO stage and treatment line are modeled as ordinal
# numeric covariates rather than fully one-hot encoded
# categorical variables within propensity score and Cox
# models.
#
# This modeling choice was prespecified to preserve model
# parsimony and estimation stability given the moderate
# sample size and relatively sparse subgroup structure.
#
# However, for covariate balance diagnostics, these
# variables are additionally expanded into one-hot encoded
# dummy indicators to allow category-specific standardized
# mean difference (SMD) assessment.
#
# ---------------------------------------------------------
# Causal interpretation
# ---------------------------------------------------------
# This is a retrospective observational analysis using
# causal inference techniques intended to reduce measured
# confounding under standard identifiability assumptions,
# including:
#
#   - conditional exchangeability
#   - positivity
#   - consistency
#
# Results should be interpreted as adjusted associative
# estimates rather than definitive causal treatment
# effects.
#
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from lifelines import (
    CoxPHFitter,
    KaplanMeierFitter
)
from lifelines.statistics import logrank_test

from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# =========================================================
# 0. Helper functions
# =========================================================

def format_p_val(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

# Rubin's Rules pooling (coefficient & variance)
def rubin_pool(estimates, variances):
    m = len(estimates)
    q_bar = np.mean(estimates)
    u_bar = np.mean(variances)
    b = np.var(estimates, ddof=1)
    total_var = u_bar + (1 + 1/m) * b
    pooled_se = np.sqrt(total_var)
    return q_bar, pooled_se, total_var

# Pool covariance matrices
def rubin_pool_covariance(beta_list, cov_list):
    m = len(beta_list)
    beta_names = beta_list[0].index
    beta_matrix = np.vstack([b.values for b in beta_list])
    q_bar = np.mean(beta_matrix, axis=0)
    u_bar = sum(cov_list) / m
    centered = beta_matrix - q_bar
    b_mat = np.dot(centered.T, centered) / (m - 1)
    total_cov = u_bar + (1 + 1/m) * b_mat
    pooled_beta = pd.Series(q_bar, index=beta_names)
    pooled_cov = pd.DataFrame(total_cov, index=beta_names, columns=beta_names)
    return pooled_beta, pooled_cov

# Effective Sample Size
def compute_ess(weights):
    return (weights.sum() ** 2) / np.sum(weights ** 2)

# SMD for continuous variables
def compute_smd_continuous(x_treat, x_control):
    mean_t = np.mean(x_treat)
    mean_c = np.mean(x_control)
    sd_t = np.var(x_treat, ddof=1)
    sd_c = np.var(x_control, ddof=1)
    pooled_sd = np.sqrt((sd_t + sd_c) / 2)
    return 0.0 if pooled_sd == 0 else (mean_t - mean_c) / pooled_sd

# SMD for binary variables
def compute_smd_binary(x_treat, x_control):
    p1, p0 = np.mean(x_treat), np.mean(x_control)
    pooled = (p1 * (1 - p1) + p0 * (1 - p0)) / 2
    return 0.0 if pooled == 0 else (p1 - p0) / np.sqrt(pooled)

# =========================================================
# 1. Load data
# =========================================================
file_path = "parp_stats.xlsx"
df = pd.read_excel(file_path)

print("=" * 80)
print("Original dataset")
print("=" * 80)
print(f"Raw sample size: {len(df)}")

# =========================================================
# 2. Initial data cleaning
# =========================================================
required_cols = [
    "maintenance_drug",
    "organ_vulnerability_score",
    "pfs_month",
    "pfs_event"
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[df["maintenance_drug"].isin([0, 1])].copy()
df = df[df["pfs_month"].notna() & df["pfs_event"].notna()].copy()
df = df[df["organ_vulnerability_score"].notna()].copy()

# =========================================================
# 3. OVS binary classification
# =========================================================

df["organ_vulnerability_score"] = (
    pd.to_numeric(df["organ_vulnerability_score"], errors="coerce")
    .round()
    .astype("Int64") 
)

ovs_to_group = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
}

df["high_vulnerability"] = df["organ_vulnerability_score"].map(ovs_to_group)
df = df[df["high_vulnerability"].notna()].copy()
df["high_vulnerability"] = df["high_vulnerability"].astype(int)
df["vuln_group"] = df["high_vulnerability"].map({0: "Low", 1: "High"})

# Display OVS distribution for reproducibility checks
# and subgroup transparency.
print("\n OVS ")
ovs_counts = df["organ_vulnerability_score"].value_counts().sort_index()
for score in ovs_counts.index:
    n = ovs_counts.loc[score]
    pct = n / len(df) * 100
    print(f"OVS {int(score)}: {n} ({pct:.1f}%)")
# =========================================================
# 4. BRCA / HRD missing as Unknown (-1)
# =========================================================
for molecular_col in ["brca_mutation", "hrd_status"]:
    if molecular_col in df.columns:
        df[molecular_col] = df[molecular_col].fillna(-1).astype(int)

print("\nBRCA / HRD missing values encoded as Unknown (-1)")

# =========================================================
# 5. Multiple Imputation (MI)
# =========================================================
impute_vars = ["age", "stage", "treatment_line"]
m = 5
imputed_datasets = []

print("\n" + "=" * 80)
print("Multiple Imputation")
print("=" * 80)

for i in range(m):
    print(f"\nGenerating imputed dataset {i+1}/{m}")
    imp_df = df.copy()
    imputer = IterativeImputer(
        random_state=42 + i,
        max_iter=50,
        sample_posterior=True
    )
    imp_df[impute_vars] = imputer.fit_transform(imp_df[impute_vars])
    # Restore integer nature for ordinal variables
    imp_df["stage"] = np.round(imp_df["stage"]).astype(int)
    imp_df["treatment_line"] = np.round(imp_df["treatment_line"]).astype(int)
    imputed_datasets.append(imp_df)

print("\nMI completed")

# =========================================================
# 6. Main analysis loop
# =========================================================
beta_list = []
covariance_list = []
all_balance_tables = []        # Store SMD tables per dataset
ess_list = []                  # ESS values per dataset
models = []                    # Fitted Cox models
model_dfs = []                 # Model dataframes 
last_ps_df = None              # Save PS dataframe of last dataset for overlap plot
last_y_ps = None

for mi_idx, mi_df in enumerate(imputed_datasets):

    print("\n" + "=" * 80)
    print(f"Imputed dataset {mi_idx+1}")
    print("=" * 80)

    # 6.1 Propensity score model
    ps_covariates = ["age", "stage", "treatment_line", "brca_mutation", "hrd_status"]
    ps_df = mi_df[ps_covariates + ["maintenance_drug"]].copy()
    ps_df["stage"] = ps_df["stage"].astype(int)
    ps_df["treatment_line"] = ps_df["treatment_line"].astype(int)

    # One-hot encode molecular variables for propensity
    # score estimation.
    #
    # BRCA/HRD are incorporated as categorical indicators,
    # including explicit "Unknown" categories.
    ps_df = pd.get_dummies(ps_df, columns=["brca_mutation", "hrd_status"], drop_first=True)
    X_ps = ps_df.drop(columns=["maintenance_drug"])
    y_ps = ps_df["maintenance_drug"]

    ps_model = LogisticRegression(max_iter=5000, random_state=42)
    ps_model.fit(X_ps, y_ps)
    ps = ps_model.predict_proba(X_ps)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)
    ps_df["ps"] = ps

    # 6.2 Stabilized IPTW
    treat_prob = y_ps.mean()
    ps_df["iptw"] = np.where(
        y_ps == 1,
        treat_prob / ps_df["ps"],
        (1 - treat_prob) / (1 - ps_df["ps"])
    )
    lower = ps_df["iptw"].quantile(0.01)
    upper = ps_df["iptw"].quantile(0.99)
    ps_df["iptw"] = ps_df["iptw"].clip(lower, upper)
    mi_df["iptw"] = ps_df["iptw"].values

    # Save last PS dataframe for overlap plot
    if mi_idx == (m - 1):
        last_ps_df = ps_df.copy()
        last_y_ps = y_ps.copy()

    # 6.3 Balance diagnostics (SMD) within this dataset
    balance_results = []
    balance_df_raw = mi_df.copy()
    balance_df_weighted = mi_df.copy()

    # Continuous variables
    cont_vars = ["age"]
    for var in cont_vars:
        # Unweighted
        t_raw = balance_df_raw[balance_df_raw["maintenance_drug"] == 1][var]
        c_raw = balance_df_raw[balance_df_raw["maintenance_drug"] == 0][var]
        smd_before = compute_smd_continuous(t_raw, c_raw)
        # Weighted
        wt_t = balance_df_weighted[balance_df_weighted["maintenance_drug"] == 1]
        wt_c = balance_df_weighted[balance_df_weighted["maintenance_drug"] == 0]
        mean_t_w = np.average(wt_t[var], weights=wt_t["iptw"])
        mean_c_w = np.average(wt_c[var], weights=wt_c["iptw"])
        var_t_w = np.average((wt_t[var] - mean_t_w) ** 2, weights=wt_t["iptw"])
        var_c_w = np.average((wt_c[var] - mean_c_w) ** 2, weights=wt_c["iptw"])
        pooled_sd_w = np.sqrt((var_t_w + var_c_w) / 2)
        smd_after = 0.0 if pooled_sd_w == 0 else (mean_t_w - mean_c_w) / pooled_sd_w
        balance_results.append({"Variable": var, "SMD_before": abs(smd_before), "SMD_after": abs(smd_after)})

    # Category-specific balance diagnostics using one-hot
    # encoded indicators.
    #
    # Although stage and treatment_line are modeled as
    # ordinal numeric covariates in the primary models,
    # dummy expansion is additionally used here to assess
    # category-level balance after weighting.
    cat_vars = [v for v in ["brca_mutation", "hrd_status", "stage", "treatment_line"] if v in df.columns]
    for var in cat_vars:
        dummies = pd.get_dummies(mi_df[var], prefix=var, dtype=float)
        for dummy_col in dummies.columns:
            temp = pd.concat([mi_df, dummies[[dummy_col]]], axis=1)
            # Unweighted
            t_raw = temp[temp["maintenance_drug"] == 1][dummy_col]
            c_raw = temp[temp["maintenance_drug"] == 0][dummy_col]
            smd_before = compute_smd_binary(t_raw, c_raw)
            # Weighted
            wt_t = temp[temp["maintenance_drug"] == 1]
            wt_c = temp[temp["maintenance_drug"] == 0]
            p1_w = np.average(wt_t[dummy_col], weights=wt_t["iptw"])
            p0_w = np.average(wt_c[dummy_col], weights=wt_c["iptw"])
            pooled_w = (p1_w * (1 - p1_w) + p0_w * (1 - p0_w)) / 2
            smd_after = 0.0 if pooled_w == 0 else (p1_w - p0_w) / np.sqrt(pooled_w)
            balance_results.append({"Variable": dummy_col, "SMD_before": abs(smd_before), "SMD_after": abs(smd_after)})

    balance_table = pd.DataFrame(balance_results)
    all_balance_tables.append(balance_table)

    # 6.4 Effective Sample Size
    ess_total = compute_ess(mi_df["iptw"].values)
    ess_list.append(ess_total)
    print(f"ESS: {ess_total:.1f}")

    # 6.5 Weighted Cox model
    cox_covariates = ["age", "stage", "treatment_line"]
    model_cols = ["pfs_month", "pfs_event", "maintenance_drug",
                  "high_vulnerability", "iptw"] + cox_covariates
    model_df = mi_df[model_cols].copy()
    model_df["age"] = (model_df["age"] - model_df["age"].mean()) / model_df["age"].std()
    model_df["drug_vuln_inter"] = model_df["maintenance_drug"] * model_df["high_vulnerability"]

    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="pfs_month", event_col="pfs_event",
            weights_col="iptw", robust=True)

    print("Weighted Cox model fitted")
    beta_list.append(cph.params_)
    covariance_list.append(cph.variance_matrix_)
    models.append(cph)
    model_dfs.append(model_df)

# =========================================================
# 7. Rubin pooling
# =========================================================
print("\n" + "=" * 80)
print("Rubin pooling")
print("=" * 80)

pooled_beta, pooled_cov = rubin_pool_covariance(beta_list, covariance_list)
print("\nPooled coefficients:")
print(pooled_beta)

# =========================================================
# 8. Derive pooled HR (Olaparib vs Niraparib)
# =========================================================
beta_drug = pooled_beta["maintenance_drug"]
beta_inter = pooled_beta["drug_vuln_inter"]
var_drug = pooled_cov.loc["maintenance_drug", "maintenance_drug"]
var_inter = pooled_cov.loc["drug_vuln_inter", "drug_vuln_inter"]
covar = pooled_cov.loc["maintenance_drug", "drug_vuln_inter"]

# maintenance_drug:
#   0 = Olaparib
#   1 = Niraparib
#
# lifelines parameterization therefore estimates the
# log-HR for Niraparib relative to Olaparib.
#
# To maintain manuscript-wide interpretability aligned
# with clinical reporting conventions, coefficients are
# directionally transformed to report:
#
#   HR (Olaparib versus Niraparib)
hr_low = np.exp(-beta_drug)
se_low = np.sqrt(var_drug)
ci_low = np.exp(-beta_drug + np.array([-1.96, 1.96]) * se_low)

# High vulnerability
var_high = var_drug + var_inter + 2 * covar
se_high = np.sqrt(var_high)
hr_high = np.exp(-beta_drug - beta_inter)
ci_high = np.exp(-beta_drug - beta_inter + np.array([-1.96, 1.96]) * se_high)

# Interaction
hr_interaction = np.exp(-beta_inter)
se_inter = np.sqrt(var_inter)
ci_interaction = np.exp(-beta_inter + np.array([-1.96, 1.96]) * se_inter)
z_inter = -beta_inter / se_inter
p_interaction = 2 * (1 - stats.norm.cdf(abs(z_inter)))

print("\n" + "=" * 80)
print("Pooled IPTW-weighted Cox results (Olaparib vs Niraparib)")
print("=" * 80)
print(f"Low vulnerability:  HR={hr_low:.3f} (95% CI {ci_low[0]:.3f}-{ci_low[1]:.3f})")
print(f"High vulnerability: HR={hr_high:.3f} (95% CI {ci_high[0]:.3f}-{ci_high[1]:.3f})")
print(f"Interaction HR={hr_interaction:.3f} (95% CI {ci_interaction[0]:.3f}-{ci_interaction[1]:.3f}), P={format_p_val(p_interaction)}")

# =========================================================
# 9. ESS summary across MI datasets
# =========================================================
print("\n" + "=" * 80)
print("ESS across imputed datasets")
print("=" * 80)
ess_arr = np.array(ess_list)
print(f"Mean ESS: {ess_arr.mean():.1f} (range {ess_arr.min():.1f}-{ess_arr.max():.1f})")
print(f"Raw N: {len(df)}")

# =========================================================
# 10. Kaplan-Meier curves (unweighted, for descriptive purpose)
# =========================================================
print("\nGenerating Kaplan-Meier curves")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
kmf = KaplanMeierFitter()
drug_labels = {0: "Olaparib", 1: "Niraparib"}

for i, group in enumerate(["Low", "High"]):
    sub = df[df["vuln_group"] == group]
    d0 = sub[sub["maintenance_drug"] == 0]
    d1 = sub[sub["maintenance_drug"] == 1]
    lr = logrank_test(d0["pfs_month"], d1["pfs_month"],
                      event_observed_A=d0["pfs_event"], event_observed_B=d1["pfs_event"])
    p_val = lr.p_value
    for drug in [0, 1]:
        d = sub[sub["maintenance_drug"] == drug]
        kmf.fit(d["pfs_month"], d["pfs_event"], label=drug_labels[drug])
        kmf.plot(ax=ax[i])
    p_str = "P < 0.001" if p_val < 0.001 else f"P = {p_val:.3f}"
    ax[i].set_title(f"Unweighted Kaplan‑Meier\n{group} Vulnerability (n = {len(sub)}, {p_str})")
    ax[i].set_xlabel("PFS (months)")
    ax[i].set_ylabel("Survival probability")
    ax[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figure/Figure_5AB_KM_by_OVS.png", dpi=300, bbox_inches="tight")
plt.close()
print("KM curves exported")

# =========================================================
# 11. Forest plot 
# =========================================================
print("\nGenerating forest plot")

n_low = len(df[df["high_vulnerability"] == 0])
n_high = len(df[df["high_vulnerability"] == 1])

# P values based on pooled estimates (Olaparib vs Niraparib)
z_low = -beta_drug / se_low
p_low_olap = 2 * (1 - stats.norm.cdf(abs(z_low)))
z_high = (-beta_drug - beta_inter) / se_high
p_high_olap = 2 * (1 - stats.norm.cdf(abs(z_high)))

groups = [
    f"Low Vulnerability\nn = {n_low}\nP = {format_p_val(p_low_olap)}",
    f"High Vulnerability\nn = {n_high}\nP = {format_p_val(p_high_olap)}"
]
hr_vals = [hr_low, hr_high]
ci_lower = [ci_low[0], ci_high[0]]
ci_upper = [ci_low[1], ci_high[1]]

x_pos = [0.3, 0.6]
fig, ax = plt.subplots(figsize=(5, 4))

for i in range(len(groups)):
    ax.errorbar(x=x_pos[i], y=hr_vals[i],
                yerr=[[hr_vals[i] - ci_lower[i]], [ci_upper[i] - hr_vals[i]]],
                fmt='o', color='darkgreen', ecolor='red', capsize=6, markersize=8)
    ax.annotate(f"{hr_vals[i]:.3f}\n({ci_lower[i]:.3f}-{ci_upper[i]:.3f})",
                xy=(x_pos[i], hr_vals[i]), xytext=(0, 15), textcoords='offset points',
                fontsize=9, ha='center')

ax.axhline(1, color='gray', linestyle='--')
ax.set_xticks(x_pos)
ax.set_xticklabels(groups)
ax.set_ylabel("HR (Olaparib vs Niraparib)")
ax.set_title("IPTW-weighted Cox model")
ax.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("figure/Figure_5E_IPTW_Cox_HR.png", dpi=300, bbox_inches="tight")
plt.close()
print("Forest plot exported")

# =========================================================
# 12. Balance diagnostics (average SMD across MI datasets)
# =========================================================
print("\nGenerating Love plot and balance table")

# Average SMD across datasets
avg_balance = all_balance_tables[0].copy()
for i in range(1, m):
    avg_balance["SMD_before"] += all_balance_tables[i]["SMD_before"]
    avg_balance["SMD_after"] += all_balance_tables[i]["SMD_after"]
avg_balance["SMD_before"] /= m
avg_balance["SMD_after"] /= m
avg_balance = avg_balance.sort_values("SMD_before", key=lambda x: np.abs(x), ascending=False).reset_index(drop=True)

print("\nAverage Standardized Mean Difference (Signed, Before/After IPTW)")
print(avg_balance.to_string(index=False))

# Love plot
plt.figure(figsize=(8, 6))
ax = plt.gca()
y_pos = np.arange(len(avg_balance))
for i in range(len(avg_balance)):
    ax.plot([avg_balance.iloc[i]["SMD_before"], avg_balance.iloc[i]["SMD_after"]],
            [y_pos[i], y_pos[i]], color="lightgray", linewidth=1.2)
ax.scatter(avg_balance["SMD_before"], y_pos, color="steelblue", marker="o", s=60, label="Before IPTW", zorder=3)
ax.scatter(avg_balance["SMD_after"], y_pos, color="darkorange", marker="x", s=70, label="After IPTW", zorder=3)
plt.axvline(0.1, linestyle=':', color='red', linewidth=1)
ax.set_yticks(y_pos)
# labels
label_map = {
    'stage_1': 'FIGO Stage I',
    'stage_2': 'FIGO Stage II',
    'stage_3': 'FIGO Stage III',
    'stage_4': 'FIGO Stage IV',
    'treatment_line_0': 'Treatment: First-line',
    'treatment_line_1': 'Treatment: Second-line',
    'treatment_line_2': 'Treatment: Later-line',
    'brca_mutation_0': 'BRCA Status: Wild-type ',
    'brca_mutation_1': 'BRCA Status: Mutated',
    'brca_mutation_-1': 'BRCA Status: Unknown',
    'hrd_status_-1': 'HRD Status: Unknown',
    'hrd_status_0': 'HRD Status: Negative',
    'hrd_status_1': 'HRD Status: Positive',
    'age': 'Age',
}
ax.set_yticklabels([label_map.get(v, v) for v in avg_balance["Variable"]])

ax.set_xlabel("Absolute Standardized Mean Difference", fontsize=11)
ax.set_title("Covariate Balance Before and After IPTW", pad=15, fontsize=13)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.legend(frameon=True, loc="upper right", fontsize=10)
smd_range = np.max(np.abs(avg_balance[["SMD_before", "SMD_after"]].values))
ax.set_xlim(0, max(smd_range * 1.1, 0.20))
plt.tight_layout()
plt.savefig("figure/Figure_5G_Love_Plot_IPTW.png", dpi=300, bbox_inches="tight")
plt.close()

n_unbalanced_before = (avg_balance["SMD_before"].abs() > 0.1).sum()
n_unbalanced_after = (avg_balance["SMD_after"].abs() > 0.1).sum()
print(f"\nCovariates with |SMD|>0.1 BEFORE IPTW: {n_unbalanced_before}")
print(f"Covariates with |SMD|>0.1 AFTER IPTW: {n_unbalanced_after}")

avg_balance.to_excel("figure/PS_Balance_Diagnostics.xlsx", index=False)
print("Love plot and balance table exported")

# =========================================================
# 13. Propensity score overlap plot 
# =========================================================
print("\nGenerating propensity score overlap plot")

plt.figure(figsize=(8, 5))
sns.kdeplot(data=last_ps_df[last_y_ps == 0], x="ps", label="Olaparib", fill=True, alpha=0.3)
sns.kdeplot(data=last_ps_df[last_y_ps == 1], x="ps", label="Niraparib", fill=True, alpha=0.3)
plt.xlabel("Propensity Score")
plt.ylabel("Density")
plt.title("Propensity Score Overlap Between Treatment Groups")
plt.legend(title="Drug")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure/Figure_5F_PS_Overlap_Plot.png", dpi=300, bbox_inches="tight")
plt.close()
print("PS overlap plot exported")

# =========================================================
# 14. G-computation-based standardized survival curves (pooled)
# =========================================================
print("\n" + "=" * 80)
print("G-computation-based standardized survival curves (MI pooled)")
print("=" * 80)

time_grid = np.linspace(0, df["pfs_month"].max(), 100)
time_12m_idx = np.argmin(np.abs(time_grid - 12))

# Collect marginal survival functions for each imputed dataset
surv_low_ola_list = []
surv_low_nira_list = []
surv_high_ola_list = []
surv_high_nira_list = []

for mi_idx, (cph, mdf) in enumerate(zip(models, model_dfs)):
    low_df = mdf[mdf["high_vulnerability"] == 0].copy()
    high_df = mdf[mdf["high_vulnerability"] == 1].copy()

    # Counterfactual datasets
    low_ola = low_df.copy()
    low_ola["maintenance_drug"] = 0
    low_ola["drug_vuln_inter"] = 0

    low_nira = low_df.copy()
    low_nira["maintenance_drug"] = 1
    low_nira["drug_vuln_inter"] = 0   # because high_vulnerability=0

    high_ola = high_df.copy()
    high_ola["maintenance_drug"] = 0
    high_ola["drug_vuln_inter"] = 0   # 0 * 1 = 0: olaparib arm, interaction term = drug × vuln = 0
    high_nira = high_df.copy()
    high_nira["maintenance_drug"] = 1
    high_nira["drug_vuln_inter"] = 1   # high_vulnerability=1, drug=1 => interaction=1

    # Predict survival functions
    s_l_ola = cph.predict_survival_function(low_ola, times=time_grid)
    s_l_nira = cph.predict_survival_function(low_nira, times=time_grid)
    s_h_ola = cph.predict_survival_function(high_ola, times=time_grid)
    s_h_nira = cph.predict_survival_function(high_nira, times=time_grid)

    # Marginalize (average over subjects)
    surv_low_ola_list.append(s_l_ola.mean(axis=1))
    surv_low_nira_list.append(s_l_nira.mean(axis=1))
    surv_high_ola_list.append(s_h_ola.mean(axis=1))
    surv_high_nira_list.append(s_h_nira.mean(axis=1))

# Pooled survival (mean across MI datasets)
std_low_ola = pd.concat(surv_low_ola_list, axis=1).mean(axis=1)
std_low_nira = pd.concat(surv_low_nira_list, axis=1).mean(axis=1)
std_high_ola = pd.concat(surv_high_ola_list, axis=1).mean(axis=1)
std_high_nira = pd.concat(surv_high_nira_list, axis=1).mean(axis=1)

# 12‑month survival difference
low_diff = std_low_ola.iloc[time_12m_idx] - std_low_nira.iloc[time_12m_idx]
high_diff = std_high_ola.iloc[time_12m_idx] - std_high_nira.iloc[time_12m_idx]

print(f"Adjusted survival difference at 12 months:")
print(f"Low vulnerability:  {low_diff:.3f} (Olaparib - Niraparib)")
print(f"High vulnerability: {high_diff:.3f}")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

n_low_g = len(model_dfs[0][model_dfs[0]["high_vulnerability"] == 0])
n_high_g = len(model_dfs[0][model_dfs[0]["high_vulnerability"] == 1])

ax[0].plot(time_grid, std_low_ola, label="Olaparib", linewidth=2)
ax[0].plot(time_grid, std_low_nira, label="Niraparib", linewidth=2)
ax[0].set_title(f"IPTW-Standardized survival\nLow Vulnerability (n = {n_low_g})")
ax[0].set_xlabel("PFS (months)")
ax[0].set_ylabel("Adjusted Survival Probability")
ax[0].grid(True, alpha=0.3)
ax[0].legend(title="Drug")

ax[1].plot(time_grid, std_high_ola, label="Olaparib", linewidth=2)
ax[1].plot(time_grid, std_high_nira, label="Niraparib", linewidth=2)
ax[1].set_title(f"IPTW-Standardized survival\nHigh Vulnerability (n = {n_high_g})")
ax[1].set_xlabel("PFS (months)")
ax[1].set_ylabel("Adjusted Survival Probability")
ax[1].grid(True, alpha=0.3)
ax[1].legend(title="Drug")

plt.tight_layout()
plt.savefig("figure/Figure_5CD_Standardized_Survival.png", dpi=300, bbox_inches="tight")
plt.close()
print("G-computation survival curves exported")

# =========================================================
# 15. Save summary
# =========================================================
summary = pd.DataFrame({
    "Group": ["Low vulnerability", "High vulnerability", "Interaction"],
    "HR": [hr_low, hr_high, hr_interaction],
    "CI_lower": [ci_low[0], ci_high[0], ci_interaction[0]],
    "CI_upper": [ci_low[1], ci_high[1], ci_interaction[1]],
    "P_value": [p_low_olap, p_high_olap, p_interaction]
})
summary.to_excel("figure/Weighted_Cox_Summary.xlsx", index=False)

# =========================================================
# Export finalized analytical outputs
# =========================================================

print("\n" + "=" * 80)
print("Analysis completed")
print("=" * 80)
print("Exported files:")
print("  - figure/Figure_5AB_KM_by_OVS.png")
print("  - figure/Figure_5CD_GComp_Standardized_Survival.png")
print("  - figure/Figure_5E_IPTW_Cox_HR.png")
print("  - figure/Figure_5F_PS_Overlap_Plot.png")
print("  - figure/Figure_5G_Love_Plot_IPTW.png")
print("  - figure/PS_Balance_Diagnostics.xlsx")
print("  - figure/Weighted_Cox_Summary.xlsx")
