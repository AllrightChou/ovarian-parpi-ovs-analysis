# -*- coding: utf-8 -*-

# =========================================================
# Causal Forest HTE Analysis 
# =========================================================
#
# PURPOSE:
#   Estimate individualized conditional average treatment
#   effects (CATEs) of olaparib versus niraparib on PFS
#   using a Generalized Random Forest (GRF) framework,
#   consistent with the causal inference estimand.
#
# KEY DESIGN DECISIONS:
#   - Treatment encoding: 0 = Olaparib, 1 = Niraparib
#   - Negative CATE values indicate relatively more
#     favorable estimated outcomes under olaparib.
#   - Missing BRCA/HRD data encoded as -1 ("Unknown"),
#     reflecting potentially informative missingness.
#   - Clinical covariates imputed via IterativeImputer
#     (max_iter=50) under MAR assumption.
#   - IPTW applied as sample weights to adjust for
#     confounding in treatment assignment.
#   - Honesty enforced via GRF subsampling mechanism.
#   - All random seeds fixed at 42 for reproducibility.
# =========================================================


import matplotlib
matplotlib.use("Agg")

# =========================================================
# 1. Imports
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from econml.grf import CausalForest

# =========================================================
# 2. Helper Functions
# =========================================================

def format_p(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def compute_concordance_rate(cate, treatment):
    """
    Compute concordance rate between observed treatment
    and model-estimated favorable treatment.
    CATE < 0: olaparib preferred (treatment == 0 is concordant)
    CATE > 0: niraparib preferred (treatment == 1 is concordant)
    """
    concordant = (
        np.sum((cate < 0) & (treatment == 0))
        +
        np.sum((cate > 0) & (treatment == 1))
    )
    return concordant / len(cate) * 100
# =========================================================
# 3. Load Data
# =========================================================

file_path = (
    "parp_stats.xlsx"
)

df = pd.read_excel(file_path)

print("=" * 80)
print(" Raw Dataset")
print("=" * 80)
print(f"Original N = {len(df)}")

# =========================================================
# 4. Basic Cleaning
# =========================================================

required_cols = [
    "maintenance_drug",
    "pfs_month",
    "pfs_event",
    "organ_vulnerability_score"
]

for col in required_cols:

    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

# only binary treatment
df = df[
    df["maintenance_drug"].isin([0, 1])
].copy()

# remove missing survival
df = df[
    df["pfs_month"].notna()
    &
    df["pfs_event"].notna()
].copy()

# remove missing OVS
df = df[
    df["organ_vulnerability_score"].notna()
].copy()

print(f"\n Cleaned N = {len(df)}")

# =========================================================
# 5. OVS Group
# =========================================================

df["high_vulnerability"] = (
    df["organ_vulnerability_score"] >= 2
).astype(int)
df = df[df["high_vulnerability"].notna()].copy()
df = df[df["organ_vulnerability_score"].isin([0,1,2,3])].copy()

# =========================================================
# 6. BRCA / HRD Unknown Category
# =========================================================

for col in ["brca_mutation", "hrd_status"]:

    if col in df.columns:

        df[col] = (
            df[col]
            .fillna(-1)
            .astype(int)
        )

# =========================================================
# 7. Covariates
# =========================================================

covariates = [
    "age",
    "stage",
    "treatment_line",
    "brca_mutation",
    "hrd_status",
    "organ_vulnerability_score"
]

available_covariates = [
    c for c in covariates
    if c in df.columns
]

print("\n Covariates:")
print(available_covariates)

# =========================================================
# 8. Prepare Data
# =========================================================

analysis_df = df[
    [
        "pfs_month",
        "pfs_event",
        "maintenance_drug"
    ]
    +
    available_covariates
].copy()


# =========================================================
# 10. Multiple Imputation
# =========================================================

print("\n" + "=" * 80)
print(" Multiple Imputation")
print("=" * 80)

X_raw = analysis_df.drop(
    columns=[
        "pfs_month",
        "pfs_event",
        "maintenance_drug"
    ]
)

# Iterative imputation
imputer = IterativeImputer(
    max_iter=50,
    random_state=42,
    initial_strategy="median"
)

X_imputed = imputer.fit_transform(X_raw)

X_imputed = pd.DataFrame(
    X_imputed,
    columns=X_raw.columns
)

print(" Imputation complete")

# =========================================================
# 11. Standardization
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_imputed)

# =========================================================
# 12. IPCW Construction
# =========================================================
print("\n" + "=" * 80)
print(" IPCW Estimation (Stratified by Treatment)")
print("=" * 80)

analysis_df["censor_event"] = 1 - analysis_df["pfs_event"]
analysis_df["ipcw"] = np.nan

for trt in [0, 1]:
    mask = analysis_df["maintenance_drug"] == trt
    if mask.sum() == 0:
        continue
    
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=analysis_df.loc[mask, "pfs_month"],
        event_observed=analysis_df.loc[mask, "censor_event"]
    )
    surv = kmf.predict(analysis_df.loc[mask, "pfs_month"]).values
    surv = np.clip(surv, 0.05, 1.0)
    analysis_df.loc[mask, "ipcw"] = 1.0 / surv

analysis_df.drop(columns="censor_event", inplace=True)

print("\n Stratified IPCW summary:")
print(analysis_df.groupby("maintenance_drug")["ipcw"].describe())

# =========================================================
# 13. Define Variables
# =========================================================

Y = analysis_df["pfs_month"].values

T = analysis_df["maintenance_drug"].values.astype(int)

X = X_scaled

sample_weight = analysis_df["ipcw"].values

print("\n Outcome:")
print(f"PFS mean = {Y.mean():.2f} months")

print("\n Treatment distribution:")
print(pd.Series(T).value_counts())

# =========================================================
# 14. Honest Causal Forest
# =========================================================

print("\n" + "=" * 80)
print(" Honest Causal Forest")
print("=" * 80)

cf = CausalForest(
    n_estimators=300,
    min_samples_leaf=15,
    max_depth=10,
    honest=True,
    inference=True,
    random_state=42
)

cf.fit(
    X,
    T,
    Y,
    sample_weight=sample_weight
)

print(" Causal forest fitted")

# =========================================================
# 15. Predict CATE
# =========================================================

cate = cf.predict(X)

cate_lower, cate_upper = cf.predict_interval(
    X,
    alpha=0.05
)

cate = cate.flatten()
cate_lower = cate_lower.flatten()
cate_upper = cate_upper.flatten()

analysis_df["CATE"] = cate
analysis_df["CATE_lower"] = cate_lower
analysis_df["CATE_upper"] = cate_upper

# =========================================================
# 16. Summary Statistics
# =========================================================

print("\n" + "=" * 80)
print(" CATE Summary")
print("=" * 80)

print(f"Mean CATE = {np.mean(cate):.3f}")
print(f"Median CATE = {np.median(cate):.3f}")

n_total = len(cate)

n_nira = np.sum(cate > 0)

print(
    f"\nPrefer Niraparib: "
    f"{n_nira} "
    f"({n_nira / n_total * 100:.1f}%)"
)

print(
    f"Prefer Olaparib: "
    f"{n_total - n_nira} "
    f"({(n_total - n_nira) / n_total * 100:.1f}%)"
)

# =========================================================
# 17. Vulnerability Stratification
# =========================================================

print("\n" + "=" * 80)
print(" OVS Stratification")
print("=" * 80)

for group_name, group_val in [
    ("Low", 0),
    ("High", 1)
]:

    mask = (
        df["high_vulnerability"]
        == group_val
    )

    cate_group = cate[mask]

    n_group = len(cate_group)

    n_pref = np.sum(cate_group > 0)

    print(
        f"{group_name} vulnerability "
        f"(n={n_group}): "
        f"{n_pref} "
        f"({n_pref / n_group * 100:.1f}%) "
        f"prefer Niraparib"
    )

# =========================================================
# 18. Permutation Importance
# =========================================================

print("\n" + "=" * 80)
print(" Variable Importance")
print("=" * 80)

importance = pd.DataFrame({
    "Variable": X_imputed.columns,
    "Importance": cf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print(importance)

importance.to_excel(
    "CausalForest_Variable_Importance.xlsx",
    index=False
)

# =========================================================
# 19. Figure 6A
# CATE Distribution
# =========================================================

# =========================================================
# 19. Figure 6A
# CATE Distribution
# =========================================================

plt.figure(figsize=(8, 5))
sns.histplot(cate, bins=30, kde=True, color='skyblue', alpha=0.7, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='No effect')
plt.xlabel("Estimated PFS Difference (Niraparib − Olaparib, months)")
plt.ylabel("Count")
plt.title("Distribution of Individual Treatment Effects", fontsize=12)
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)

plt.tight_layout()
plt.legend()

plt.savefig(
    "Figure_6A_CATE_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 20. Figure 6B
# OVS Trend
# =========================================================

plot_df = df.copy().reset_index(drop=True)
plot_df["CATE"] = cate

plot_df["Treatment"] = (
    plot_df["maintenance_drug"]
    .map({
        0: "Olaparib",
        1: "Niraparib"
    })
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)


sns.stripplot(
    data=plot_df, x='organ_vulnerability_score', y='CATE', hue='Treatment',
    palette=['skyblue', 'coral'], alpha=0.5, jitter=True, size=5, ax=ax
)

group_means = (
    plot_df
    .groupby("organ_vulnerability_score")["CATE"]
    .mean()
)


ax.axvspan(1.8, 3.2, alpha=0.05, color='red')
ax.text(2.5, -3.8, 'High Vulnerability', fontsize=10, color='red', ha='center', va='top')

ax.set_xticks([0, 1, 2, 3])

ax.set_xlabel(
    "Organ Vulnerability Score"
)

ax.set_ylabel("Estimated PFS Difference (Niraparib − Olaparib, months)")

ax.set_title(
    "Individual Treatment Effect by Organ Vulnerability Score",
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], title='Drug', loc='upper right')
ax.axhline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)
ax.grid(True, linestyle='--', alpha=0.4)
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.grid(False)
plt.tight_layout()

plt.savefig(
    "Figure_6B_OVS_Trend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
# =========================================================
# 21. Figure 6C
# Therapeutic Alignment
# =========================================================



is_concordant = (
    ((cate > 0) & (T == 1))
    |
    ((cate < 0) & (T == 0))
)

is_discord_ola = (
    (cate > 0)
    &
    (T == 0)
)

is_discord_nira = (
    (cate < 0)
    &
    (T == 1)
)

pct_concordant = np.mean(is_concordant) * 100
pct_discord_ola = np.mean(is_discord_ola) * 100
pct_discord_nira = np.mean(is_discord_nira) * 100

fig, ax = plt.subplots(
    figsize=(8, 5)
)

labels = [
    "Aligned Assignments",
    "Model-Estimated \nPreference: Niraparib\n(Observed: Olaparib)",
    "Model-Estimated \nPreference: Olaparib\n(Observed: Niraparib)"
]

values = [
    pct_concordant,
    pct_discord_ola,
    pct_discord_nira
]

colors = ['#82ca9d', '#ff9999', '#ffcc99'] 


bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)


ax.set_ylabel(
    "Percentage of Cohort (%)"
)
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.set_title(
    "Concordance Between Observed Treatment Assignments\n"
    "and Model-Estimated Favorable Treatment",
    fontsize=11
)
ax.set_ylim(0, max(values) + 15)

ax.bar_label(bars, fmt='%.1f%%', fontsize=11, fontweight='bold')

ax.grid(False)
plt.tight_layout()

plt.savefig(
    "Figure_6C_Therapeutic_Alignment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================
# 22. Figure 6D
# Bootstrap Stability
# =========================================================

print("\n" + "=" * 80)
print(" Bootstrap Stability")
print("=" * 80)
observed_concordance = compute_concordance_rate(cate, T)
print(f"Observed concordance rate = {observed_concordance:.2f}%")
n_bootstrap = 1000

concordance_rates = []

for i in range(n_bootstrap):

    idx = np.random.choice(
        len(cate),
        size=len(cate),
        replace=True
    )

    cate_sample = cate[idx]
    t_sample = T[idx]

    rate = compute_concordance_rate(
        cate_sample,
        t_sample
    )

    concordance_rates.append(rate)

concordance_mean = np.mean(concordance_rates)

concordance_ci = np.percentile(
    concordance_rates,
    [2.5, 97.5]
)

print(
    f"Concordance mean = "
    f"{concordance_mean:.2f}%"
)

print(
    f"95% CI = "
    f"[{concordance_ci[0]:.2f}%, "
    f"{concordance_ci[1]:.2f}%]"
)

plt.figure(figsize=(10, 6))


sns.histplot(concordance_rates, bins=30, kde=True, color='lightgreen', alpha=0.8,edgecolor='black')
plt.axvline(
    observed_concordance,
    color='black',
    linestyle=':',
    linewidth=2,
    label=f'Observed: {observed_concordance:.1f}%'
)
# ==============================================

plt.axvline(
    concordance_mean,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {concordance_mean:.2f}%"
)

plt.xlabel(
    "Concordance Rate (%)"
)

plt.ylabel("Frequency")

plt.title(
    "Bootstrap Distribution of Concordance Rate"
)

plt.legend()

plt.grid(True, alpha=0.3)
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)

plt.tight_layout()

plt.savefig(
    "Figure_6D_Bootstrap_Stability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# =========================================================
# 23. Save Outputs
# =========================================================

analysis_df.to_excel(
    "CausalForest_CATE_Output.xlsx",
    index=False
)

print("\n" + "=" * 80)
print(" ALL ANALYSES COMPLETED")
print("=" * 80)

print("\nGenerated Outputs:")

outputs = [
    "Figure_6A_CATE_Distribution.png",
    "Figure_6B_OVS_Trend.png",
    "Figure_6C_Therapeutic_Alignment.png",
    "Figure_6D_Bootstrap_Stability.png",
    "CausalForest_CATE_Output.xlsx",
    "CausalForest_Variable_Importance.xlsx"
]

for f in outputs:
    print(f"  - {f}")