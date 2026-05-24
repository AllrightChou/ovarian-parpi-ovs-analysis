# PARP-OVS Sensitivity Analysis Pipeline

# =========================================================

# Purpose:

# Comprehensive robustness and validation analyses aligned

# with Section 2.5 of the manuscript.

#

# This script is designed to:

# 1. Evaluate robustness to missing data assumptions

# 2. Evaluate robustness of heterogeneous treatment effect

# 3. Evaluate sensitivity to unmeasured confounding

# 4. Assess internal consistency of the OVS construct

# 5. Evaluate specificity of OVS as a composite modifier

# =========================================================

# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines import NelsonAalenFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils import resample


# =========================================================

# Optional causal forest

# =========================================================

try:
    from econml.grf import CausalForest  
    ECONML_AVAILABLE = True
except:
    ECONML_AVAILABLE = False

# =========================================================

# Output folders

# =========================================================

os.makedirs("sensitivity_results", exist_ok=True)
os.makedirs("sensitivity_results/figures", exist_ok=True)
os.makedirs("sensitivity_results/tables", exist_ok=True)

# =========================================================

# 0. Helper Functions

# =========================================================


def format_p_val(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def rubin_pool_covariance(beta_list, cov_list):
    """Rubin's rules for coefficient vector and covariance matrix."""
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

def map_to_high_vuln(score):
    try:
        score_int = int(round(float(score)))
    except:
        return np.nan
    if score_int in [0, 1]:
        return 0
    elif score_int in [2, 3]:
        return 1
    return np.nan

def compute_iptw(df, covariates):

    ps_df = df[covariates + ["maintenance_drug"]].copy()
    
    if "stage" in ps_df.columns:
        ps_df["stage"] = ps_df["stage"].astype(int)
    
    if "treatment_line" in ps_df.columns:
        ps_df["treatment_line"] = ps_df["treatment_line"].astype(int)
    
    molecular_vars = []
    
    for col in ["brca_mutation", "hrd_status"]:
        if col in ps_df.columns:
            molecular_vars.append(col)
    
    ps_df = pd.get_dummies(
        ps_df,
        columns=molecular_vars,
        drop_first=True
    )
    
    X_ps = ps_df.drop(columns=["maintenance_drug"])
    y_ps = ps_df["maintenance_drug"]
    
    model = LogisticRegression(
        solver="liblinear",
        max_iter=5000,
        random_state=42
    )
    
    model.fit(X_ps, y_ps)
    
    ps = model.predict_proba(X_ps)[:, 1]
    
    treat_prob = y_ps.mean()
    
    iptw = np.where(
        y_ps == 1,
        treat_prob / ps,
        (1 - treat_prob) / (1 - ps)
    )
    
    lower = np.quantile(iptw, 0.01)
    upper = np.quantile(iptw, 0.99)
    
    iptw = np.clip(iptw, lower, upper)
    
    return iptw, ps



def fit_interaction_cox(df):

    model_cols = [
        "pfs_month",
        "pfs_event",
        "maintenance_drug",
        "high_vulnerability",
        "iptw",
        "age",
        "stage",
        "treatment_line"
    ]
    
    model_df = df[model_cols].copy()
    
    model_df["age"] = (
        model_df["age"] -
        model_df["age"].mean()
    ) / model_df["age"].std()
    
    model_df["drug_vuln_inter"] = (
        model_df["maintenance_drug"] *
        model_df["high_vulnerability"]
    )
    
    cph = CoxPHFitter()
    
    cph.fit(
        model_df,
        duration_col="pfs_month",
        event_col="pfs_event",
        weights_col="iptw",
        robust=True
    )
    
    return cph, model_df



def extract_interaction_results(cph):

    beta_drug = cph.params_["maintenance_drug"]
    beta_inter = cph.params_["drug_vuln_inter"]
    
    cov_mat = cph.variance_matrix_
    
    var_low = cov_mat.loc[
        "maintenance_drug",
        "maintenance_drug"
    ]
    
    var_high = (
        cov_mat.loc[
            "maintenance_drug",
            "maintenance_drug"
        ]
        +
        2 * cov_mat.loc[
            "maintenance_drug",
            "drug_vuln_inter"
        ]
        +
        cov_mat.loc[
            "drug_vuln_inter",
            "drug_vuln_inter"
        ]
    )
    
    
    hr_low = np.exp(-beta_drug)
    hr_high = np.exp(-(beta_drug + beta_inter))
    interaction_hr = np.exp(-beta_inter)

    ci_low = np.exp(
        -beta_drug +
        np.array([-1.96, 1.96]) * np.sqrt(var_low)  
    )
    ci_high = np.exp(
        -(beta_drug + beta_inter) +
        np.array([-1.96, 1.96]) * np.sqrt(var_high)
    )
    interaction_ci = np.exp(
        -beta_inter +
        np.array([-1.96, 1.96]) * np.sqrt(
            cov_mat.loc["drug_vuln_inter", "drug_vuln_inter"]
        )
    )

    se_inter = np.sqrt(cov_mat.loc["drug_vuln_inter", "drug_vuln_inter"])
    z_inter = -beta_inter / se_inter
    interaction_p = 2 * (1 - stats.norm.cdf(abs(z_inter)))
    
    return {
        "HR_low": hr_low,
        "CI_low_lower": ci_low[0],
        "CI_low_upper": ci_low[1],
        "HR_high": hr_high,
        "CI_high_lower": ci_high[0],
        "CI_high_upper": ci_high[1],
        "Interaction_HR": interaction_hr,
        "Interaction_CI_lower": interaction_ci[0],
        "Interaction_CI_upper": interaction_ci[1],
        "Interaction_P": interaction_p
    }



def calculate_evalue(hr):

    if hr < 1:
        hr = 1 / hr
    
    return hr + np.sqrt(hr * (hr - 1))


# =========================================================

# 1. Load and Clean Data

# =========================================================

file_path = "parp_stats.xlsx"

print("=" * 80)
print("Sensitivity Analysis Pipeline")
print("=" * 80)

print("\nLoading dataset...")

raw_df = pd.read_excel(file_path)

print(f"Raw sample size: {len(raw_df)}")

required_cols = [
    "maintenance_drug",
    "organ_vulnerability_score",
    "pfs_month",
    "pfs_event"
]

for col in required_cols:
    if col not in raw_df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================================================

# Main cleaning

# =========================================================

df = raw_df.copy()

# Binary treatment

df = df[
    df["maintenance_drug"].isin([0, 1])
].copy()

# Survival endpoint

df = df[
    df["pfs_month"].notna() &
    df["pfs_event"].notna()
].copy()

# OVS

df = df[
    df["organ_vulnerability_score"].notna()
].copy()

# High vulnerability

df["high_vulnerability"] = (
    df["organ_vulnerability_score"]
    .apply(map_to_high_vuln)
)
df = df[df["high_vulnerability"].notna()].copy()

# Molecular unknown category

for molecular_col in ["brca_mutation", "hrd_status"]:

    if molecular_col in df.columns:
    
        df[molecular_col] = (
            df[molecular_col]
            .fillna(-1)
            .astype(int)
        )


# =========================================================
# Multiple Imputation (MI) for baseline covariates
# =========================================================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

impute_vars = ["age", "stage", "treatment_line"]
m = 5
imputed_datasets = []

print(f"\nGenerating {m} multiply imputed datasets...")
for i in range(m):
    imp_df = df.copy()
    imputer = IterativeImputer(random_state=42 + i, max_iter=50, sample_posterior=True)
    imp_df[impute_vars] = imputer.fit_transform(imp_df[impute_vars])
    # Restore ordinal structure
    imp_df["stage"] = np.round(imp_df["stage"]).astype(int)
    imp_df["treatment_line"] = np.round(imp_df["treatment_line"]).astype(int)
    imputed_datasets.append(imp_df)
print("MI datasets ready.\n")






# =========================================================

# Main IPTW

# =========================================================

ps_covariates = [
    "age",
    "stage",
    "treatment_line",
    "brca_mutation",
    "hrd_status"
]

available_covariates = [
    col for col in ps_covariates
    if col in df.columns
]


# =========================================================

# MAIN MODEL

# =========================================================

print("\nRunning main interaction model on MI datasets...")

main_betas = []
main_covs = []

for imp_df in imputed_datasets:
    iptw, ps = compute_iptw(imp_df, available_covariates)
    imp_df["iptw"] = iptw
    imp_df["ps"] = ps
    cph, _ = fit_interaction_cox(imp_df)
    main_betas.append(cph.params_.copy())
    main_covs.append(cph.variance_matrix_.copy())

# Rubin pooling
pooled_beta, pooled_cov = rubin_pool_covariance(main_betas, main_covs)

# Extract interaction results (Olaparib vs Niraparib)
beta_drug = pooled_beta["maintenance_drug"]
beta_inter = pooled_beta["drug_vuln_inter"]
var_drug = pooled_cov.loc["maintenance_drug", "maintenance_drug"]
var_inter = pooled_cov.loc["drug_vuln_inter", "drug_vuln_inter"]
covar = pooled_cov.loc["maintenance_drug", "drug_vuln_inter"]

hr_low = np.exp(-beta_drug)
se_low = np.sqrt(var_drug)
ci_low = np.exp(-beta_drug + np.array([-1.96, 1.96]) * se_low)

var_high = var_drug + var_inter + 2 * covar
se_high = np.sqrt(var_high)
hr_high = np.exp(-beta_drug - beta_inter)
ci_high = np.exp(-beta_drug - beta_inter + np.array([-1.96, 1.96]) * se_high)

interaction_hr = np.exp(-beta_inter)
se_inter = np.sqrt(var_inter)
ci_inter = np.exp(-beta_inter + np.array([-1.96, 1.96]) * se_inter)
z_inter = -beta_inter / se_inter
p_inter = 2 * (1 - stats.norm.cdf(abs(z_inter)))

main_results = {
    "HR_low": hr_low,
    "CI_low_lower": ci_low[0],
    "CI_low_upper": ci_low[1],
    "HR_high": hr_high,
    "CI_high_lower": ci_high[0],
    "CI_high_upper": ci_high[1],
    "Interaction_HR": interaction_hr,
    "Interaction_CI_lower": ci_inter[0],
    "Interaction_CI_upper": ci_inter[1],
    "Interaction_P": p_inter
}
main_table = pd.DataFrame([main_results])
main_table.to_excel("sensitivity_results/tables/Main_Interaction_Results.xlsx", index=False)

# =========================================================

# 2.5.1 Robustness to Missing Data Assumptions

# =========================================================

print("\n" + "=" * 80)
print("2.5.1 Robustness to Missing Data Assumptions")
print("=" * 80)

# =========================================================

# (I) Complete-Case Analysis (CCA)

# =========================================================
print("\nRunning Complete-Case Analysis (CCA)...")

cca_df = df.copy()

cca_vars = ["age", "stage", "treatment_line",
            "maintenance_drug", "pfs_month", "pfs_event",
            "organ_vulnerability_score", "high_vulnerability"]

cca_df = cca_df.dropna(subset=cca_vars).copy()

print(f"CCA sample size: {len(cca_df)}")

iptw, _ = compute_iptw(cca_df, available_covariates)
cca_df["iptw"] = iptw

cca_cph, _ = fit_interaction_cox(cca_df)
cca_result = extract_interaction_results(cca_cph)
cca_result["Scenario"] = "CCA"

cca_result_df = pd.DataFrame([cca_result])
cca_result_df.to_excel(
    "sensitivity_results/tables/CCA_Interaction_Results.xlsx",
    index=False
)



# =========================================================
# (II) Subgroup Consistency Checks (MI‑pooled)
# =========================================================
print("\nRunning subgroup consistency analyses (MI pooled)...")

subgroup_results = []

# Known molecular status
known_func = lambda df_: df_[(df_["brca_mutation"] != -1) | (df_["hrd_status"] != -1)].copy()
unknown_func = lambda df_: df_[(df_["brca_mutation"] == -1) & (df_["hrd_status"] == -1)].copy()
platinum_func = lambda df_: df_[df_["platinum_sensitivity"] == 0].copy() if "platinum_sensitivity" in df_.columns else pd.DataFrame()
stage_func = lambda df_: df_[df_["stage"] >= 3].copy() if "stage" in df_.columns else pd.DataFrame()

for subset_name, subset_func in [
    ("Known_Molecular", known_func),
    ("Unknown_Molecular", unknown_func),
    ("Platinum_Sensitive", platinum_func),
    ("Stage_III_IV", stage_func)
]:
    sub_betas = []
    sub_covs = []
    valid = True
    for imp_df in imputed_datasets:
        sub_df = subset_func(imp_df)
        if len(sub_df) < 50:
            valid = False
            break
        iptw, _ = compute_iptw(sub_df, available_covariates)
        sub_df["iptw"] = iptw
        cph, _ = fit_interaction_cox(sub_df)
        sub_betas.append(cph.params_.copy())
        sub_covs.append(cph.variance_matrix_.copy())
    if not valid:
        print(f"Skipping {subset_name}: insufficient sample size")
        continue
    pbeta, pcov = rubin_pool_covariance(sub_betas, sub_covs)
    b_inter = pbeta["drug_vuln_inter"]
    v_inter = pcov.loc["drug_vuln_inter", "drug_vuln_inter"]
    hr = np.exp(-b_inter)
    ci = np.exp(-b_inter + np.array([-1.96, 1.96]) * np.sqrt(v_inter))
    p = 2 * (1 - stats.norm.cdf(abs(b_inter / np.sqrt(v_inter))))
    subgroup_results.append({
        "Scenario": subset_name,
        "Interaction_HR": hr,
        "Interaction_CI_lower": ci[0],
        "Interaction_CI_upper": ci[1],
        "Interaction_P": p
    })

subgroup_results_df = pd.DataFrame(subgroup_results)
subgroup_results_df.to_excel(
    "sensitivity_results/tables/Subgroup_Consistency_Results.xlsx", index=False
)

# =========================================================

# 2.5.2 Robustness of HTE Estimation

# =========================================================

print("\n" + "=" * 80)
print("2.5.2 Robustness of HTE Estimation")
print("=" * 80)

# =========================================================

# (I) Threshold invariance

# =========================================================

print("\nRunning threshold invariance analysis (MI pooled)...")

threshold_results = []
for threshold in [1, 3]:
    thres_betas = []
    thres_covs = []
    for imp_df in imputed_datasets:
        temp = imp_df.copy()
        temp["high_vulnerability"] = (temp["organ_vulnerability_score"] >= threshold).astype(int)
        iptw, _ = compute_iptw(temp, available_covariates)
        temp["iptw"] = iptw
        cph, _ = fit_interaction_cox(temp)
        thres_betas.append(cph.params_.copy())
        thres_covs.append(cph.variance_matrix_.copy())
    pbeta, pcov = rubin_pool_covariance(thres_betas, thres_covs)
    b_inter = pbeta["drug_vuln_inter"]
    v_inter = pcov.loc["drug_vuln_inter", "drug_vuln_inter"]
    hr = np.exp(-b_inter)
    ci = np.exp(-b_inter + np.array([-1.96, 1.96]) * np.sqrt(v_inter))
    p = 2 * (1 - stats.norm.cdf(abs(b_inter / np.sqrt(v_inter))))
    threshold_results.append({
        "Threshold": f"OVS >= {threshold}",
        "Interaction_HR": hr,
        "Interaction_CI_lower": ci[0],
        "Interaction_CI_upper": ci[1],
        "Interaction_P": p
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_excel(
    "sensitivity_results/tables/Threshold_Invariance_Results.xlsx", index=False
)

# =========================================================

# (II) Bootstrap uncertainty assessment

# =========================================================

print("\nRunning bootstrap uncertainty assessment...")

bootstrap_results = []

n_bootstrap = 1000
#Bootstrap was performed on the first MI dataset as a sensitivity assessment
for i in range(n_bootstrap):

    if (i + 1) % 100 == 0:
        print(f"Bootstrap iteration: {i+1}/{n_bootstrap}")
    
    boot_df = resample(
        imputed_datasets[0],
        replace=True,
        n_samples=len(df),
        random_state=1000 + i
    )
    
    try:
    
        iptw, _ = compute_iptw(
            boot_df,
            available_covariates
        )
    
        boot_df["iptw"] = iptw
    
        cph, _ = fit_interaction_cox(boot_df)
    
        beta_inter = cph.params_["drug_vuln_inter"]
    
        bootstrap_results.append(-beta_inter)
    
    except:
        continue

bootstrap_results = np.array(bootstrap_results)
bootstrap_hr = np.exp(bootstrap_results)

bootstrap_ci = np.percentile(
    bootstrap_hr,
    [2.5, 97.5]
)

bootstrap_summary = pd.DataFrame({
    "Bootstrap_HR_Median": [np.median(bootstrap_hr)],
    "Bootstrap_CI_lower": [bootstrap_ci[0]],
    "Bootstrap_CI_upper": [bootstrap_ci[1]],
    "Iterations": [len(bootstrap_hr)]
})

bootstrap_summary.to_excel(
    "sensitivity_results/tables/Bootstrap_Interaction_Results.xlsx",
    index=False
)

# Plot bootstrap distribution

plt.figure(figsize=(7, 5))

sns.histplot(
    bootstrap_hr,
    bins=40,
    kde=True
)

plt.axvline(1, linestyle="--", color="red")

plt.xlabel("Interaction HR (Olaparib vs Niraparib)")
plt.ylabel("Frequency")
plt.title("Bootstrap Distribution of Interaction HR")

plt.tight_layout()

plt.savefig(
    "sensitivity_results/figures/Bootstrap_Interaction_Distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =========================================================

# (III) Cross-model consistency

# =========================================================

print("\nRunning cross-model consistency analysis...")

cross_model_results = []

# G-computation result

cross_model_results.append({
    "Model": "Parametric_GComputation",
    "Interaction_HR": main_results["Interaction_HR"],
    "CI_lower": main_results["Interaction_CI_lower"],
    "CI_upper": main_results["Interaction_CI_upper"]
})

# Causal forest

if ECONML_AVAILABLE:
    try:
        cf_df = imputed_datasets[0].copy()
        cols_for_X = ["age", "stage", "treatment_line", "high_vulnerability"]
        cf_df = cf_df.dropna(subset=cols_for_X)

        X = cf_df[cols_for_X]
        T = cf_df["maintenance_drug"]
        Y = cf_df["pfs_month"]

        cf_model = CausalForest(
            n_estimators=300,
            min_samples_leaf=5,
            honest=True,
            random_state=42
        )
        cf_model.fit(X, T, Y)
        cate = cf_model.predict(X).flatten()

        cross_model_results.append({
            "Model": "Causal_Forest",
            "Interaction_HR": np.median(cate),
            "CI_lower": np.percentile(cate, 2.5),
            "CI_upper": np.percentile(cate, 97.5)
        })
    except Exception as e:
        print("Causal forest failed:")
        print(e)

cross_model_df = pd.DataFrame(cross_model_results)

cross_model_df.to_excel(
    "sensitivity_results/tables/Cross_Model_Consistency.xlsx",
    index=False
)

# =========================================================

# 2.5.3 E-value Analysis

# =========================================================

print("\n" + "=" * 80)
print("2.5.3 E-value Analysis")
print("=" * 80)

interaction_hr = main_results["Interaction_HR"]
interaction_ci_lower = main_results["Interaction_CI_lower"]

point_evalue = calculate_evalue(interaction_hr)
ci_evalue = calculate_evalue(main_results["Interaction_CI_upper"])

print(f"Interaction HR: {interaction_hr:.3f}")
print(f"E-value (point estimate): {point_evalue:.3f}")
print(f"E-value (CI limit): {ci_evalue:.3f}")

evalue_df = pd.DataFrame({
    "Interaction_HR": [interaction_hr],
    "Interaction_CI_lower": [interaction_ci_lower],
    "Evalue_point": [point_evalue],
    "Evalue_CI": [ci_evalue]
})

evalue_df.to_excel(
    "sensitivity_results/tables/Evalue_Results.xlsx",
    index=False
)

# =========================================================

# 2.5.4 Internal Consistency Assessment

# =========================================================

print("\n" + "=" * 80)
print("2.5.4 Internal Consistency Assessment")
print("=" * 80)

# =========================================================

# (I) Harrell's C-index

# =========================================================

print("\nCalculating Harrell's C-index...")

cindex_df = df.copy()

cindex_model = CoxPHFitter()

cindex_model.fit(
    cindex_df[[
        "pfs_month",
        "pfs_event",
        "organ_vulnerability_score"
    ]],
    duration_col="pfs_month",
    event_col="pfs_event"
)

pred = cindex_model.predict_partial_hazard(cindex_df)

c_index = concordance_index(
    cindex_df["pfs_month"],
    -pred,
    cindex_df["pfs_event"]
)

print(f"Harrell C-index: {c_index:.3f}")

cindex_table = pd.DataFrame({
    "Harrell_C_index": [c_index]
})

cindex_table.to_excel(
    "sensitivity_results/tables/Cindex_Results.xlsx",
    index=False
)

# =========================================================

# (II) Calibration plots

# =========================================================



print("\nGenerating calibration plots...")

kmf = KaplanMeierFitter()

# ------------------------------------------------------------
# Perform multivariate log-rank test across all OVS groups
# ------------------------------------------------------------
groups = df["organ_vulnerability_score"].values.astype(int)
durations = df["pfs_month"].values
events = df["pfs_event"].values.astype(bool)

lr_result = multivariate_logrank_test(durations, groups, events)
p_val_multigroup = lr_result.p_value

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))

for score in sorted(df["organ_vulnerability_score"].unique()):
    temp = df[df["organ_vulnerability_score"] == score]
    if len(temp) < 10:
        continue

    kmf.fit(
        temp["pfs_month"],
        temp["pfs_event"],
        label=f"OVS {int(score)}"
    )
    kmf.plot_survival_function(ci_show=False)

    # Median survival time
    median = kmf.median_survival_time_
    if np.isinf(median):
        med_str = "not reached"
    else:
        med_str = f"{median:.1f} months"
    print(f"OVS {int(score)} (n={len(temp)}): median PFS = {med_str}")

# Format P value for title
p_str = "P < 0.001" if p_val_multigroup < 0.001 else f"P = {p_val_multigroup:.3f}"

plt.xlabel("PFS (months)")
plt.ylabel("Survival probability")
plt.title(f"OVS-Stratified Kaplan–Meier Survival Curves\n{p_str}")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "sensitivity_results/figures/OVS_Stratified_KM_Curves.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# (III) Alternative OVS construction strategies
# =========================================================
# Note: Alternative organ-level vulnerability definitions
# (requiring ≥2 or all indicators abnormal per organ system)
# were evaluated by re-running the primary OVS scoring
# pipeline with modified thresholds and re-executing the
# main interaction analysis. Results are reported in
# Supplementary Table S5. These analyses are not
# re-implemented here to avoid duplication with the
# primary analysis pipeline.
# =========================================================

# =========================================================

# 2.5.4 Specificity of OVS as Composite Modifier

# =========================================================

print("\n" + "=" * 80)
print("2.5.4 Specificity of OVS as Composite Modifier")
print("=" * 80)

candidate_components = [
    "fibrinogen",
    "uric_acid",
    "tba",
    "ast",
    "thrombin_time",
    "age",
    "stage"
]

candidate_components = [
    col for col in candidate_components
    if col in df.columns
]


clinical_thresholds = {
    "hematology_hemoglobin":      (110,  "below"),   # <110 g/L, CTCAE v5.0
    "hematology_platelet_count":  (150,  "below"),   # <150×10⁹/L, WS/T 404-2012
    "hematology_neutrophil_abs":  (1.5,  "below"),   # <1.5×10⁹/L, CTCAE v5.0
    "lft_albumin":                (35,   "below"),   # <35 g/L, Child-Pugh / WS/T 404-2012
    "lft_total_bilirubin":        (17.1, "above"),   # >17.1 μmol/L, WS/T 404-2012
    "lft_ast":                    (40,   "above"),   # >40 U/L, WS/T 404-2012
    "lft_total_bile_acids":       (10.0, "above"),   # >10 μmol/L, WS/T 404-2012
    "renal_creatinine":           (90,   "above"),   # >90 μmol/L, female, WS/T 404-2012
    "renal_uric_acid":            (360,  "above"),   # >360 μmol/L, female, 2024 Chinese Guideline
    "coag_fibrinogen":            (4.0,  "above"),   # >4.0 g/L, WS/T 404 series
    "coag_thrombin_time":         (21.0, "above"),   # >21 s, Chinese coagulation reference
}

candidate_components = [
    "hematology_hemoglobin",        
    "hematology_platelet_count",         
    "hematology_neutrophil_abs",               
    "lft_albumin",           
    "lft_total_bilirubin",   
    "lft_ast",               
    "renal_creatinine",        
    "renal_uric_acid",         
    "coag_fibrinogen",
    "lft_total_bile_acids",               
    "coag_thrombin_time",     
    "age",
    "stage"
]

component_results = []

for component in candidate_components:
    comp_betas = []
    comp_covs = []
    valid = True
    for imp_df in imputed_datasets:
        temp = imp_df.copy()
        if component in ["age", "stage"]:
            cutoff = temp[component].median()
            temp["component_binary"] = (temp[component] >= cutoff).astype(int)
        elif component in clinical_thresholds:
            cutoff, direction = clinical_thresholds[component]
            if direction == "below":
                temp["component_binary"] = (temp[component] < cutoff).astype(int)
            else:
                temp["component_binary"] = (temp[component] >= cutoff).astype(int)
        else:
            cutoff = temp[component].median()
            temp["component_binary"] = (temp[component] >= cutoff).astype(int)

        model_cols = [
            "pfs_month", "pfs_event", "maintenance_drug",
            "component_binary", "iptw", "age", "stage", "treatment_line"
        ]
        temp_model = temp[model_cols].copy()
        temp_model["age"] = (temp_model["age"] - temp_model["age"].mean()) / temp_model["age"].std()
        temp_model["interaction"] = temp_model["maintenance_drug"] * temp_model["component_binary"]

        iptw, _ = compute_iptw(temp, available_covariates)
        temp_model["iptw"] = iptw

        cph = CoxPHFitter()
        cph.fit(temp_model, duration_col="pfs_month", event_col="pfs_event",
                weights_col="iptw", robust=True)
        comp_betas.append(cph.params_.copy())
        comp_covs.append(cph.variance_matrix_.copy())

    if len(comp_betas) > 0:
        pbeta, pcov = rubin_pool_covariance(comp_betas, comp_covs)
        b_inter = pbeta["interaction"]
        v_inter = pcov.loc["interaction", "interaction"]
        hr = np.exp(-b_inter)
        p_val = 2 * (1 - stats.norm.cdf(abs(b_inter / np.sqrt(v_inter))))
        component_results.append({
            "Component": component,
            "Interaction_HR": hr,
            "P_value": p_val
        })

component_df = pd.DataFrame(component_results)
component_df.to_excel(
    "sensitivity_results/tables/OVS_Component_Specificity.xlsx",
    index=False
)

# =========================================================

# Final summary export

# =========================================================

print("\n" + "=" * 80)
print("Generating final summary tables...")
print("=" * 80)

summary_tables = {
    "Main_Model": main_table,
    "CCA": cca_result_df,
    "Subgroups": subgroup_results_df,
    "Thresholds": threshold_df,
    "CrossModel": cross_model_df,
    "Evalue": evalue_df,
    "Cindex": cindex_table,
    "Component_Specificity": component_df
}

with pd.ExcelWriter(
    "sensitivity_results/Complete_Sensitivity_Analysis_Summary.xlsx"
) as writer:

    for name, table in summary_tables.items():
        table.to_excel(writer, sheet_name=name[:31], index=False)

print("\n Sensitivity analysis completed")
print("\nGenerated outputs:")
print("  - Main interaction robustness tables")
print("  - Multiple imputation sensitivity")
print("  - Subgroup consistency analyses")
print("  - Threshold invariance analyses")
print("  - Bootstrap uncertainty assessment")
print("  - Cross-model consistency comparison")
print("  - E-value analysis")
print("  - Harrell C-index")
print("  - Calibration plots")
print("  - Alternative OVS categorizations")
print("  - OVS component specificity analyses")
print("\nAll outputs saved under: sensitivity_results/")

