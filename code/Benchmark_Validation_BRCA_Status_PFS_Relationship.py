# -*- coding: utf-8 -*-

# =============================================================================
# Benchmark Analysis for BRCA-Associated Progression-Free Survival
# =============================================================================
#
# Study Context
# -----------------------------------------------------------------------------
# This script implements the benchmark component of the study evaluating
# treatment-effect heterogeneity among ovarian cancer patients receiving
# PARP inhibitor maintenance therapy.
#
# The benchmark analysis was designed as a methodological calibration step
# prior to the primary OVS-based heterogeneity analyses. Specifically,
# the objective was to assess whether the causal inference framework used
# in the main study could reproduce a well-established molecular association
# under real-world observational conditions.
#
# BRCA mutation status was selected as the benchmark molecular variable
# because BRCA-associated progression-free survival benefit has been
# consistently demonstrated in prior PARP inhibitor studies.
#
#
# Analytical Overview
# -----------------------------------------------------------------------------
# The analysis estimates the association between BRCA mutation status and
# progression-free survival (PFS) using inverse probability of treatment
# weighting (IPTW) combined with weighted Cox proportional hazards models.
#
# Three analytical scenarios are evaluated:
#
#   1. Complete-case analysis (CCA)
#      - Patients with missing baseline covariates are excluded.
#
#   2. Multiple imputation analysis (MI) [Primary benchmark analysis]
#      - BRCA mutation status itself is NEVER imputed.
#        Missing baseline covariates are multiply imputed using
#        IterativeImputer.
#      - Rubin's Rules are applied to pool estimates across imputations.
#
#   3. Platinum-sensitive subgroup analysis
#      - Sensitivity analysis restricted to platinum-sensitive patients.
#
#
# Missing Data Strategy
# -----------------------------------------------------------------------------
# BRCA mutation status itself is NEVER imputed.
#
# Only baseline covariates with incomplete observations are imputed:
#
#   - age
#   - FIGO stage
#   - treatment line
#
# This design is consistent with the study protocol and manuscript methods.
#
# Patients lacking BRCA testing are excluded from the benchmark cohort,
# because the objective of this benchmark analysis is specifically to
# evaluate whether the analytical framework reproduces known BRCA-associated
# survival patterns.
#
#
# Propensity Score Weighting
# -----------------------------------------------------------------------------
# Stabilized IPTW is estimated using logistic regression with:
#
#   - age
#   - stage
#   - treatment line
#
# To improve numerical stability and reduce extreme-weight influence:
#
#   - propensity scores are truncated to [0.01, 0.99]
#   - IPTW weights are truncated at the 1st and 99th percentiles
#
# Robust (sandwich) variance estimation is used in all weighted Cox models.
#
#
# Outcome Definition
# -----------------------------------------------------------------------------
# Outcome:
#   - Progression-free survival (PFS)
#
# Survival variables:
#   - pfs_month
#   - pfs_event
#
#
# Benchmark Objective
# -----------------------------------------------------------------------------
# This benchmark analysis is NOT intended as a definitive causal estimate
# of BRCA effect size.
#
# Instead, the purpose is to evaluate whether the analytical framework
# used in the primary OVS analysis can recover directionally consistent
# survival associations under observational real-world conditions.
#
#
# =============================================================================

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
from scipy.stats import norm
warnings.filterwarnings('ignore')


# ───────────────────────────────────────
# 1. Load data
# ───────────────────────────────────────
file_path = "parp_stats.xlsx"
df = pd.read_excel(file_path)

# ───────────────────────────────────────
# 2. Variable definition (STRICTLY MATCH METHODS)
# ───────────────────────────────────────
covariates = [
    'age',
    'stage',
    'treatment_line'
]

for col in covariates + ['brca_mutation','pfs_month','pfs_event','treatment_line']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df = df.dropna(subset=['pfs_month','pfs_event'])

# BRCA-complete benchmark cohort
# only patients with observed BRCA testing are retained
df_brca = df[df['brca_mutation'].notna()].copy()


# ───────────────────────────────────────
# 3. Ordinal covariate handling
# consistency with main analysis
# ───────────────────────────────────────

# stage and treatment_line
# are treated as ordinal numeric variables
# to preserve model parsimony and stability
# under limited sample size

df_brca['stage'] = (
    df_brca['stage']
    .astype(float)
)

df_brca['treatment_line'] = (
    df_brca['treatment_line']
    .astype(float)
)

# ───────────────────────────────────────
# 4. IPTW estimation (stabilized + truncated)
# ───────────────────────────────────────
def compute_iptw(df):

    X = df[covariates]
    y = df['brca_mutation']

    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X, y)

    ps = model.predict_proba(X)[:,1]

    # positivity safeguard:
    # avoid unstable IPTW estimates caused by extreme propensity scores
    ps = np.clip(ps, 0.01, 0.99)

    pA = y.mean()

    w = np.where(y==1, pA/ps, (1-pA)/(1-ps))

    # weight truncation:
    # reduce influence of extreme IPTW observations
    lower, upper = np.percentile(w, [1,99])
    w = np.clip(w, lower, upper)

    return w

# ───────────────────────────────────────
# 5. Cox model
# ───────────────────────────────────────
def run_cox(df):

    df = df.copy()

    df['iptw'] = compute_iptw(df)

    cph = CoxPHFitter()

    cph.fit(
        df[['pfs_month','pfs_event','brca_mutation','iptw']],
        duration_col='pfs_month',
        event_col='pfs_event',
        weights_col='iptw',
        robust=True
    )

    coef = cph.params_['brca_mutation']

    se = cph.standard_errors_['brca_mutation']

    p = cph.summary.loc[
        'brca_mutation',
        'p'
    ]

    return coef, se, p
# ───────────────────────────────────────
# 6. Rubin's Rules
# ───────────────────────────────────────
def rubin_pool(coefs, ses):

    m = len(coefs)

    Q_bar = np.mean(coefs)
    U_bar = np.mean(np.array(ses)**2)
    B = np.var(coefs, ddof=1)

    T = U_bar + (1 + 1/m) * B
    se_total = np.sqrt(T)

    z = Q_bar / se_total

    p = 2 * (1 - norm.cdf(abs(z)))

    return Q_bar, se_total, p

# ───────────────────────────────────────
# 7. CCA
# ───────────────────────────────────────
df_cca = df_brca.dropna(subset=covariates)
coef_cca, se_cca, p_cca = run_cox(df_cca)

# ───────────────────────────────────────
# 8. Multiple Imputation (STRICT: DO NOT IMPUTE BRCA)
# ───────────────────────────────────────
def multiple_imputation_analysis(df, m=5):

    coefs = []
    ses = []

    for i in range(m):

        imputer = IterativeImputer(
            random_state=42+i,
            max_iter=50,
            sample_posterior=True
        )

        df_imp = df.copy()

        # ONLY baseline covariates are imputed
        # BRCA status is intentionally excluded from imputation
        # to avoid imposing unverifiable assumptions on molecular missingness
        df_imp[covariates] = imputer.fit_transform(
            df_imp[covariates]
        )

        coef, se, _ = run_cox(df_imp)

        coefs.append(coef)
        ses.append(se)

    return rubin_pool(coefs, ses)

# ───────────────────────────────────────
# 9. MI (full cohort)
# ───────────────────────────────────────
coef_mi, se_mi, p_mi = multiple_imputation_analysis(df_brca, m=5)

# ───────────────────────────────────────
# 10. MI (platinum-sensitive subgroup)
# ───────────────────────────────────────
# platinum sensitivity is coded with 0, platinum resistant is coded with 1, adjust accordingly.
df_sens = df_brca[df_brca['platinum_sensitivity']==0].copy()
coef_sens, se_sens, p_sens = multiple_imputation_analysis(df_sens, m=5)

# ───────────────────────────────────────
# 11. Summary
# ───────────────────────────────────────
def summarize(coef, se):
    hr = np.exp(coef)
    ci_low = np.exp(coef - 1.96*se)
    ci_high = np.exp(coef + 1.96*se)
    return hr, ci_low, ci_high

print("\n===== Final Results =====")

for name, coef, se, n in [
    ("CCA", coef_cca, se_cca, len(df_cca)),
    ("MI", coef_mi, se_mi, len(df_brca)),
    ("MI (Platinum-sensitive)", coef_sens, se_sens, len(df_sens))
]:
    hr, l, u = summarize(coef, se)
    print(f"{name:30} | N={n:<4} | HR={hr:.3f} ({l:.3f}-{u:.3f})")


# ───────────────────────────────────────
# 11.1 Propensity Score Overlap
# Figure 4C
# propensity score overlap diagnostics
# ───────────────────────────────────────

import seaborn as sns
import matplotlib.pyplot as plt

print("\n" + "=" * 80)
print("📊 Propensity Score Overlap")
print("=" * 80)

# =========================================================
# compute IPTW on BRCA cohort
# =========================================================

df_ps = df_brca.dropna(
    subset=covariates
).copy()
df_ps['iptw'] = compute_iptw(df_ps)

# logistic model for visualization
X_ps = df_ps[covariates]
y_ps = df_ps['brca_mutation']

ps_model = LogisticRegression(
    solver='liblinear',
    max_iter=1000
)

ps_model.fit(X_ps, y_ps)

df_ps['ps'] = ps_model.predict_proba(X_ps)[:,1]

# =========================================================
# plotting style
# =========================================================

sns.set_theme(
    style="whitegrid",
    context="paper"
)


# =========================================================
# KDE overlap
# shaded density like main analysis
# =========================================================

plt.figure(figsize=(8, 5))
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_color('#cccccc')
    spine.set_linewidth(0.8)

sns.kdeplot(
    data=df_ps[y_ps == 0],
    x='ps',
    label='BRCA Wild-type',
    fill=True,
    alpha=0.30,
    linewidth=2,
    ax=ax
)

sns.kdeplot(
    data=df_ps[y_ps == 1],
    x='ps',
    label='BRCA Mutated',
    fill=True,
    alpha=0.30,
    linewidth=2,
    ax=ax
)

# =========================================================
# figure formatting
# =========================================================

ax.set_title(
    "Distribution of Propensity Scores by BRCA Status",
    pad=15
)

ax.set_xlabel("Propensity Score")

ax.set_ylabel("Density")

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    frameon=True
)

plt.tight_layout()
# restrict x-axis display to observed overlap region
# for visualization clarity
ax.set_xlim(0, 0.7)
# =========================================================
# save
# =========================================================

plt.savefig(
    "figure/Figure_4C_PS_Overlap_BRCA.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("\n✅ Figure 4C exported")
print("  - Figure_4C_PS_Overlap_BRCA.png")


# ───────────────────────────────────────
# 11.2 Covariate Balance Diagnostics
# Figure 4D
# SMD diagnostics
# ───────────────────────────────────────

print("\n" + "=" * 80)
print("📊 Covariate Balance Diagnostics")
print("=" * 80)

# =========================================================
# helper functions
# =========================================================

def compute_smd_continuous(x_treat, x_control):

    mean_t = np.mean(x_treat)
    mean_c = np.mean(x_control)

    sd_t = np.var(x_treat, ddof=1)
    sd_c = np.var(x_control, ddof=1)

    pooled_sd = np.sqrt((sd_t + sd_c) / 2)

    if pooled_sd == 0:
        return 0

    return (mean_t - mean_c) / pooled_sd


def compute_smd_binary(x_treat, x_control):

    p1 = np.mean(x_treat)
    p0 = np.mean(x_control)

    pooled = (
        p1 * (1 - p1)
        +
        p0 * (1 - p0)
    ) / 2

    if pooled == 0:
        return 0

    return (p1 - p0) / np.sqrt(pooled)

# =========================================================
# weighted dataset
# =========================================================

df_balance = df_ps.copy()

balance_results = []

# =========================================================
# continuous variables
# modeling: ordinal
# =========================================================

continuous_vars = [
    'age'
]

for var in continuous_vars:

    # raw
    t_raw = df_balance[
        df_balance['brca_mutation'] == 1
    ][var]

    c_raw = df_balance[
        df_balance['brca_mutation'] == 0
    ][var]

    smd_raw = compute_smd_continuous(
        t_raw,
        c_raw
    )

    # weighted
    wt_t = df_balance[
        df_balance['brca_mutation'] == 1
    ]

    wt_c = df_balance[
        df_balance['brca_mutation'] == 0
    ]

    mean_t_w = np.average(
        wt_t[var],
        weights=wt_t['iptw']
    )

    mean_c_w = np.average(
        wt_c[var],
        weights=wt_c['iptw']
    )

    var_t_w = np.average(
        (wt_t[var] - mean_t_w) ** 2,
        weights=wt_t['iptw']
    )

    var_c_w = np.average(
        (wt_c[var] - mean_c_w) ** 2,
        weights=wt_c['iptw']
    )

    pooled_sd_w = np.sqrt(
        (var_t_w + var_c_w) / 2
    )

    smd_w = (
        (mean_t_w - mean_c_w) / pooled_sd_w
        if pooled_sd_w != 0
        else 0
    )

    balance_results.append({
        'Variable': var,
        'SMD_before': abs(smd_raw),
        'SMD_after': abs(smd_w)
    })

# =========================================================
# categorical display for stage / platinum_lines
# =========================================================

categorical_display_vars = [
    'stage',
    'treatment_line'
]

for var in categorical_display_vars:

    temp_var = df_balance[var].fillna(-999).astype(int)  # 或 .astype(str)
    dummies = pd.get_dummies(temp_var, prefix=var)

    for dummy_col in dummies.columns:

        temp = pd.concat(
            [df_balance, dummies[dummy_col]],
            axis=1
        )

        # raw
        t_raw = temp[
            temp['brca_mutation'] == 1
        ][dummy_col]

        c_raw = temp[
            temp['brca_mutation'] == 0
        ][dummy_col]

        smd_raw = compute_smd_binary(
            t_raw,
            c_raw
        )

        # weighted
        wt_t = temp[
            temp['brca_mutation'] == 1
        ]

        wt_c = temp[
            temp['brca_mutation'] == 0
        ]

        p1_w = np.average(
            wt_t[dummy_col],
            weights=wt_t['iptw']
        )

        p0_w = np.average(
            wt_c[dummy_col],
            weights=wt_c['iptw']
        )

        pooled_w = (
            p1_w * (1 - p1_w)
            +
            p0_w * (1 - p0_w)
        ) / 2

        smd_w = (
            (p1_w - p0_w) / np.sqrt(pooled_w)
            if pooled_w != 0
            else 0
        )

        balance_results.append({
            'Variable': dummy_col,
            'SMD_before': abs(smd_raw),
            'SMD_after': abs(smd_w)
        })

# =========================================================
# balance table
# =========================================================

balance_table = pd.DataFrame(
    balance_results
)

balance_table = balance_table.sort_values(
    'SMD_before',
    ascending=False
).reset_index(drop=True)

print(balance_table)

# =========================================================
# Love plot
# consistent with main analysis
# =========================================================



plt.figure(figsize=(8, 6))
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_color('#cccccc')  
    spine.set_linewidth(0.8)

y_pos = np.arange(len(balance_table))

# connecting lines
for i in range(len(balance_table)):

    ax.plot(
        [
            balance_table.iloc[i]['SMD_before'],
            balance_table.iloc[i]['SMD_after']
        ],
        [y_pos[i], y_pos[i]],
        color='lightgray',
        linewidth=1
    )

# before IPTW
ax.scatter(
    balance_table['SMD_before'],
    y_pos,
    color='steelblue',
    marker='o',
    s=60,
    label='Before IPTW'
)

# after IPTW
ax.scatter(
    balance_table['SMD_after'],
    y_pos,
    color='darkorange',
    marker='x',
    s=70,
    label='After IPTW'
)

# threshold
ax.axvline(
    0.1,
    linestyle='--',
    color='red',
    linewidth=1
)

# labels
ax.set_yticks(y_pos)
label_map = {
    'stage_1': 'FIGO Stage I',
    'stage_2': 'FIGO Stage II',
    'stage_3': 'FIGO Stage III',
    'stage_4': 'FIGO Stage IV',
    'treatment_line_0': 'Treatment: First-line',
    'treatment_line_1': 'Treatment: Second-line',
    'treatment_line_2': 'Treatment: Later-line',
}
ax.set_yticklabels([label_map.get(v, v) for v in balance_table['Variable']])

ax.grid(False)
ax.set_xlabel(
    "Absolute Standardized Mean Difference"
)

ax.set_title(
    "Covariate Balance Before and After IPTW",
    pad=15
)
ax.grid(False)
ax.grid(
    axis='x',
    linestyle=':',
    alpha=0.5
)

ax.legend(
    frameon=True
)
ax.set_xlim(0, 0.2)
plt.tight_layout()

# =========================================================
# save
# =========================================================

plt.savefig(
    "figure/Figure_4D_Love_Plot_BRCA.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("\n✅ Figure 4D exported")
print("  - Figure_4D_Love_Plot_BRCA.png")


# ───────────────────────────────────────
# 12. Kaplan–Meier Curve
# ───────────────────────────────────────

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 80)
print("📊 Kaplan–Meier Survival Analysis")
print("=" * 80)



sns.set_theme(
    style="whitegrid",
    context="paper"
)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# optional multilingual font support
plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC',
    'PingFang SC',
    'DejaVu Sans'
]

plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# BRCA subgroup
# only complete BRCA cases included
# =========================================================

km_df = df[
    df['brca_mutation'].notna()
].copy()

km_df['brca_mutation'] = pd.to_numeric(
    km_df['brca_mutation'],
    errors='coerce'
)

km_df['pfs_month'] = pd.to_numeric(
    km_df['pfs_month'],
    errors='coerce'
)

km_df['pfs_event'] = pd.to_numeric(
    km_df['pfs_event'],
    errors='coerce'
)

# remove invalid survival rows
km_df = km_df.dropna(
    subset=['pfs_month', 'pfs_event']
)

km_df = km_df[
    km_df['pfs_month'] >= 0
]

# =========================================================
# BRCA encoding check
# =========================================================

print("\n📊 BRCA encoding distribution:")
print(
    km_df['brca_mutation']
    .value_counts()
    .sort_index()
)

# =========================================================
# subgroup definition
# =========================================================

mut = km_df[
    km_df['brca_mutation'] == 1
]

wt = km_df[
    km_df['brca_mutation'] == 0
]

n_mut = len(mut)
n_wt = len(wt)

print(f"\nBRCA mutated: n = {n_mut}")
print(f"BRCA wild-type: n = {n_wt}")

# =========================================================
# log-rank test
# non-parametric survival comparison
# =========================================================

lr = logrank_test(
    mut['pfs_month'],
    wt['pfs_month'],
    event_observed_A=mut['pfs_event'],
    event_observed_B=wt['pfs_event']
)

p_val = lr.p_value

p_text = (
    "P < 0.001"
    if p_val < 0.001
    else f"P = {p_val:.3f}"
)

print(f"\nLog-rank test: {p_text}")

# =========================================================
# Kaplan–Meier fitting
# =========================================================

kmf = KaplanMeierFitter()

fig, ax = plt.subplots(
    figsize=(8, 6)
)


for spine in ax.spines.values():
    spine.set_color('#cccccc')  
    spine.set_linewidth(0.8)

# -------------------------
# BRCA mutated
# -------------------------

kmf.fit(
    durations=mut['pfs_month'],
    event_observed=mut['pfs_event'],
    label=f"BRCA Mutated (n={n_mut})"
)

kmf.plot(
    ax=ax,
    ci_show=True,
    linewidth=2
)

# -------------------------
# BRCA wild-type
# -------------------------

kmf.fit(
    durations=wt['pfs_month'],
    event_observed=wt['pfs_event'],
    label=f"BRCA Wild-type (n={n_wt})"
)

kmf.plot(
    ax=ax,
    ci_show=True,
    linewidth=2
)


ax.set_title(
    f"Progression-Free Survival by BRCA Mutation Status\n{p_text}",
    pad=15
)

ax.set_xlabel("PFS (months)")

ax.set_ylabel(
    "Progression-Free Survival Probability"
)

ax.grid(
    True,
    alpha=0.3
)

ax.legend(
    loc='upper right',
    frameon=True
)

plt.tight_layout()

# =========================================================
# save figure
# =========================================================

plt.savefig(
    "figure/Figure_4A_KM_BRCA_PFS.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("\n✅ Kaplan–Meier figure exported")
print("  - Figure_4A_KM_BRCA_PFS.png")



# 15
#───────────────────────────────────────
# Extract the HR and CI for the forest plot
# ───────────────────────────────────────

# CCA
hr_cca, ci_low_cca, ci_high_cca = summarize(coef_cca, se_cca)
p_cca_final = p_cca  

# MI
hr_mi, ci_low_mi, ci_high_mi = summarize(coef_mi, se_mi)
p_mi_final = p_mi

# Platinum-sensitive subgroup
hr_sens, ci_low_sens, ci_high_sens = summarize(coef_sens, se_sens)
p_sens_final = p_sens


def format_p(p):
    if p < 0.001:
        return r"$P<0.001$"
    else:
        return f"$P={p:.3f}$"  

groups = [
    f'Complete-case \n n = {len(df_cca)} \n {format_p(p_cca_final)}',
    f'Multiple Imputation \n n = {len(df_brca)} \n {format_p(p_mi_final)}',
    f'Platinum-sensitive \n n = {len(df_sens)} \n {format_p(p_sens_final)}'
]
hr_values = [hr_cca, hr_mi, hr_sens]           
ci_low_values = [ci_low_cca, ci_low_mi, ci_low_sens]
ci_high_values = [ci_high_cca, ci_high_mi, ci_high_sens]
colors = ['steelblue', 'tomato', 'seagreen']

plt.figure(figsize=(8, 5))
ax = plt.gca()

for spine in ax.spines.values():
    spine.set_color('#cccccc')  
    spine.set_linewidth(0.8)
for i in range(len(groups)):
    hr = hr_values[i]
    ci_low = ci_low_values[i]
    ci_high = ci_high_values[i]

    
    yerr_lower = hr - ci_low
    yerr_upper = ci_high - hr

    
    yerr = np.array([[yerr_lower], [yerr_upper]])

    # Plot error bars
    plt.errorbar(
        i, hr,
        yerr=yerr,
        fmt='o',
        color=colors[i],
        ecolor=colors[i],
        capsize=6,
        markersize=8,
        linestyle='None'
    )

    # Mark the HR value above the dot
    plt.text(i, hr * 1.03, f"HR={hr:.3f}", 
             ha='center', va='bottom', fontsize=9, color=colors[i])
    
    
    plt.text(i, ci_low * 0.9, f"{ci_low:.3f}", 
             ha='center', va='bottom', fontsize=8, color=colors[i], alpha=0.7)
    
    
    plt.text(i, ci_high * 1.05, f"{ci_high:.3f}", 
             ha='center', va='top', fontsize=8, color=colors[i], alpha=0.7)
# Add a horizontal rule (HR=1)
plt.axhline(1, color='gray', linestyle='--', linewidth=1)

# Set the coordinate axes
plt.xticks(range(len(groups)), groups, rotation=0)
plt.ylabel('Hazard Ratio for Progression-Free Survival')
plt.title(
    'Consistency of IPTW-adjusted Hazard Ratios Across Analytical Strategies'
)
plt.ylim(0.3, 1.4)  
plt.grid(axis='y', linestyle=':', alpha=0.7)

# Save
plt.savefig("figure/Figure_4B_BRCA_HR_Consistency.png", dpi=300, bbox_inches='tight')
plt.close()