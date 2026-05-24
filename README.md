# README  

## PARP Inhibitor Treatment Heterogeneity Analysis  
This repository contains the analytical code accompanying our study on treatment-effect heterogeneity and organ vulnerability in patients receiving PARP inhibitor maintenance therapy.  
The project includes conventional survival analyses, inverse probability weighted causal inference models, causal forest–based heterogeneous treatment effect estimation, and robustness/sensitivity analyses.  

## Repository Structure  
```
code/
│
├── Benchmark_Validation_BRCA_Status_PFS_Relationship.py
├── Evaluation_Treatment_Effect_Heterogeneity_via_OVS.py
├── Exploratory_Analysis_of_Treatment_Heterogeneity_via_Causal_Forest.py
├── Robustness_and_Validation_Analyses.py
└── featureDict.py

```

## File Descriptions  
```
Benchmark_Validation_BRCA_Status_PFS_Relationship.py

```
Benchmark survival analyses evaluating the relationship between BRCA/HRD-related molecular status and progression-free survival (PFS).  
Main analyses include:  
* Kaplan–Meier survival analysis  
* Cox proportional hazards regression  
* Stratified subgroup analyses  
* Baseline validation analyses  
  
```
Evaluation_Treatment_Effect_Heterogeneity_via_OVs.py

```
Primary treatment-effect heterogeneity analysis using Organ Vulnerability Scores (OVSs).  
Main analyses include:  
* IPTW-adjusted Cox regression  
* Generalized estimating / marginal treatment effect analyses  
* Interaction testing between treatment and OVS  
* Subgroup-specific treatment effect estimation  
* Publication-oriented visualization outputs  
  
```
Exploratory_Analysis_of_Treatment_Heterogeneity_via_Causal_Forest.py

```
Machine learning–based exploratory heterogeneity analysis using causal forests. Because the outcome is specified as observed PFS duration rather than a censoring-adjusted survival estimand, results should be interpreted as exploratory assessments of differential treatment response patterns rather than confirmatory causal survival effect estimates. Inverse probability of censoring weights (IPCW) are applied as sample weights to partially account for censoring bias.  
Main analyses include:  
* Honest causal forest estimation with IPCW weighting  
* Individualized treatment response difference estimation  
* Concordance analysis between observed and model-estimated treatment preference  
* Variable importance estimation  
* Bootstrap stability analyses  
* Heterogeneity visualization figures  
  
```
Robustness_and_Validation_Analyses.py

```
Sensitivity and robustness analyses supporting the primary findings.  
Main analyses include:  
* Complete-case analysis (CCA) as sensitivity analysis  
* Subgroup consistency analyses across molecular and clinical subpopulations  
* OVS threshold invariance analyses  
* Bootstrap-based uncertainty assessment  
* E-value analysis for unmeasured confounding  
* Internal consistency assessment of the OVS construct  
* Specificity analysis comparing composite OVS against individual biomarker  
  
```
featureDict.py

```
Dictionary file defining variable names, feature mappings, and clinical annotations used throughout the project.  

## Main Dependencies  
Key Python packages used in this project include:  
```
pandas >= 1.3
numpy >= 1.21
scikit-learn >= 1.0
lifelines >= 0.27
econml >= 0.14
matplotlib >= 3.4
seaborn >= 0.11
statsmodels >= 0.13
scipy >= 1.7

```
Recommended Python version:  
```
Python >= 3.9

```

## Data Availability  
The original clinical dataset is not publicly released due to patient privacy and institutional restrictions.  
The repository therefore contains analysis code only.  

## Reproducibility  
To reproduce the analyses:  
1. Prepare the processed clinical dataset  
2. Update the input file paths in each script  
3. Run scripts sequentially according to the analysis workflow  
Suggested order:  
```
1. Benchmark_Validation_BRCA_Status_PFS_Relationship.py

```
```
2. Evaluation_Treatment_Effect_Heterogeneity_via_OVS.py
3. Exploratory_Analysis_of_Treatment_Heterogeneity_via_Causal_Forest.py

```
```
4. Robustness_and_Validation_Analyses.py

```

## Citation  
A formal citation will be provided upon publication. In the interim, please contact the corresponding author if you wish to reference this work.  

## License  
This project is licensed under the MIT License for academic and non-commercial research use. See LICENSE file for details.  
