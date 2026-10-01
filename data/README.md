# Data

The study uses the **BCSC Risk Factors dataset** (2020 release) from the
[Breast Cancer Surveillance Consortium](https://www.bcsc-research.org/). It is
distributed by the BCSC under their own terms, so it isn't included here.

Download the three files from the BCSC website and place them in this folder:

```
data/
├── bcsc_risk_factors_summarized1_092020.csv
├── bcsc_risk_factors_summarized2_092020.csv
└── bcsc_risk_factors_summarized3_092020.csv
```

Each row is a combination of risk factors with a `count` of how many women share it.
Code 9 means "unknown" in every column except `age_group_5_years`.

| column | meaning |
|---|---|
| `age_group_5_years` | age in 5-year bands (1 = 18–29 … 13 = 85+) |
| `race_eth` | race / ethnicity |
| `first_degree_hx` | breast cancer in a first-degree relative |
| `age_menarche` | age at first period |
| `age_first_birth` | age at first birth |
| `BIRADS_breast_density` | BI-RADS breast density |
| `current_hrt` | current hormone replacement therapy |
| `menopaus` | menopausal status |
| `bmi_group` | BMI group |
| `biophx` | previous breast biopsy |
| `breast_cancer_history` | prior breast cancer diagnosis (the target) |

Please acknowledge the BCSC as their data-use terms require if you use the data.
