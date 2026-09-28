# Data Science Weekend (DSW) 2023: Strategic Customer Retention with Active Learning and LLM-Driven Churn Solutions

> Predict churn among a telecom company's high-value customers on an imbalanced dataset, then generate personalized retention advice with an LLM.

## Problem
A telecom company needs to predict which of its high-value customers will churn. The dataset (7,043 customers) has tenure, location (Jakarta or Bandung), device class, usage of games, music, education and video products, call center and MyApp usage, payment method, monthly purchase, coordinates, predicted CLTV (Customer Lifetime Value, in thousands of IDR) and a binary churn label. The classes are imbalanced (26.5% churn vs 73.5% non-churn), so we compare models on macro F1, reported alongside balanced accuracy.

## Approach
- **Preprocessing:** added a derived `internet_service` flag, ordinal/binary encoding of the categorical columns, and one-hot encoding of `Payment Method`.
- **High-value segment:** kept only customers with `CLTV (Predicted Thou. IDR) > 7000` (1,752 rows), then made a stratified 80/20 train/validation split.
- **EDA:** correlation heatmaps and churn rate per feature. Tenure, device class and paying with pulsa (prepaid phone credit) have the strongest correlation with churn.
- **Model:** CatBoost (default parameters) throughout.
- **Imbalance handling compared:** no resampling (baseline), SMOTE, ADASYN, random undersampling, NearMiss (v2), and **pool-based active learning**.
- **Active learning:** the labeled pool starts as all churners plus 10 non-churners. On each round, CatBoost is retrained and the 50 non-churn samples with predictions closest to 0.5 are moved from the unlabeled pool into the labeled set. The round with the best validation macro F1 is kept and saved as `best_model.pkl`.
- **LLM and app (described in the slides; code in a separate repo):** gpt-3.5-turbo via the OpenAI API generates a likely churn reason and a retention solution for each customer, plus insights on feature-vs-churn tables. A Streamlit dashboard serves predictions and analysis.

## Results
Validation set (351 high-CLTV customers), CatBoost, from `notebooks/churn_prediction_active_learning.ipynb`:

| Method | Macro F1 | Balanced accuracy |
|---|---|---|
| Baseline (no resampling) | 0.657 | 0.636 |
| SMOTE | 0.680 | 0.669 |
| ADASYN | 0.708 | 0.701 |
| Random undersampling | 0.674 | 0.732 |
| NearMiss (v2) | 0.384 | 0.571 |
| **Active learning (best round, 1,050 labeled samples)** | **0.732** | 0.729 |

These match the F1 comparison chart in the presentation.

## Repository structure
```
dsw-2023/
├── README.md
├── notebooks/
│   └── churn_prediction_active_learning.ipynb        # final solution: EDA, preprocessing, CatBoost + active learning vs. resampling methods
├── experiments/
│   ├── catboost_cv_baseline.ipynb                    # early CatBoost 5-fold stratified CV on the full dataset (accuracy 0.793)
│   └── catboost_eda_and_active_learning_draft.ipynb  # EDA, CatBoost CV on F1, and a first (failing) active learning attempt
├── docs/
│   └── dsw_2023_three_outliers_presentation.pdf      # final presentation "PPT DSW 2023 - Three Outliers"
└── data/
    └── telco_customer_churn_adapted_v2.xlsx          # competition dataset (Telco customer churn, adapted)
```

## How to run
- The dataset was provided by the DSW 2023 organizers and is included at `data/telco_customer_churn_adapted_v2.xlsx`. The notebooks read it from `../data/`, so run them from inside their own folder.
- Main dependencies: `pandas` (< 2.0, because the notebook uses `DataFrame.append`), `numpy`, `scikit-learn`, `imbalanced-learn`, `catboost`, `xgboost`, `matplotlib`, `seaborn`, `plotly`, `openpyxl`. `experiments/catboost_cv_baseline.ipynb` also imports `keras`, and `experiments/catboost_eda_and_active_learning_draft.ipynb` trains CatBoost with `task_type="GPU"`.
- The final notebook writes the best active-learning model to `best_model.pkl` in the working directory.

## Links
- Streamlit app and deployment code: https://github.com/edutjie/dsw2023
- Live demo: https://threeoutliers-dsw.streamlit.app/

## Team
[Three Outliers](../README.md#team): [Eduardus Tjitrahardja](https://www.linkedin.com/in/edutjie/), [Ikhlasul Akmal Hanif](https://www.linkedin.com/in/ikhlasul-akmal-h/), [Rahmat Bryan Naufal](https://www.linkedin.com/in/rahmat-bryan-naufal/)
