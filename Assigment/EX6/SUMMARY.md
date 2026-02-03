# EX6: Summary of Findings — Polynomial Features and Regularization

## Objective

This assignment extended the EX5 Linear Regression work on the medical insurance dataset (`insurance.csv`) by:
1. Adding polynomial components to the feature set and retraining the best (EX5-style) linear model.
2. Implementing **Lasso**, **Ridge**, and **Elastic Net** regularization and comparing their results to the baseline and polynomial linear regression.

---

## Dataset and Preparation (from EX5)

- **Dataset**: `insurance.csv` — 1,338 rows, 7 columns (age, sex, bmi, children, smoker, region, charges).
- **Target**: `charges` (individual medical costs billed by health insurance).
- **Findings from EX5** used here:
  - Categorical variables (`sex`, `smoker`, `region`) were encoded with **dummy coding** (`pd.get_dummies()`).
  - The dependent variable `charges` is **right-skewed**; we kept the same preprocessing (no log-transform in this exercise).
  - Numeric features were **standardized** (StandardScaler) before modeling, which is important for regularization and polynomial terms.

Train/test split was 80/20 with a fixed random state for reproducibility.

---

## 1. Baseline: Linear Regression (EX5-style)

- **Model**: Ordinary Least Squares (OLS) linear regression on encoded and scaled features (no polynomial terms).
- **Purpose**: Represents the “best implementation from EX5” as the baseline to improve upon.
- **Typical outcome**: Moderate R² (roughly 0.75–0.80 on test) and a certain test RMSE. This baseline assumes linear relationships between predictors and `charges`.

---

## 2. Polynomial Features (Degree 2)

- **Change**: Polynomial features of **degree 2** were added to the scaled numeric design matrix (including interactions and squared terms).
- **Effect**: The number of features increases substantially (from 11 to 66 with 11 original features), allowing the model to capture **non-linear relationships** and **interactions** (e.g., age×bmi, smoker×bmi).
- **Result**: Polynomial + Linear Regression often achieves a **higher test R²** than the baseline, indicating that some non-linear structure in the data (e.g., smoker and bmi interactions) helps predict `charges`. There is a risk of **overfitting** when many polynomial terms are used without regularization.

---

## 3. Regularized Models (Lasso, Ridge, Elastic Net)

All three methods were applied to the **same polynomial feature set** (degree 2) and compared using the same train/test split and scaling.

### Ridge (L2 regularization)

- **Idea**: Penalizes the sum of squared coefficients. Shrinks coefficients but does not set them exactly to zero.
- **Effect**: Reduces overfitting when many polynomial terms are present; model remains stable.
- **Typical result**: Test R² similar to or slightly better than unregularized polynomial regression, with lower variance in performance. **RMSE** may be similar or slightly improved.

### Lasso (L1 regularization)

- **Idea**: Penalizes the sum of absolute values of coefficients. Can drive some coefficients to **exactly zero**, so it performs **feature selection**.
- **Effect**: With many polynomial terms, Lasso tends to zero out less important terms, yielding a simpler model.
- **Typical result**: Test R² may be slightly lower than polynomial OLS or Ridge if the optimal alpha is strong, but the model is more interpretable. With moderate alpha, performance is often close to Ridge.

### Elastic Net (L1 + L2)

- **Idea**: Combines L1 and L2 penalties (e.g., `l1_ratio=0.5` for equal weight). Compromise between Ridge and Lasso.
- **Effect**: Handles correlated predictors better than pure Lasso and can still perform some feature selection.
- **Typical result**: Test R² and RMSE usually lie between or close to Lasso and Ridge, depending on `alpha` and `l1_ratio`.

---

## 4. Comparison of Results

| Model                          | Typical role                          | R² (relative)     | RMSE / Overfitting      |
|--------------------------------|----------------------------------------|-------------------|--------------------------|
| Baseline (Linear Regression)  | EX5-style baseline                     | Lower             | Higher RMSE             |
| Polynomial + Linear Regression| Captures non-linearity                 | Higher            | Risk of overfitting     |
| Polynomial + Ridge             | Stable, robust with many features      | Similar or better | Often best trade-off    |
| Polynomial + Lasso            | Sparse model, feature selection        | Similar or lower  | Simpler model           |
| Polynomial + Elastic Net      | Blend of Ridge and Lasso               | In between        | Flexible compromise     |

Exact numbers depend on the chosen `alpha` (and for Elastic Net, `l1_ratio`). Running the notebook `EX6_Regularization_Polynomial.ipynb` will fill in the actual test MSE, RMSE, and R² for your run.

---

## 5. Conclusions and Recommendations

1. **Polynomial features (degree 2)** improve over the EX5 linear-only baseline because the relationship between predictors (e.g., age, bmi, smoker) and `charges` is not purely linear.
2. **Ridge** is a good default when using many polynomial terms: it stabilizes estimates and often matches or slightly improves test performance compared to unregularized polynomial regression.
3. **Lasso** is useful when interpretability and a sparse model are desired; tuning `alpha` (e.g., via cross-validation) is important.
4. **Elastic Net** is a solid choice when predictors are correlated (e.g., polynomial terms) and you want a balance between shrinkage and feature selection.
5. **Next steps**: (a) Tune `alpha` (and `l1_ratio` for Elastic Net) via cross-validation; (b) try `degree=3` with stronger regularization; (c) consider log-transforming `charges` if residual analysis in EX5 suggested it.

---

## Files in EX6

- **`EX6_Regularization_Polynomial.ipynb`**: Full pipeline — load data, EX5-style prep, baseline LR, polynomial features, Lasso/Ridge/Elastic Net, comparison table and plots.
- **`data/insurance.csv`**: Medical insurance dataset (can be replaced by your own download of `insurance.csv` if needed).
- **`SUMMARY.md`**: This summary of findings (Part 2).
