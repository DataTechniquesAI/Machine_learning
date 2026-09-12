# Machine Learning
# House Price Prediction — Data-to-Model Pipeline

A regression pipeline that predicts residential **SalePrice** from the
[Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques),
covering data cleaning, feature engineering, model comparison,
hyperparameter tuning, and honest evaluation of model performance.

## Problem Statement

**Task:** Predict a house's `SalePrice` (in USD) using its structural,
location, and quality features.

**Framing:** Regression problem — the target variable is continuous.

**Target metric:** MAE (Mean Absolute Error), reported in dollars.
Chosen over RMSE because it's directly interpretable ("the model is off
by ~$X on average") and less dominated by a handful of extreme
high-price outliers — a buyer or seller cares about typical error, not
squared error. R² is reported alongside as a secondary check of
overall fit.

**Why it matters:** An accurate SalePrice estimate helps buyers,
sellers, and agents benchmark a listing price before negotiation.

## Dataset

- Source: Ames Housing dataset (`train.csv`), 1460 rows, 80 feature columns.
- Target: `SalePrice`.
- Not included in this repo — download it from Kaggle and place it in
  the project root as `train.csv` before running the notebook.

## Approach

1. **Exploration** — shape, dtypes, missing values, distributions.
2. **Feature engineering**
   - `MSSubClass` recast from numeric code to categorical (it's a
     house-type code, not an ordinal number).
   - `HouseAge = YrSold - YearBuilt`
   - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
3. **Preprocessing** (inside an sklearn `Pipeline`/`ColumnTransformer`,
   fit only on the training split to avoid leakage):
   - Categorical: most-frequent imputation → one-hot encoding
   - Numerical: median imputation
4. **Models trained**
   - `DummyRegressor` (mean baseline)
   - `LinearRegression`
   - `RandomForestRegressor`, tuned via `GridSearchCV` (5-fold CV) over
     `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
5. **Reproducibility** — fixed `random_state=42` throughout; the full
   fitted pipeline (preprocessing + model) is saved with `joblib`.

## Leakage Risks Avoided

- All imputation statistics are fit only on `X_train` inside the
  pipeline — never on the full dataset before splitting.
- `OneHotEncoder` is fit only on training categories
  (`handle_unknown="ignore"` guards against unseen categories at test time).
- No target-derived features were engineered (nothing computed from `SalePrice`).
- A single train/test split is made before any preprocessing is fit.

## Evaluation

- **Metrics:** MAE and R² on the held-out test set, compared against
  the baseline and across all three models.
- **Diagnostics:** actual-vs-predicted scatter plot, residual plot.
- **Failure analysis:** the 5 largest absolute-error predictions are
  inspected individually to check for patterns (e.g. unusually
  high/low-priced or atypical properties).

## Results

| Model                  |   MAE ($) |   R²   |
|------------------------|----------:|-------:|
| Baseline (mean)        |   62,575.93 |    —   |
| Linear Regression      |   20,481.98 | 0.8721 |
| Random Forest          |   17,542.29 | 0.8911 |
| Random Forest (tuned)  | **17,468.52** | **0.8931** |

The tuned Random Forest gives the lowest error and best fit, cutting
MAE by roughly **72% relative to the baseline** and outperforming
Linear Regression by about **$3,000 (~15%)** in average prediction error.

## Limitations

- Accuracy drops for unusually expensive or inexpensive houses —
  the model underpredicts some high-priced homes and overpredicts
  some lower-priced ones.
- Rare property characteristics may not be well captured by the
  available features.
- Evaluated on a single train/test split; performance may vary with a
  different split or on genuinely new data (e.g. a different market/year).
- Predictions are estimates, not a substitute for a real appraisal or
  the sole basis for a pricing decision.

## Project Structure

```
.
├── assessment_ml.ipynb        # main notebook: EDA → features → models → evaluation
├── train.csv                  # dataset (not included — download from Kaggle)
├── house_price_model.pkl      # saved fitted pipeline (generated after running)
└── README.md
```

## How to Run

1. Clone the repo and install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```
2. Download `train.csv` from Kaggle and place it in the project root.
3. Open `assessment_ml.ipynb` in Jupyter and run all cells top to bottom.
4. The fitted pipeline is saved as `house_price_model.pkl` and can be
   reloaded with:
   ```python
   import joblib
   model = joblib.load("house_price_model.pkl")
   model.predict(new_data)
   ```

## Author

Sumia ([@DataTechniquesAI](https://github.com/DataTechniquesAI))
