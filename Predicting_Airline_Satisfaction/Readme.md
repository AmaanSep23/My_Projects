# Airline Passenger Satisfaction Prediction

Binary classification project built for the Kaggle Playground Series S6E10 competition. The model predicts whether an airline passenger will be satisfied, using flight details, passenger profile and service ratings.

- **Competition:** [Playground Series S6E10 - Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)
- **Task:** Supervised binary classification
- **Metric:** ROC-AUC (submission is a probability between 0 and 1 per passenger)
- **Best validation ROC-AUC:** **0.9561** (Random Forest)

## Problem

Airlines want to know which services drive satisfaction and where to improve. Target `satisfaction`: 1 = satisfied, 0 = neutral or dissatisfied.

## Dataset

| File | Rows | Purpose |
|---|---|---|
| `train.csv` | 699,635 | Features + target, used for training and validation |
| `test.csv` | 299,844 | Features only, used for final predictions |
| `sample_submission.csv` | 299,844 | Required submission format (`id`, `satisfaction`) |

- 21 input features: 17 numerical (age, flight distance, 14 service ratings, departure and arrival delay) and 4 categorical (`Gender`, `Customer Type`, `Type of Travel`, `Class`).
- Target is mildly imbalanced: 55.6% not satisfied, 44.4% satisfied.
- Data is not included in this repo. Download it from the competition page.

## Approach

1. **EDA**
   - Checked missing values (only `Arrival Delay in Minutes`, 292 rows) and duplicates (none).
   - Class is a strong signal: 72.5% of Business passengers are satisfied vs 16.8% in Economy.
   - Gender has almost no effect (43.8% vs 44.9%).
   - Satisfied passengers are older on average (42.1 vs 36.5 years).
   - Correlation heatmap: `Online boarding`, `Inflight entertainment`, `Seat comfort` and `Flight Distance` stand out.
2. **Preprocessing**
   - Filled missing `Arrival Delay in Minutes` with the median.
   - Converted target to 0/1 and dropped `id` from the features.
   - One-hot encoded categoricals (`drop_first=True`), giving 22 final features.
   - 80/20 train/validation split (`random_state=42`).
   - Applied the same steps to the test set and aligned its columns to the training columns.
3. **Modeling**
   - Baseline: Logistic Regression with `StandardScaler` in a pipeline.
   - Final: Random Forest (300 trees, `min_samples_leaf=2`).
4. **Submission:** retrained the Random Forest on the full training data and predicted probabilities for the test set.

## Results

| Model | Validation Accuracy | Validation ROC-AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.8776 | 0.9214 |
| Random Forest | - | **0.9561** |

The Random Forest improves ROC-AUC by about 0.035 over the baseline.

## Tech Stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn. Built in Google Colab.

## Limitations and Next Steps

- Only two models tried, with no hyperparameter tuning.
- No cross-validation; results come from a single 80/20 split.
- Try gradient boosting (XGBoost, LightGBM, CatBoost) and tune with cross-validation.
- Add feature engineering (for example, average service rating, total delay).
- Convert the notebook into modular `src/` scripts (preprocess, train, predict).
