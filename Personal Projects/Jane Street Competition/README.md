# Jane Street Real-Time Market Data Forecasting

## Hypothesis

Financial market returns contain predictable signal that can be extracted
from anonymized feature data representing real market conditions. If a
gradient boosting model is trained on a sufficiently large dataset of
historical market observations with proper time-ordered validation, it
should produce forecasts that outperform a naive baseline and generalize
to unseen future market conditions.

---

## Background

This project is an entry into the Jane Street Real-Time Market Data
Forecasting competition on Kaggle, one of the most prestigious quantitative
finance competitions available to the public. Jane Street is a global
quantitative trading firm known for hiring the strongest mathematical and
computational talent in the world.

The competition requires participants to forecast a financial target
(responder_6) across thousands of financial instruments using anonymized
feature data representing real market conditions. The challenge is
structured as a real-time inference problem where predictions must be
generated sequentially, simulating live trading conditions.

This project is significant not just as a machine learning exercise but
as an engagement with the same class of problem that professional
quantitative researchers at major trading firms work on daily.

---

## Dataset

Source: Jane Street Real-Time Market Data Forecasting (Kaggle)
Size: 47,127,338 rows, 103 columns after merging lagged features
Training observations used: 8,000,000 (sampled due to memory constraints)
Validation observations: 752,136 (last 20 trading dates)

Data structure:

| File | Description |
|------|-------------|
| train.parquet | Historical market observations with features and targets |
| lags.parquet | Lagged responder values from previous time periods |
| test.parquet | Live inference data for competition submission |
| features.csv | Feature metadata and descriptions |
| responders.csv | Target variable descriptions |

Features:

| Type | Count | Description |
|------|-------|-------------|
| feature_00 to feature_78 | 79 | Anonymized market features |
| responder_0_lag to responder_8_lag | 9 | Lagged target values from prior periods |

Target: responder_6 (anonymized financial return metric)

Evaluation metric: Weighted RMSE where each observation is weighted
by a provided weight column reflecting the economic significance of
each prediction.

---

## Methodology

### Data Pipeline

The raw data was stored in Parquet format, a columnar storage format
optimized for large-scale analytical workloads. Two separate Parquet
datasets were merged:

1. train.parquet was loaded with PyArrow
2. lags.parquet was loaded with FastParquet due to compatibility requirements
3. The two datasets were merged on date_id and symbol_id to attach
   lagged responder values to each training row

Final merged shape: 47,127,338 rows by 103 columns

### Memory Management

At 47 million rows the full dataset exceeds the available RAM on a standard
Colab instance. Three strategies were used to manage memory:

- Features and lag columns were cast to float32 (halving memory vs float64)
- The original train DataFrame was deleted after extraction with gc.collect()
- Training was capped at 8,000,000 rows via random sampling

### Time-Ordered Validation Split

A critical methodological decision was using a time-ordered validation
split rather than random splitting. The last 20 trading dates were held
out as validation data.

Random splitting would allow the model to train on future data and
validate on past data, producing optimistically biased results that
would not reflect true out-of-sample performance. In financial forecasting
all validation must strictly respect temporal ordering to simulate real
trading conditions where the future is never available at training time.

### Model Selection

LightGBM (Light Gradient Boosting Machine) was selected for the following
reasons:

- Handles large datasets efficiently with histogram-based tree learning
- Supports sample weights natively for weighted RMSE optimization
- Early stopping prevents overfitting without manual epoch tuning
- Significantly faster than XGBoost on datasets of this scale
- Industry standard for tabular financial data at major quant firms

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | rmse | Matches competition metric |
| learning_rate | 0.05 | Conservative to avoid overfitting |
| num_leaves | 127 | Moderate tree complexity |
| min_data_in_leaf | 256 | Prevents overfitting on sparse leaves |
| feature_fraction | 0.85 | Feature subsampling for regularization |
| bagging_fraction | 0.85 | Row subsampling for regularization |
| n_estimators | 10,000 | Upper bound with early stopping |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 0.1 | L2 regularization |
| early_stopping | 300 | Stop if no improvement for 300 rounds |

### Early Stopping Result

Training stopped at round 295 out of 10,000 maximum estimators.
The model converged quickly, suggesting the feature set has moderate
but real predictive power and that additional trees would overfit
rather than generalize.

---

## Experiments

### Experiment 1: Data Merging and Lag Feature Construction

The first challenge was merging the lagged responder data onto the
training set. lags.parquet required FastParquet rather than PyArrow
due to schema compatibility issues. After merging, dtype normalization
was required to ensure the join keys (date_id and symbol_id) matched
exactly between the two datasets.

This produced 9 additional lag features representing the previous
values of responders 0 through 8, giving the model temporal context
about how the target and related metrics have been behaving for each
symbol.

### Experiment 2: Memory-Constrained Training at Scale

Training on 47 million rows exceeded available RAM. The solution was:

1. Cast all float64 columns to float32 to halve memory usage
2. Delete the full merged DataFrame immediately after extracting
   the required columns
3. Cap training at 8 million rows via random sampling

The 8 million row cap represents approximately 17% of available training
data. Increasing this cap with more RAM or a larger compute instance
would likely improve model performance.

### Experiment 3: Time-Ordered LightGBM with Early Stopping

LightGBM was trained with weighted RMSE as the objective, matching the
competition scoring metric exactly. Early stopping on the held-out
validation set prevented overfitting.

Training results:

Round 200: Validation RMSE 0.720741
Round 295: Validation RMSE 0.720678 (best)
Round 400: Validation RMSE 0.720797 (degrading)

Early stopping triggered at round 295 with best validation RMSE of
0.720678. The improvement from round 200 to 295 is marginal (0.000063),
suggesting the model approached its performance ceiling quickly given
the 8 million row training cap.

---

## Results

Model: LightGBM Regressor
Training rows: 8,000,000 (sampled)
Validation rows: 752,136 (last 20 trading dates)
Best iteration: 295 of 10,000
Validation weighted RMSE: 0.720678

The validation weighted RMSE of 0.720678 represents the model's average
prediction error weighted by the economic significance of each observation.
In financial forecasting, marginal improvements in weighted RMSE translate
directly to improved trading performance at scale.

---

## Conclusion and Takeaways

1. Temporal structure is the most important methodological constraint

The single most critical decision in this project was using a time-ordered
validation split. Random splitting in time-series financial data produces
look-ahead bias, where the model trains on future observations and
validates on past ones. This would produce artificially strong validation
scores that collapse immediately in live trading. Every design decision
in financial ML must respect the direction of time.

2. Scale requires engineering discipline as much as modeling skill

At 47 million rows the bottleneck is not model selection but memory
management, data types, and pipeline efficiency. Casting to float32,
deleting intermediate DataFrames, and using columnar Parquet storage
were all necessary to run this experiment on a standard compute instance.
Working at this scale requires thinking like a data engineer as much as
a data scientist.

3. Early stopping revealed fast convergence

The model stopped at round 295 out of 10,000, improving only marginally
between rounds 200 and 295 before degrading. This suggests that with
only 8 million of the available 47 million training rows, the model
reached its information ceiling quickly. The primary lever for improving
performance would be increasing the training data volume rather than
tuning hyperparameters.

4. Lag features provide temporal context

Merging lagged responder values from prior trading periods gave the model
awareness of recent target behavior for each symbol. These 9 lag features
represent the model's memory of recent market conditions and are a standard
technique in financial time series modeling.

5. LightGBM is the appropriate tool at this scale

LightGBM's histogram-based tree algorithm processes large datasets
significantly faster than XGBoost or Random Forest while producing
competitive accuracy. Its native support for sample weights allowed
the training objective to match the competition evaluation metric exactly,
ensuring the model optimizes for what is actually being scored.

6. Limitations

- Training was capped at 8 million of 47 million available rows due
  to memory constraints, likely leaving significant performance on the table
- Hyperparameters were not tuned via cross validation due to computational
  cost at this scale
- No feature engineering was performed beyond lag merging
- The competition requires real-time inference via a provided API which
  was not fully implemented in this notebook
- A RMSE of 0.720678 reflects genuine but modest predictive signal,
  consistent with the difficulty of financial forecasting

---

## Tech Stack

Python           Data processing and modeling
pandas           Data loading, merging, and manipulation
numpy            Numerical operations and weighted RMSE calculation
pyarrow          Parquet reading for training data
fastparquet      Parquet reading for lag data
lightgbm         Gradient boosting model
scikit-learn     Train test split utilities and metrics
joblib           Model serialization
gc               Memory management via garbage collection

---

## How To Run

1. Accept the competition rules and download data from Kaggle:
   https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting

2. Place kaggle.json in your environment and authenticate:
   kaggle competitions download -c jane-street-real-time-market-data-forecasting

3. Install dependencies:
   pip install lightgbm pyarrow fastparquet pandas numpy scikit-learn joblib

4. Run the notebook in Google Colab with a high RAM runtime
   (recommended: 25GB+ RAM)

5. Output file generated:
   lgbm_forecast.pkl

Note: The full dataset is 11.5GB. A stable internet connection and
sufficient disk space are required before running.
