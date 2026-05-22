# House Price Predictor

## Hypothesis

House sale prices are determined by a combination of structural, locational,
and qualitative features that can be systematically captured and modeled.
If the relationships between these features and sale price are learned
effectively, a machine learning model should be able to predict the sale
price of an unseen house with meaningful accuracy.

A secondary hypothesis: ensemble methods that combine multiple models will
outperform any single model, and Bayesian hyperparameter optimization will
find better parameters than grid search for complex models with large
search spaces.

---

## Background

This project uses the Ames Housing dataset from the Kaggle House Prices
Advanced Regression Techniques competition. The dataset contains 1,460
training observations and 81 features describing residential properties
in Ames, Iowa, sold between 2006 and 2010.

The dataset is significantly more complex than standard housing datasets,
containing a mix of continuous, ordinal, and nominal features, substantial
missing data across 19 columns, and a wide range of property types from
modest single-family homes to large luxury properties.

The target variable, SalePrice, ranges from $34,900 to $755,000 with a
mean of approximately $180,921, suggesting right skew driven by high-value
outliers.

---

## Dataset

Source: Kaggle House Prices Advanced Regression Techniques
Training set: 1,460 houses, 80 features
Test set: 1,459 houses (no sale price, used for competition submission)
Target: SalePrice (USD)

Feature categories:

| Category | Examples |
|----------|---------|
| Location | Neighborhood, zoning, lot configuration |
| Structure | Building type, house style, year built |
| Quality | Overall quality, exterior quality, kitchen quality |
| Size | Lot area, ground living area, basement area, garage area |
| Condition | Overall condition, exterior condition, basement condition |
| Utilities | Heating, electrical, central air |
| Extras | Pool, fence, fireplace, miscellaneous features |

Missing data summary:

| Feature | Missing Count | Interpretation |
|---------|--------------|----------------|
| PoolQC | 1,453 | No pool |
| MiscFeature | 1,406 | No misc feature |
| Alley | 1,369 | No alley access |
| Fence | 1,179 | No fence |
| FireplaceQu | 690 | No fireplace |
| LotFrontage | 259 | Missing measurement |

---

## Methodology

### Preprocessing Pipeline

A sklearn ColumnTransformer was built to handle numerical and categorical
features separately:

Numerical features:
- Missing values imputed using median strategy
- RobustScaler applied to reduce influence of outliers
- RobustScaler chosen over StandardScaler due to significant outliers
  detected across 30 features using IQR method

Categorical features:
- Missing values imputed using most frequent strategy
- OneHotEncoder applied with drop first to avoid multicollinearity
- handle unknown set to ignore for test set categories not seen in training

### Why RobustScaler

IQR outlier analysis revealed significant outliers across 30 features.
EnclosedPorch had 208 outlier values, BsmtFinSF2 had 167, and OverallCond
had 125. RobustScaler uses median and IQR rather than mean and standard
deviation, making it substantially less sensitive to these extreme values.

### Evaluation Metric

Mean Absolute Error (MAE) was used as the primary evaluation metric.
MAE represents the average dollar error in predictions and is more
interpretable than RMSE for a business audience. A MAE of $16,000 on
a $180,000 mean sale price represents approximately 8.8% average error.

---

## Experiments

### Experiment 1: Baseline Model Comparison

Seven models were trained with default hyperparameters to establish
baselines before tuning:

| Model | MAE |
|-------|-----|
| Linear Regression | $19,931 |
| Lasso | $19,839 |
| Ridge | $19,720 |
| ElasticNet | $20,761 |
| Bagging Regressor | $19,566 |
| Random Forest | $17,576 |
| XGBoost | $18,298 |

Random Forest and XGBoost were the clear top performers, with tree-based
ensemble methods outperforming all linear models by over $2,000 MAE.
This suggests nonlinear relationships and feature interactions are
important in this dataset that linear models cannot capture.

---

### Experiment 2: Random Forest Hyperparameter Tuning with Optuna

Optuna Bayesian optimization was used rather than GridSearchCV for two
reasons. The search space for Random Forest is too large for exhaustive
grid search to be practical, and Bayesian optimization learns from prior
trials to focus the search on promising regions of the parameter space.

Parameters searched:

n_estimators:      50 to 200
max_depth:         5 to 30
min_samples_split: 2 to 10
min_samples_leaf:  1 to 4
max_features:      sqrt or log2

Trials run: 50
Optimization metric: negative MAE via 5-fold cross validation

Best parameters found:
n_estimators:      157
max_depth:         22
min_samples_split: 2
min_samples_leaf:  1
max_features:      sqrt

Validation Set MAE: $17,739

Improvement over default Random Forest: $164 (marginal)

---

### Experiment 3: XGBoost Hyperparameter Tuning with Optuna

XGBoost has a significantly larger hyperparameter space than Random Forest,
making Bayesian optimization even more appropriate. GPU acceleration was
used to reduce computation time across 150 trials.

Parameters searched:

n_estimators:      50 to 200
learning_rate:     0.001 to 0.1 (log scale)
max_depth:         1 to 10
min_child_weight:  1 to 20
subsample:         0.05 to 1.0 (log scale)
colsample_bytree:  0.05 to 1.0 (log scale)
gamma:             1e-8 to 1.0 (log scale)
reg_alpha:         1e-8 to 1.0 (log scale)
reg_lambda:        1e-8 to 1.0 (log scale)

Trials run: 150
Optimization metric: negative MAE via 5-fold cross validation

Best parameters found:
n_estimators:     190
learning_rate:    0.0533
max_depth:        4
min_child_weight: 3
subsample:        0.773
colsample_bytree: 0.546
gamma:            9.73e-08
reg_alpha:        2.29e-05
reg_lambda:       0.0192

Validation Set MAE: $16,096

Improvement over default XGBoost: $2,202

---

### Experiment 4: Stacking Ensemble

A StackingRegressor was built combining the tuned Random Forest and
tuned XGBoost as base learners, with Ridge regression as the meta-learner.

The meta-learner learns how to best combine the predictions of the two
base models, potentially correcting systematic errors that each model
makes individually.

Result: MAE $16,147

The stacking ensemble performed worse than the best individual XGBoost
model, suggesting the two base models were making correlated errors that
the meta-learner could not effectively correct.

---

### Experiment 5: Voting Ensemble

A VotingRegressor was built combining the same two tuned models using
simple averaging of predictions rather than a learned combination.

Result: MAE $16,302

Simple averaging also underperformed the best individual XGBoost model,
confirming that ensemble combination did not add value over the best
single model in this case.

---

## Results

| Model | MAE | Notes |
|-------|-----|-------|
| Linear Regression (baseline) | $19,931 | Default params |
| Ridge (baseline) | $19,720 | Default params |
| Random Forest (baseline) | $17,576 | Default params |
| XGBoost (baseline) | $18,298 | Default params |
| Random Forest (tuned) | $17,739 | 50 Optuna trials |
| XGBoost (tuned) | $16,096 | 150 Optuna trials, GPU |
| Stacking RF + XGB | $16,147 | Ridge meta-learner |
| Voting RF + XGB | $16,302 | Simple averaging |

Best model: Tuned XGBoost at MAE $16,096
Improvement over best baseline: $2,202 (12% reduction in error)
Improvement over linear baseline: $3,835 (19% reduction in error)

---

## Conclusion and Takeaways

1. The hypothesis was supported

House prices can be predicted with meaningful accuracy from structural
and locational features. The best model achieves approximately 8.8%
average error relative to mean sale price, which is competitive with
published results on this dataset.

2. Nonlinearity matters significantly in housing price prediction

The consistent outperformance of tree-based models over linear models
by over $2,000 MAE confirms that house prices have important nonlinear
relationships and feature interactions that linear models cannot capture.
Overall quality interacting with neighborhood, for example, is likely
more predictive than either feature alone.

3. Bayesian optimization adds more value for XGBoost than Random Forest

Tuning XGBoost with Optuna reduced MAE by $2,202 while tuning Random
Forest only reduced MAE by $164. This reflects XGBoost having a larger
and more sensitive hyperparameter space where learning rate, subsample,
and regularization parameters interact in ways that random or grid search
would explore less efficiently.

4. Ensemble methods did not outperform the best individual model

Both stacking and voting ensembles underperformed the tuned XGBoost
alone. This is a well documented phenomenon when the base models are
highly correlated, meaning they make similar predictions on similar
houses, the ensemble cannot diversify errors effectively. A more
heterogeneous ensemble including linear models might have produced
better results.

5. Missing data interpretation is a modeling decision

The majority of missing values in this dataset represent absence rather
than unknown values. A house missing PoolQC almost certainly has no pool.
Imputing with most frequent or median rather than a dedicated absence
indicator may have lost some signal. Future work would encode absence
explicitly as a binary feature.

6. Limitations

- No log transformation of SalePrice despite right skew which may
  have improved model performance on high value properties
- Outlier detection identified many outliers but none were removed
  or capped which may have degraded model performance
- No feature engineering beyond raw features such as total square
  footage, house age at sale, or years since remodel
- Test set predictions cannot be evaluated without Kaggle submission

---

## Tech Stack

Python           Data processing and modeling
pandas           Data loading and exploration
numpy            Numerical operations and outlier detection
scikit-learn     Preprocessing pipeline, model training, evaluation
xgboost          Gradient boosting with GPU acceleration
optuna           Bayesian hyperparameter optimization
matplotlib       Visualization
seaborn          Distribution plots and bar charts

---

## How To Run

1. Download train.csv and test.csv from Kaggle:
   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

2. Install dependencies:
   pip install pandas numpy scikit-learn xgboost optuna matplotlib seaborn

3. Run the notebook in Google Colab or Jupyter

Note: XGBoost GPU acceleration requires a CUDA-compatible GPU.
Remove device=gpu from XGBRegressor to run on CPU.
