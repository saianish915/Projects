# Telco Customer Churn Predictor

## Hypothesis

Customer churn in the telecommunications industry is not random. Customers
who leave are likely exhibiting measurable behavioral and contractual signals
prior to churning. If these signals can be identified and quantified, a
machine learning model should be able to predict which customers are at risk
of churning before they leave, allowing the business to intervene proactively.

---

## Background

Customer churn is one of the most costly problems in the telecommunications
industry. Acquiring a new customer costs five times more than retaining an
existing one. Despite this, most telecom companies respond to churn reactively
rather than proactively.

This project uses the IBM Telco Customer Churn dataset from Kaggle to build
a model that identifies at-risk customers based on their service usage
patterns, contract type, and billing behavior. The dataset was then deployed
as an interactive Streamlit web application allowing non-technical users to
assess churn risk for individual customers in real time.

The central question: which customer features are most predictive of churn,
and can a model identify at-risk customers with enough precision to be
actionable for a retention team?

---

## Dataset

Source: IBM Telco Customer Churn (Kaggle)
Size: 7,043 customers
Target: Churn (Yes = 1, No = 0)
Class distribution: 73% non-churn, 27% churn

Features used after cleaning:

| Category | Features |
|----------|----------|
| Account | Contract type, paperless billing, payment method |
| Services | Phone, multiple lines, internet, security, backup, protection, tech support, streaming |
| Billing | Monthly charges, total charges, tenure |
| Demographics | Gender, senior citizen status, partner, dependents |

### Data Leakage Investigation

The raw dataset contained several columns that were identified and removed
as leakage before modeling:

| Column | Reason Removed |
|--------|----------------|
| Customer Status | Directly encodes whether customer churned, stayed, or joined |
| Satisfaction Score | Post-churn survey collected after the outcome is known |
| Churn Score | Derived metric calculated from the churn outcome |
| Churn Category | Describes why the customer churned |
| Churn Reason | Describes why the customer churned |
| CLTV | Customer lifetime value derived from churn outcome |

Including any of these columns produced 100% accuracy, confirming that
they encode the answer rather than predict it. A model trained with leakage
is completely useless in production because the leaked features would not
be available at the time of prediction.

After removing leakage columns, the model trained on 17 genuine predictive
features and produced realistic results of 82% accuracy and 0.62 F1 score
on the churn class.

### Key Preprocessing Steps
- Dropped all leakage columns and geographic/administrative fields
- Converted TotalCharges from string to float
- Dropped rows with missing TotalCharges values
- Label encoded all categorical features

---

## Methodology

### Model Selection

Random Forest Classifier was selected over logistic regression for the
following reasons:

- Captures nonlinear relationships between features and churn
- Naturally handles mixed feature types without scaling
- Provides feature importance scores for interpretability
- Robust to outliers and multicollinearity

### Hyperparameter Tuning

GridSearchCV with 5-fold cross validation was used to find optimal
hyperparameters, optimizing for F1 score rather than accuracy due to
class imbalance (73% non-churn, 27% churn):

Parameters searched:

n_estimators:      [100, 200]
max_depth:         [None, 10, 20]
min_samples_split: [2, 5]
min_samples_leaf:  [1, 2]

Total combinations tested: 24
Cross validation folds: 5
Scoring metric: F1

Best parameters found:
n_estimators:      200
max_depth:         20
min_samples_split: 2
min_samples_leaf:  2

### Why F1 Over Accuracy

With 73% of customers not churning, a model that predicts everyone stays
would achieve 73% accuracy while being completely useless. F1 score balances
precision and recall, penalizing both false positives and false negatives,
making it the appropriate metric for imbalanced classification problems.

### Explainability with SHAP

SHAP (SHapley Additive exPlanations) values were computed using
KernelExplainer on a sample of 20 test customers to understand which
features drove individual predictions. This moves beyond global feature
importance to explain why the model flagged a specific customer as at risk.

---

## Experiments

### Experiment 1: Data Leakage Detection and Removal

The first run produced 100% accuracy across all metrics with a perfect
confusion matrix of zero errors. This immediately indicated data leakage
rather than genuine model performance.

Investigation revealed that Customer Status and Satisfaction Score remained
in the feature set despite an initial drop attempt. Customer Status directly
encodes the churn outcome (Churned, Stayed, Joined) and Satisfaction Score
is a post-churn survey result collected after the customer had already left
or stayed.

After removing all leakage columns accuracy dropped to 82% and F1 for the
churn class dropped to 0.62, representing genuine predictive performance.

This experiment demonstrates a critical principle in machine learning:
perfect or near-perfect results on real-world classification problems are
almost always a sign of a problem rather than a success.

### Experiment 2: Hyperparameter Tuning with GridSearchCV

Applied GridSearchCV across 24 parameter combinations with 5-fold cross
validation optimizing for F1 score.

Best parameters found:
max_depth: 20, min_samples_leaf: 2, min_samples_split: 2, n_estimators: 200

### Experiment 3: SHAP Explainability Analysis

Applied SHAP KernelExplainer on 20 test customers to understand individual
prediction drivers.

Top feature by SHAP importance: Contract type

Contract type was the single strongest predictor of churn, consistent with
business intuition. Month-to-month customers face no financial penalty for
leaving and churn at significantly higher rates than customers on one-year
or two-year contracts.

---

## Results

Model: Random Forest Classifier
Features used: 17
Accuracy: 82%

Classification Report:

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| No Churn (0) | 0.85 | 0.91 | 0.88 | 1,035 |
| Churn (1) | 0.70 | 0.56 | 0.62 | 374 |
| Accuracy | | | 0.82 | 1,409 |
| Macro Avg | 0.78 | 0.74 | 0.75 | 1,409 |
| Weighted Avg | 0.81 | 0.82 | 0.81 | 1,409 |

Confusion Matrix:

| | Predicted No Churn | Predicted Churn |
|--|-------------------|-----------------|
| Actual No Churn | 947 | 88 |
| Actual Churn | 166 | 208 |

In business terms:
- 947 customers correctly identified as staying
- 208 customers correctly identified as churning
- 88 false alarms (flagged as churn risk but stayed)
- 166 missed churners (predicted to stay but actually left)

---

## Conclusion and Takeaways

1. Data leakage is the most dangerous failure mode in ML

The most important finding in this project was not the final model
performance but the detection and removal of data leakage. A 100% accurate
model that relies on post-outcome data is not a model at all. It is a
lookup table. Identifying and removing leakage columns before reporting
results is a fundamental requirement of honest machine learning practice.

2. Contract type is the strongest predictor of churn

SHAP analysis confirmed that contract type drives churn predictions more
than any other feature. Month-to-month customers churn at dramatically
higher rates than customers on annual or biannual contracts. This finding
directly translates to a business intervention: offering contract lock-in
incentives to high-risk month-to-month customers is the most actionable
retention strategy this model supports.

3. Recall on the churn class is the critical metric

The model catches 56% of actual churners (recall of 0.56) and misses 44%.
For a retention team operating on a fixed budget, recall matters more than
precision because the cost of missing a churner who leaves far exceeds the
cost of a false alarm retention offer. Future work would use a lower
classification threshold to increase recall at the cost of some precision.

4. F1 optimization matters more than accuracy for imbalanced problems

Optimizing for accuracy on this dataset would produce a misleading model.
By optimizing for F1 during hyperparameter search the model learned to
identify churners rather than simply predicting the majority class.

5. Deployment bridges the gap between model and business user

A model that lives in a Jupyter notebook has no business value. Wrapping
the pipeline in a Streamlit app makes the model accessible to a retention
team who can query individual customers in real time without any technical
knowledge. This reflects how machine learning models are actually used in
production environments.

6. Limitations

- Label encoding assumes ordinal relationships between categories which
  may not hold for features like payment method or contract type.
  One-hot encoding would be more theoretically appropriate.
- SHAP was computed on only 20 samples due to computational cost of
  KernelExplainer. TreeExplainer would be faster and allow full test
  set explanation.
- Recall of 0.56 on the churn class means 44% of churners are missed.
  Lowering the classification threshold from 0.5 to 0.3 would likely
  improve recall significantly at the cost of more false alarms.
- No cost-sensitive learning was applied. In practice the cost of missing
  a churner likely exceeds the cost of a false positive retention offer,
  suggesting recall should be weighted more heavily than precision.

---

## Streamlit Web Application

The model was deployed as an interactive web application using Streamlit,
allowing non-technical users to predict churn risk for individual customers
without writing any code.

### How It Works

The app takes 19 customer inputs through a simple form interface:

| Input Category | Fields |
|----------------|--------|
| Demographics | Gender, senior citizen status, partner, dependents |
| Services | Phone, multiple lines, internet, security, backup, protection, tech support, streaming |
| Account | Contract type, paperless billing, payment method |
| Billing | Monthly charges, total charges, tenure |

The user fills in the form and clicks Predict. The app returns a binary
prediction (Churn or No Churn) and the exact churn probability as a
percentage.

### Example Output

Prediction:        Churn
Churn Probability: 73.4%

The probability score is more actionable than a binary prediction alone.
A retention team can set their own threshold based on the cost of
intervention versus the cost of losing the customer.

### Running The App Locally

1. Ensure churn_pipeline.pkl is in the same directory as streamlit_app.py

2. Install Streamlit:
   pip install streamlit

3. Run the app:
   streamlit run streamlit_app.py

4. App opens automatically at:
   http://localhost:8501

---

## Tech Stack

Python           Data processing and modeling
pandas           Data cleaning and feature engineering
scikit-learn     Random Forest, GridSearchCV, train test split, metrics
SHAP             Model explainability and feature attribution
matplotlib       Visualization
seaborn          Feature importance plots
joblib           Model serialization for deployment
Streamlit        Interactive web application for model deployment

---

## How To Run

### Model Training

1. Download telco.csv from Kaggle:
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn

2. Install dependencies:
   pip install pandas numpy scikit-learn shap matplotlib seaborn joblib streamlit

3. Run the training script:
   python churnprobability.py

4. Output files generated:
   churn_model_rf.pkl
   feature_columns.pkl
   churn_pipeline.pkl

### Streamlit App

1. Ensure churn_pipeline.pkl is in the same directory as streamlit_app.py

2. Run the app:
   streamlit run streamlit_app.py

3. App opens automatically at:
   http://localhost:8501
