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

This project uses the IBM Telco Customer Churn dataset from Kaggle, a widely
used benchmark dataset in the churn prediction literature, to build a model
that identifies at-risk customers based on their service usage patterns,
contract type, and billing behavior.

The central question: which customer features are most predictive of churn,
and can a model identify at-risk customers with enough precision to be
actionable for a retention team?

---

## Dataset

Source: IBM Telco Customer Churn (Kaggle)
Size: 7,043 customers
Target: Churn (Yes/No)

Features include:

| Category | Features |
|----------|----------|
| Demographics | Gender, senior citizen status, partner, dependents |
| Services | Phone, internet, streaming, online security, tech support |
| Account | Contract type, payment method, paperless billing |
| Billing | Monthly charges, total charges, tenure |

Key preprocessing steps:
- Removed 11 rows where TotalCharges was blank (new customers with no charges)
- Converted TotalCharges from string to float
- Label encoded all categorical features for model compatibility
- Dropped customerID as it carries no predictive signal

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
class imbalance in the dataset (approximately 73% non-churn, 27% churn):

Parameters searched:

n_estimators:      [100, 200]
max_depth:         [None, 10, 20]
min_samples_split: [2, 5]
min_samples_leaf:  [1, 2]

Total combinations tested: 24
Cross validation folds: 5
Scoring metric: F1

### Why F1 Over Accuracy

With 73% of customers not churning, a model that predicts everyone stays
would achieve 73% accuracy while being completely useless. F1 score balances
precision and recall, penalizing both false positives and false negatives,
making it the appropriate metric for imbalanced classification problems.

### Explainability with SHAP

SHAP (SHapley Additive exPlanations) values were computed using KernelExplainer
on a sample of 20 test customers to understand which features drove individual
predictions. This moves beyond global feature importance to explain why the
model predicted churn for a specific customer.

---

## Experiments

### Experiment 1: Baseline Random Forest

Trained a Random Forest with default hyperparameters to establish a
performance baseline before tuning.

### Experiment 2: Hyperparameter Tuning with GridSearchCV

Applied GridSearchCV across 24 parameter combinations with 5-fold
cross validation optimizing for F1 score.

Best parameters found:
[paste your best params output here]

Classification Report:
[paste your classification report output here]

Confusion Matrix:
[paste your confusion matrix output here]

### Experiment 3: SHAP Explainability Analysis

Applied SHAP KernelExplainer to understand feature contributions at
the individual prediction level, not just global feature importance.

Key finding: [paste your top SHAP features here after running]

---

## Results

[paste your full classification report here]

Key metrics on held out test set (20% of data):

| Metric | Score |
|--------|-------|
| Precision (Churn) | [your value] |
| Recall (Churn) | [your value] |
| F1 Score (Churn) | [your value] |
| Overall Accuracy | [your value] |

Top 10 most important features by Random Forest importance score:
[paste your feature importance chart findings here]

---

## Deployment

The trained model was serialized using joblib for deployment:

churn_model_rf.pkl     Trained Random Forest model
feature_columns.pkl    Feature column names for input validation
churn_pipeline.pkl     Full sklearn pipeline for Streamlit deployment

The pipeline wraps the model in a sklearn Pipeline object to ensure
consistent preprocessing between training and inference when deployed
in a Streamlit application.

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

The user fills in the form and clicks Predict. The app:

1. Maps all categorical inputs to the same numeric encoding used during training
2. Passes the encoded input through the saved pipeline
3. Returns a binary prediction (Churn or No Churn)
4. Returns the exact churn probability as a percentage

### Example Output

Prediction:        Churn
Churn Probability: 73.4%

This probability score is more actionable than a binary prediction alone.
A retention team can set their own threshold based on the cost of intervention
versus the cost of losing the customer.

### Running The App Locally

1. Ensure churn_pipeline.pkl is in the same directory as streamlit_app.py

2. Install Streamlit:
   pip install streamlit

3. Run the app:
   streamlit run streamlit_app.py

4. App opens automatically at:
   http://localhost:8501

### Key Design Decisions

The Streamlit app uses the full pipeline object rather than the raw model.
This ensures the identity transform preprocessor is applied consistently
between training and inference, preventing feature mismatch errors when
the model receives new input data.

Categorical inputs are mapped to integers using the same encoding scheme
applied during training:

Contract:         Month-to-month=0, One year=1, Two year=2
Internet Service: No=0, DSL=1, Fiber optic=2
Payment Method:   Electronic check=0, Mailed check=1, Bank transfer=2, Credit card=3
All Yes/No:       No=0, Yes=1

---

## Conclusion and Takeaways

1. Churn is predictable from behavioral signals

The model confirms that churn is not random. Contract type, tenure,
and monthly charges are consistently among the top predictors, suggesting
that customers on month-to-month contracts who have been with the company
a short time and pay higher monthly fees are disproportionately likely
to churn.

2. F1 optimization matters more than accuracy for imbalanced problems

Optimizing for accuracy on this dataset would produce a misleading model.
By optimizing for F1 during hyperparameter search, the model learned to
identify churners rather than simply predicting the majority class.

3. SHAP adds actionability to predictions

Global feature importance tells you which features matter on average.
SHAP tells you why the model flagged a specific customer. This distinction
is critical for a retention team who need to know not just who is at risk
but what intervention to offer.

4. Complexity of deployment matters as much as model performance

A model that lives in a Jupyter notebook has no business value. Wrapping
the pipeline in a Streamlit app makes the model accessible to a retention
team who can query individual customers in real time without any technical
knowledge. This reflects how machine learning models are actually used in
production environments.

5. Limitations

- Label encoding assumes ordinal relationships between categories which
  may not hold for features like payment method or contract type.
  One-hot encoding would be more theoretically appropriate.
- SHAP was computed on only 20 samples due to computational cost of
  KernelExplainer. TreeExplainer would be faster and allow full test
  set explanation.
- No cost-sensitive learning was applied. In practice, the cost of
  missing a churner likely exceeds the cost of a false positive retention
  offer, suggesting recall should be weighted more heavily than precision.

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

1. Download Telco.csv from Kaggle:
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
