# Covid-19 Mortality Predictor

## Hypothesis

Covid-19 mortality is not random. Patients who die from the virus are
likely to share measurable clinical and demographic characteristics that
distinguish them from patients who survive. If these characteristics can
be identified and quantified, a predictive model should be able to assess
mortality risk at the point of admission based on patient data alone.

---

## Background

During the Covid-19 pandemic, healthcare systems faced extreme pressure to
allocate limited resources including ICU beds, ventilators, and medical
staff to the patients most likely to need them. Early identification of
high-risk patients could improve survival rates by ensuring critical
interventions reach those who need them before their condition deteriorates.

This project uses a large-scale Covid-19 patient dataset sourced from
Kaggle, originally compiled by the Mexican government, to build a binary
classifier that predicts whether a patient will die based on their
comorbidities, demographics, and clinical presentation at admission.

The central question: which patient characteristics are most strongly
associated with Covid-19 mortality, and can a model identify high-risk
patients accurately enough to be clinically useful?

---

## Dataset

Source: Covid-19 Patient Data (Kaggle, Mexican Government)
Target: died_or_not (1 = died, 0 = survived)

Raw features included:

| Category | Features |
|----------|----------|
| Demographics | Sex, age |
| Comorbidities | Diabetes, COPD, asthma, hypertension, obesity, cardiovascular disease, renal disease, immunosuppression, other disease |
| Clinical | Pneumonia, intubation, ICU admission, pregnancy |
| Administrative | Patient type, medical unit, USMER classification |

### Data Encoding

The raw dataset used a non-standard encoding where:
- 2 represented No for binary features
- 97 and 99 represented missing or unknown values
- CLASIFFICATION_FINAL above 3 represented non-Covid cases

All binary features were recoded to standard 0/1 encoding:
- 2 → 0 (No becomes 0, Yes stays 1)
- 97 and 99 → 0 (missing values treated as absence of condition)
- CLASIFFICATION_FINAL > 3 → 0 (non-confirmed Covid cases excluded)

### Feature Engineering

Seven new binary features were derived from existing columns:

| New Feature | Definition |
|-------------|------------|
| male_or_not | 1 if patient is male |
| returned_home_or_not | 1 if patient was sent home rather than hospitalized |
| died_or_not | 1 if DATE_DIED is not 9999-99-99 (target variable) |
| heart_or_not | 1 if patient has cardiovascular disease |
| blood_vessel_or_not | Inverse of heart_or_not |
| elder_or_not | 1 if age >= 65 |
| youth_or_not | 1 if age <= 18 |

Original columns SEX, AGE, DATE_DIED, CARDIOVASCULAR, PATIENT_TYPE, and
MEDICAL_UNIT were dropped after feature engineering as their information
was captured in the derived features. Rows with remaining NaN values were
dropped before modeling.

---

## Methodology

### Model Selection

Linear Regression was used as the baseline model for binary mortality
prediction. While logistic regression or tree-based classifiers would be
more theoretically appropriate for binary classification, linear regression
serves as an interpretable baseline that reveals the linear relationship
between each feature and mortality probability.

### Correlation Analysis

A triangular correlation heatmap was generated across all features to
identify which features correlate most strongly with mortality and to
detect multicollinearity between predictors.

### Train Test Split

Data was split 80/20 with a fixed random state of 42 to ensure
reproducibility across runs.

---

## Experiments

### Experiment 1: Data Cleaning and Encoding

The raw dataset required significant preprocessing before modeling was
possible. The non-standard encoding system (2=No, 97/99=missing) meant
that feeding raw data into any model would produce meaningless results.

Key decisions made:
- Missing values coded as 97 or 99 were treated as absence of condition
  rather than imputed. This reflects clinical reality where unknown
  comorbidity status at admission is functionally equivalent to no
  known comorbidity.
- Non-confirmed Covid cases (CLASIFFICATION_FINAL > 3) were recoded to 0
  to focus the model on confirmed Covid-19 patients only.
- Remaining NaN values were dropped after initial recoding as they
  represented a small proportion of records and could not be reliably
  imputed.

### Experiment 2: Feature Engineering

Rather than using raw demographic columns directly, new binary indicator
features were created to capture clinically meaningful thresholds:

- Elder status (age >= 65) rather than raw age, reflecting the known
  clinical threshold for elevated Covid-19 risk
- Youth status (age <= 18) to capture the opposing end of the age spectrum
- Mortality as a derived binary rather than using the raw date column

### Experiment 3: Correlation Heatmap Analysis

A triangular correlation heatmap was generated to understand feature
relationships before modeling.

Key correlations with died_or_not:

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| returned_home_or_not | -0.52 | Strongest predictor - sent home means survived |
| INTUBED | +0.50 | Intubation strongly associated with death |
| PNEUMONIA | +0.47 | Pneumonia is a major mortality risk factor |
| ICU | +0.20 | ICU admission indicates severe cases |
| elder_or_not | +0.14 | Age 65+ associated with higher mortality |
| youth_or_not | -0.014 | Young patients have lower mortality risk |

Notable multicollinearity between comorbidities:

| Feature Pair | Correlation |
|--------------|-------------|
| COPD and ASTHMA | 0.91 |
| COPD and INMSUPR | 0.85 |
| COPD and HIPERTENSION | 0.84 |
| ASTHMA and INMSUPR | 0.87 |

These comorbidities are highly correlated with each other, consistent
with clinical reality where patients frequently present with multiple
concurrent conditions. This multicollinearity may destabilize linear
regression coefficient estimates.

### Experiment 4: Linear Regression Baseline

A Linear Regression model was trained on the processed feature set.

R-squared on test set: 0.415

The model explains 41.5% of the variance in Covid-19 mortality outcomes.
For a binary classification problem modeled with linear regression this
is a reasonable baseline result, though a proper classification model
evaluated with F1 score and AUC-ROC would be more meaningful.

---

## Results

Model: Linear Regression
R-squared: 0.415

The model explains 41.5% of the variance in patient mortality outcomes.

Key findings from correlation analysis:

Strongest predictors of mortality:
- Intubation (+0.50) and pneumonia (+0.47) are the strongest clinical
  signals, reflecting that patients who required mechanical ventilation
  or developed pneumonia were far more likely to die
- Being sent home (-0.52) is the strongest single predictor of survival,
  which is intuitive as patients stable enough to be discharged are by
  definition not in critical condition
- Elder status (+0.14) confirms the well-documented age-based mortality
  risk for Covid-19

Multicollinearity finding:
- Comorbidity features are highly intercorrelated (COPD, asthma,
  hypertension, immunosuppression all correlate above 0.80 with each
  other), suggesting that these conditions tend to cluster together in
  high-risk patients rather than acting as independent risk factors

---

## Conclusion and Takeaways

1. The hypothesis was supported

Patient characteristics available at admission are meaningfully
predictive of Covid-19 mortality. The correlation analysis confirms
that intubation, pneumonia, and elder status are among the strongest
predictors, consistent with clinical literature on Covid-19 risk factors.
An R-squared of 0.415 indicates the model captures real signal rather
than noise.

2. Clinical severity indicators dominate over comorbidities

The strongest predictors are not the comorbidities such as diabetes or
obesity but rather the clinical severity indicators at admission such as
intubation and pneumonia. This suggests that what happens to the patient
after admission is more predictive than what conditions they arrived with.

3. Comorbidities cluster rather than act independently

The high intercorrelations between COPD, asthma, hypertension,
immunosuppression, and other conditions suggest these should not all be
treated as independent predictors. A dimensionality reduction approach
such as PCA could extract a single comorbidity burden score that
captures the shared variance more cleanly.

4. Linear regression is insufficient for binary classification

Using R-squared as an evaluation metric for a binary outcome is
theoretically inappropriate. R-squared measures variance explained
which does not map cleanly to classification performance. A logistic
regression or Random Forest model with F1 score and AUC-ROC evaluation
would be more appropriate and almost certainly produce better predictions.

5. Missing value treatment is a significant modeling decision

Treating 97 and 99 as 0 assumes that unknown comorbidity status
implies absence of that condition. In reality missing data may be
non-random. Patients in worse condition may have had incomplete
records. A more rigorous approach would use multiple imputation
or a missingness indicator feature.

6. Limitations

- Linear regression is not the appropriate model for binary
  classification and should be replaced with logistic regression
  or a tree-based classifier
- Missing values were imputed as 0 which may introduce bias
- Class imbalance between survivors and deaths was not addressed
- No evaluation metrics appropriate for classification such as
  precision, recall, F1, or AUC-ROC were computed
- High multicollinearity between comorbidity features may destabilize
  coefficient estimates
- The correlation heatmap shows heart_or_not and blood_vessel_or_not
  are perfectly negatively correlated at -1.0 as they are inverses of
  each other and one should be dropped

---

## Tech Stack

Python           Data processing and modeling
pandas           Data cleaning and feature engineering
numpy            Array operations and binary encoding
scikit-learn     Linear regression, train test split
seaborn          Correlation heatmap visualization
matplotlib       Plot rendering

---

## How To Run

1. Download Covid Data.csv from Kaggle

2. Install dependencies:
   pip install pandas numpy scikit-learn seaborn matplotlib

3. Run the script:
   python covid_mortality.py
