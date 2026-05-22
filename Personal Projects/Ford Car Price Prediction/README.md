# Ford Used Car Price Predictor

## Hypothesis

The resale price of a used Ford vehicle is not arbitrary but is
systematically determined by a combination of measurable vehicle
characteristics. If the key drivers of used car value can be identified
and modeled, a regression model should be able to predict the price of
an unseen Ford vehicle based on its features alone.

---

## Background

The used car market presents a classic information asymmetry problem.
Sellers know more about their vehicle than buyers, and buyers often
struggle to determine whether a listed price is fair. A data-driven
pricing model trained on historical transactions can help buyers identify
underpriced vehicles and help sellers price competitively.

This project uses a dataset of approximately 17,966 used Ford vehicle
listings to build a linear regression model that predicts price from
vehicle characteristics including model, year, transmission, mileage,
fuel type, MPG, and engine size.

---

## Dataset

Source: Ford Used Car Dataset (Kaggle)
Raw size: 17,966 listings
Target: Price (GBP)

Features:

| Feature | Description |
|---------|-------------|
| model | Ford model name (Fiesta, Focus, Kuga, etc.) |
| year | Year of manufacture |
| transmission | Manual, Automatic, or Semi-Auto |
| mileage | Total miles driven |
| fuelType | Petrol, Diesel, Hybrid, or Electric |
| tax | Annual road tax (GBP) |
| mpg | Miles per gallon fuel efficiency |
| engineSize | Engine displacement in litres |

### Data Cleaning Decisions

Three filtering decisions were made before modeling:

1. Focus model removed
   The Ford Focus was removed from the dataset. This decision likely
   reflects an anomaly in Focus pricing data that distorted model
   performance, such as extreme outliers or inconsistent pricing.

2. Other fuel type removed
   Listings with fuelType of Other were removed as this category was
   too ambiguous to encode meaningfully and represented a very small
   proportion of listings.

3. Years after 2020 removed
   Listings from 2021 and beyond were removed as these likely represented
   data entry errors given the dataset was compiled around 2020.

---

## Methodology

### Exploratory Data Analysis

Scatter plots were generated for each feature against price to understand
the nature and direction of relationships:

| Feature | Expected Relationship with Price |
|---------|----------------------------------|
| mpg | Negative - higher efficiency cars tend to be smaller and cheaper |
| tax | Positive - higher tax often correlates with larger more expensive vehicles |
| engineSize | Positive - larger engines command higher prices |
| year | Positive - newer cars are worth more |
| mileage | Negative - higher mileage reduces resale value |
| fuelType | Variable - hybrid and electric command premiums |
| transmission | Variable - automatic typically commands a premium |

### Encoding

Categorical features were label encoded using sklearn LabelEncoder:
- model: encoded to integer
- transmission: encoded to integer
- fuelType: encoded to integer

Note: Label encoding assumes an ordinal relationship between categories
which does not hold for nominal features like model or fuel type. One-hot
encoding would be more theoretically appropriate.

### Feature Selection

Seven features were selected as model inputs:
model, year, transmission, mileage, fuelType, mpg, engineSize

Tax was excluded from the model inputs despite being in the dataset.

### Train Test Split

An 80/20 split was implemented using a random mask rather than sklearn
train_test_split. 80% of rows were randomly assigned to training and
the remaining 20% to testing.

### Model

Linear Regression was used as the modeling approach, estimating a
linear relationship between the seven input features and price.

---

## Experiments

### Experiment 1: Exploratory Scatter Analysis

Scatter plots were generated for each feature against price to visually
identify which features had the strongest and clearest relationship
with sale price.

Key observations:
- Engine size showed the clearest positive relationship with price
- Year showed a strong positive relationship with price
- Mileage showed a negative relationship with price as expected
- MPG showed a noisy negative relationship

### Experiment 2: Data Cleaning Impact

Three filtering decisions were made before modeling. Each decision
removed potentially problematic data that could have distorted model
performance:

- Focus removal cleaned up model-specific pricing anomalies
- Other fuel type removal removed an uninterpretable category
- Post-2020 year removal removed likely data entry errors

### Experiment 3: Linear Regression

A Linear Regression model was trained on the cleaned and encoded dataset.

Model coefficients show the marginal impact of each feature on price
holding all other features constant.

R-squared on test set: [paste your model.score output here]

---

## Results

Model: Linear Regression
R-squared: [paste your score here]

Model coefficients: [paste your model.coef_ output here]
Intercept: [paste your model.intercept_ output here]

An R-squared of X means the model explains X% of the variance in used
Ford vehicle prices using the seven selected features.

---

## Conclusion and Takeaways

1. The hypothesis was partially supported

Linear regression can capture the broad directional relationships between
vehicle characteristics and price. Year, mileage, and engine size are
likely the dominant predictors consistent with general used car market
intuition.

2. Label encoding is inappropriate for nominal categories

Encoding Ford model names as integers (0, 1, 2, etc.) implies that
the numeric distance between model codes is meaningful, which it is not.
A Fiesta encoded as 0 and a Kuga encoded as 5 does not mean the Kuga
is five times the Fiesta in some meaningful sense. One-hot encoding
would treat each model as an independent category without implied ordering.

3. Linear regression likely underfits this problem

Used car prices likely have nonlinear relationships with mileage and year,
and significant interaction effects between features such as model and
year, or engine size and fuel type. A gradient boosting model would
almost certainly outperform linear regression on this dataset.

4. The random mask split introduces variability

Using a random mask without a fixed random state means each run produces
a different train test split, making results non-reproducible.
sklearn train_test_split with random_state=42 would fix this.

5. Limitations

- Label encoding of nominal features introduces false ordinal relationships
- Tax feature excluded without documented justification
- No fixed random state makes results non-reproducible between runs
- Linear model cannot capture nonlinear price depreciation curves
- No cross validation was performed so the single test score may not
  generalize
- Focus model removed without full documentation of why

---

## Tech Stack

Python           Data processing and modeling
pandas           Data loading and cleaning
numpy            Array operations and train test splitting
matplotlib       Scatter plot visualizations
scikit-learn     Label encoding and linear regression

---

## How To Run

1. Download ford.csv from Kaggle:
   https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes

2. Install dependencies:
   pip install pandas numpy matplotlib scikit-learn

3. Run the script:
   python ford_price_predictor.py
