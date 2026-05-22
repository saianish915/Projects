# Airbnb Price Predictor

## Hypothesis

Airbnb listing prices are not arbitrary but are systematically influenced
by measurable features of the listing including location, room type,
availability, and host characteristics. If this is true, a linear model
trained on these features should be able to predict nightly price with
meaningful accuracy.

---

## Background

Airbnb pricing is a complex function of dozens of variables. Hosts set
prices manually, often inconsistently, creating both overpriced and
underpriced listings in the same neighborhood. Understanding what drives
price allows:

- Hosts to price listings competitively
- Guests to identify fair value
- Investors to evaluate rental income potential

This project uses a dataset of over 100,000 New York City Airbnb listings
to build and evaluate a linear regression model for nightly price prediction.

---

## Dataset

Source: Airbnb Open Data (New York City)
Size: 103,005 listings

Features include:

| Feature | Description |
|---------|-------------|
| neighbourhood group | Borough (Manhattan, Brooklyn, Queens, Bronx, Staten Island) |
| neighbourhood | Specific neighborhood within borough |
| lat / long | Geographic coordinates |
| room type | Entire home, private room, or shared room |
| Construction year | Year the property was built |
| service fee | Additional fee charged per booking |
| minimum nights | Minimum stay requirement |
| number of reviews | Total reviews received |
| last review | Date of most recent review |
| reviews per month | Average monthly review frequency |
| review rate number | Average rating score |
| calculated host listings count | Number of listings the host manages |
| availability 365 | Number of days available per year |
| instant bookable | Whether listing can be booked instantly |
| cancellation policy | Strict, moderate, or flexible |

Target variable: nightly price (USD)

---

## Methodology

### Data Preprocessing

The price column was stored as a formatted string with dollar signs and
commas. These were stripped and the column converted to float before
modeling:

price column: "$966 " → 966.0

### Model Selection

Linear regression was selected as the baseline model for the following
reasons:

- Interpretable coefficients allow understanding of which features
  drive price
- Fast to train on large datasets
- Establishes a performance baseline for more complex models

### Train Test Split

Data was split 80/20 with a fixed random state of 42 to ensure
reproducibility:

Training set: 82,404 listings
Test set:     20,601 listings

### Evaluation Metric

Root Mean Squared Error (RMSE) was used as the primary evaluation metric.
RMSE penalizes large errors more heavily than MAE, which is appropriate
for price prediction where large mispredictions are costly.

---

## Results

Model: Linear Regression
RMSE: [your RMSE value here]

RMSE represents the average dollar error in the model's price predictions
on unseen listings.

---

## Limitations and Future Work

1. Linear assumption may not hold
   Airbnb prices likely have nonlinear relationships with features.
   Gradient boosting or random forest models would likely improve RMSE
   significantly.

2. Text features not used
   The listing name and house rules columns contain pricing signals
   that a linear model cannot capture. NLP feature extraction could
   add value.

3. Outliers not handled
   Luxury listings priced above $5,000 per night likely distort the
   model. Outlier removal or log transformation of the target variable
   would improve performance.

4. Temporal features not engineered
   Last review date was not converted into a recency feature. A listing
   with a recent review signals active demand which likely correlates
   with price.

5. Geographic features underutilized
   Raw latitude and longitude were included but neighborhood level
   clustering or distance to landmarks would better capture location
   value.

---

## Tech Stack

Python          Data processing and modeling
pandas          Data cleaning and feature preparation
scikit-learn    Linear regression, train test split, RMSE evaluation

---

## How To Run

1. Clone the repository
2. Install dependencies:
   pip install pandas scikit-learn
3. Place Airbnb_Open_Data.csv in the project directory
4. Run the script:
   python airbnb_predictor.py

---

## Disclaimer

This project is for educational purposes only and is not intended
for commercial use.
