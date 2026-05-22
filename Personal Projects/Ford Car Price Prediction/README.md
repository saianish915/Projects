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
Training set: 14,527 listings
Test set: 3,436 listings
Target: Price (GBP)

Features:

| Feature | Description |
|---------|-------------|
| model | Ford model name (Fiesta, Kuga, etc.) |
| year | Year of manufacture |
| transmission | Manual, Automatic, or Semi-Auto |
| mileage | Total miles driven |
| fuelType | Petrol, Diesel, Hybrid, or Electric |
| mpg | Miles per gallon fuel efficiency |
| engineSize | Engine displacement in litres |

### Data Cleaning Decisions

Three filtering decisions were made before modeling:

1. Focus model removed
   The Ford Focus was removed from the dataset due to anomalous pricing
   patterns that distorted model performance.

2. Other fuel type removed
   Listings with fuelType of Other were removed as this category was
   too ambiguous to encode meaningfully and represented a very small
   proportion of listings.

3. Years after 2020 removed
   Listings from 2021 and beyond were removed as these likely represented
   data entry errors given the dataset was compiled around 2020.

---

## Exploratory Data Analysis

Scatter plots were generated for each feature against price to understand
the nature and direction of relationships before modeling.

### Key Findings From Scatter Plots

MPG vs Price:
Strong negative relationship. Lower MPG vehicles command higher prices,
likely reflecting that larger more powerful engines are less fuel efficient
but more desirable. A cluster of outliers at 200+ MPG suggests data quality
issues in a small number of listings.

Tax vs Price:
Most listings cluster around 100 to 200 GBP annual tax. Higher tax
vehicles do not show a clear price premium, suggesting tax is less
predictive than other features.

Engine Size vs Price:
Clear positive relationship at larger engine sizes (5.0L vehicles
command 25,000 to 50,000 GBP). Engine size of 0 likely represents
electric vehicles with no combustion engine.

Year vs Price:
The strongest visual relationship. Price increases exponentially with
year of manufacture, with post-2015 vehicles showing dramatic price
increases. This is the most visually compelling predictor of price.

Fuel Type vs Price:
Petrol and diesel dominate the dataset. Hybrid vehicles show a moderate
price premium. Electric vehicles are sparsely represented with prices
around 16,000 GBP.

Mileage vs Price:
Clear exponential decay. Price drops sharply through the first 25,000
miles then continues declining more gradually. This is the classic
used car depreciation curve and one of the clearest relationships
in the dataset.

Transmission vs Price:
Automatic and Semi-Auto transmissions tend toward slightly higher prices
than manual, consistent with general used car market patterns, though
the relationship is noisy.

---

## Methodology

### Encoding

Categorical features were label encoded using sklearn LabelEncoder:
- model encoded to integer
- transmission encoded to integer
- fuelType encoded to integer

### Feature Selection

Seven features were selected as model inputs:
model, year, transmission, mileage, fuelType, mpg, engineSize

### Train Test Split

An 80/20 split was implemented using a random mask. 14,527 listings
were used for training and 3,436 for testing.

### Model

Linear Regression was used as the modeling approach, estimating a
linear relationship between the seven input features and price.

---

## Experiments

### Experiment 1: Exploratory Scatter Analysis

Scatter plots confirmed that year and mileage have the clearest and
strongest relationships with price. Engine size shows a positive
relationship particularly at the higher end. MPG shows a negative
relationship consistent with larger less efficient vehicles being
more expensive.

### Experiment 2: Data Cleaning Impact

Three filtering decisions removed problematic data before modeling.
Removing the Focus model, Other fuel type, and post-2020 listings
cleaned anomalies that would have distorted coefficient estimates
and inflated error on the test set.

### Experiment 3: Linear Regression

A Linear Regression model was trained on the cleaned and encoded dataset.

Model coefficients:
year:         +1263.77  (each additional year adds ~1,264 GBP)
engineSize:   +4182.45  (each litre of engine size adds ~4,182 GBP)
transmission:  -274.66  (negative due to label encoding ordering)
mileage:         -0.06  (each additional mile reduces price by 6 pence)
fuelType:      -427.15  (varies by encoding order)
mpg:           -111.91  (higher MPG associated with lower price)
model:          +26.17  (varies by encoding order)

Intercept: -2,533,447.86

The large negative intercept reflects that the model needs to offset
the large positive contribution of year (e.g. year 2019 multiplied
by 1,263.77 would be unrealistically large without the intercept).

R-squared on test set: 0.747

---

## Results

Model: Linear Regression
Features: 7
Training set: 14,527 listings
Test set: 3,436 listings
R-squared: 0.747

The model explains 74.7% of the variance in used Ford vehicle prices
using seven features. This is a strong result for a linear model on
a real-world pricing dataset, suggesting that year, mileage, and
engine size together capture the majority of price variation.

---

## Conclusion and Takeaways

1. The hypothesis was supported

74.7% of used Ford price variance is explained by just seven features.
Year and mileage are the dominant drivers, consistent with standard
used car depreciation theory. A buyer or seller can estimate fair
market value with reasonable confidence using only these observable
characteristics.

2. Year is the single most important predictor

The coefficient of +1,263.77 per year means each additional year of
manufacture adds approximately 1,264 GBP to the expected price holding
all other features constant. The scatter plot confirms this relationship
is strong and consistent across the entire dataset.

3. Mileage follows the classic depreciation curve

The scatter plot shows the classic exponential decay of price with
mileage. The linear coefficient of -0.056 per mile captures the average
rate of this decay but misses the nonlinearity, where early miles
depreciate value much faster than later miles. A log transformation
of mileage would likely improve model performance.

4. Label encoding introduces false ordinal relationships

Encoding Ford model names and fuel types as integers implies a numeric
ordering that does not exist. A Fiesta encoded as 3 and a Kuga encoded
as 7 does not mean the Kuga is mathematically 4 units more than the
Fiesta in any meaningful sense. One-hot encoding would treat each
category independently and likely improve coefficient interpretability
and model performance.

5. Limitations

- Label encoding of nominal features introduces false ordinal
  relationships for model, transmission, and fuelType
- No fixed random state on the train test split makes results
  slightly variable between runs
- Linear model cannot capture the nonlinear depreciation curves
  visible in the mileage and year scatter plots
- No cross validation was performed so the single R-squared score
  may not fully represent generalization performance
- Outliers such as the 200+ MPG listings and the single 3.2L engine
  listing were not investigated or removed
- A gradient boosting model would almost certainly outperform linear
  regression on this dataset

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
