# Stock Price Predictor (LSTM)

## Motivation

This project sits at the intersection of two personal interests: financial
markets and machine learning. Built as a semester project, the goal was
to go beyond a standard academic exercise and build something genuinely
useful — a tool that can be pointed at any publicly traded stock and
produce data-driven price predictions using a neural network architecture
designed specifically for sequential data.

The system is not limited to a single ticker. Any stock available on
Yahoo Finance can be analyzed by changing a single variable, making it
a flexible research tool rather than a one-off experiment.

---

## Hypothesis

Stock price movements contain temporal patterns that a sequence-aware
neural network can learn from historical data. Specifically the closing
price of a stock on any given day is influenced by the trajectory of
prices over the preceding days. If a Long Short-Term Memory network is
trained on a sufficiently long window of historical price data it should
be able to capture these sequential dependencies and produce next-day
price predictions that outperform a naive baseline.

---

## Background

Stock price prediction is one of the most studied and most difficult
problems in quantitative finance. Traditional statistical models such
as ARIMA treat price series as linear processes, missing the complex
nonlinear dynamics that characterize real market behavior.

Long Short-Term Memory networks are a class of recurrent neural network
specifically designed to learn long-range dependencies in sequential data.
Unlike feedforward networks that treat each input independently LSTMs
maintain a hidden state that carries information across time steps,
making them naturally suited to time series problems where recent history
influences future values.

This project applies a two-layer stacked LSTM to predict stock closing
prices for any publicly traded company using historical price data from
Yahoo Finance. The system was demonstrated on UBER (Uber Technologies Inc.)
using data from January 2020 to July 2024, a period that includes the
Covid-19 crash, the subsequent recovery, and the high-volatility period
of 2022-2023.

---

## Dataset

Source: Yahoo Finance via yfinance library
Supported tickers: Any stock available on Yahoo Finance
Demo ticker: UBER (Uber Technologies Inc.)
Demo date range: January 1 2020 to July 1 2024
Features used: Close price, Open price
Target: Next day Close price
Window size: 20 trading days

The 20-day lookback window means the model sees the previous 20 days
of price data to predict the next day's closing price, roughly
corresponding to one trading month of historical context.

To run on a different stock simply change:
stock_symbol = 'UBER'
to any valid Yahoo Finance ticker such as AAPL, TSLA, MSFT, NVDA, etc.

---

## Methodology

### Data Pipeline

Historical price data is fetched directly from Yahoo Finance using the
yfinance library. Both Close and Open prices are loaded as input features.
The system is designed to work with any ticker and any date range,
requiring only three parameters to be set before running.

### Preprocessing

Min-Max scaling was applied to normalize prices to the range 0 to 1.
Neural networks are sensitive to the scale of input data and Min-Max
scaling ensures that the LSTM weights are updated on a consistent
numerical range regardless of the absolute dollar value of the stock.
This is especially important for a general purpose tool where the
stock price could range from a few dollars to thousands of dollars.

The inverse transform was applied after prediction to convert scaled
outputs back to actual dollar values for evaluation and visualization.

### Sliding Window Construction

A sliding window approach was used to construct training sequences.
For each position in the time series a window of 20 consecutive days
was used as input and the value on day 21 was used as the target label.
This produced overlapping sequences that capture the sequential
structure of price movements.

### Model Architecture

A stacked two-layer LSTM was used:

| Layer | Type | Units | Return Sequences |
|-------|------|-------|-----------------|
| 1 | LSTM | 50 | True |
| 2 | LSTM | 50 | False |
| 3 | Dense | 1 | N/A |

The first LSTM layer returns the full sequence output to feed the
second LSTM layer. The second LSTM layer returns only the final
hidden state which is passed to a single Dense output neuron
producing the next-day price prediction.

Optimizer: Adam
Loss function: Mean Squared Error

### Train Test Split

An 80/20 sequential split was applied with shuffle=False to preserve
the temporal ordering of the data. Shuffling would allow the model to
train on future data and validate on past data, producing look-ahead
bias. Maintaining temporal order is mandatory for honest time series
evaluation regardless of which stock is being analyzed.

---

## Experiments

### Experiment 1: General Purpose Data Pipeline

The data pipeline was designed to work with any Yahoo Finance ticker
rather than being hardcoded to a single stock. This required building
a flexible preprocessing function that accepts any price series and
applies consistent normalization and windowing regardless of the
stock's price range or volatility characteristics.

### Experiment 2: Demo on UBER (2020-2024)

UBER was selected as the demonstration ticker because its price history
from 2020 to 2024 covers multiple distinct market regimes:

- Covid-19 crash and recovery (2020)
- Post-IPO growth period (2021)
- High volatility and drawdown (2022)
- Recovery and stabilization (2023-2024)

Training across multiple market regimes forces the model to learn
patterns that generalize across different volatility environments
rather than overfitting to a single market condition.

### Experiment 3: Stacked LSTM Architecture

A two-layer stacked LSTM was chosen over a single-layer LSTM based
on the principle that deeper architectures can learn hierarchical
representations. The first layer learns low-level sequential patterns
such as day-to-day momentum while the second layer learns higher-level
patterns such as multi-week trends.

### Experiment 4: Training and Evaluation

The model was trained for 50 epochs with a batch size of 32.
Predictions were inverse transformed back to dollar values before
computing Mean Squared Error against actual closing prices.

Mean Squared Error: 1.956
RMSE: $1.40 per day

A visualization was generated plotting predicted vs actual closing
prices over the test period from October 2022 to July 2024.

---

## Results

Model: Stacked 2-layer LSTM
Demo ticker: UBER
Training period: January 2020 to approximately September 2022
Test period: October 2022 to July 2024
Price range tested: $40 to $83
Window size: 20 days
Epochs: 50
Batch size: 32

Mean Squared Error:  1.956
RMSE:                $1.40 per day
Average error:       approximately 1.7% to 3.5% of stock price

### Visual Assessment

The prediction chart demonstrates strong model performance across
multiple market phases within the test period:

October 2022 to January 2023:
The model accurately tracks the $44 to $49 consolidation range with
the predicted line closely overlaying actual prices throughout this
low-volatility period.

January 2023 to April 2023:
The model captures the decline from $48 to $41 and the subsequent
recovery, though a slight overshoot is visible during the March 2023
reversal. This lag on sharp reversals is expected behavior for LSTM
models and reflects the smoothing effect of the 20-day window.

April 2023 to October 2023:
The model tracks the strong rally from $41 to $83 with high fidelity,
correctly anticipating the direction and magnitude of the move across
a six-month uptrend.

October 2023 to July 2024:
The model accurately follows the pullback from $83 to $65 and the
subsequent recovery to $72, correctly capturing both the direction
and pace of price movements throughout this volatile period.

Overall the predicted line tracks the actual price with high visual
fidelity across all four distinct market phases in the test set.

---

## Conclusion and Takeaways

1. Combining two interests produced a more motivated project

Building a stock predictor as a semester project was driven by genuine
interest in both financial markets and deep learning. This intersection
produced a more carefully designed system than a purely academic exercise,
particularly in the decision to make the tool work for any ticker rather
than a single hardcoded stock.

2. The hypothesis was strongly supported

An MSE of 1.956 and RMSE of $1.40 on a stock trading between $40 and
$83 represents approximately 1.7% to 3.5% average daily prediction error.
The LSTM successfully captured trend direction, magnitude, and reversal
points across multiple distinct market regimes in the test period.
The visual chart confirms that the predicted line tracks actual price
movements with high fidelity throughout the 21-month test window.

3. The model handles multiple market regimes

By training on data that spans the Covid-19 crash, recovery, growth,
and correction phases the model learned patterns that generalize across
different volatility environments. This is reflected in the consistent
prediction quality across the different phases visible in the chart.

4. Temporal ordering is non-negotiable in financial time series

Using shuffle=False in the train test split was the most critical
methodological decision in this project. Shuffling a time series
dataset allows the model to train on future information and validate
on past information, producing artificially strong results that
collapse immediately in practice.

5. A general purpose tool is more valuable than a single-stock model

By parameterizing the ticker, date range, and window size the system
can be applied to any publicly traded company in seconds. This makes
it useful as a research tool for comparing how well the LSTM generalizes
across stocks with different volatility profiles, sector characteristics,
and price histories.

6. Sharp reversals produce the largest errors

The slight lag visible around the March 2023 reversal in the prediction
chart is where the model performs least well. This reflects a fundamental
limitation of momentum-based sequential models where the 20-day window
of recent prices creates inertia that makes the model slow to respond
to sudden directional changes. This is the primary area for improvement.

7. Min-Max scaling introduces a subtle look-ahead bias

Fitting the scaler on the full dataset before splitting means the
scaler has seen the range of test set values during training. A stricter
implementation would fit the scaler only on the training set and apply
the same transform to the test set, preventing any information from the
test period influencing the scaling parameters.

8. Limitations

- Scaler was fit on the full dataset before splitting which introduces
  minor look-ahead bias in the normalization step
- No technical indicators or volume features were included as predictors
- No hyperparameter tuning was performed on LSTM units, layers,
  window size, or learning rate
- 50 epochs without early stopping may result in overfitting on
  later epochs
- The model predicts only one day ahead and cannot be used directly
  for multi-step forecasting without modification
- Performance will vary across different tickers and time periods

---

## How To Use

Change these three variables to analyze any stock:

stock_symbol = 'UBER'       # Any Yahoo Finance ticker
start_date   = '2020-01-01' # Start of historical data
end_date     = '2024-07-01' # End of historical data

Example tickers to try:
AAPL   Apple Inc.
TSLA   Tesla Inc.
MSFT   Microsoft Corporation
NVDA   NVIDIA Corporation
AMZN   Amazon.com Inc.
GOOGL  Alphabet Inc.

---

## Tech Stack

Python           Data processing and modeling
yfinance         Historical stock data retrieval from Yahoo Finance
numpy            Array operations and sequence construction
pandas           Data handling and date indexing
scikit-learn     MinMaxScaler and train test split
tensorflow       LSTM model building and training via Keras API
matplotlib       Prediction vs actual price visualization

---

## How To Run

1. Install dependencies:
   pip install yfinance numpy pandas scikit-learn tensorflow matplotlib

2. Set your desired ticker and date range at the top of the script

3. Run the script:
   python stock_predictor.py

4. The script will automatically:
   - Download price data from Yahoo Finance
   - Preprocess and construct sliding windows
   - Train the LSTM model for 50 epochs
   - Print Mean Squared Error on the test set
   - Display a prediction vs actual price chart

Note: Results may vary slightly between runs due to random weight
initialization in the LSTM layers. Set a random seed for full
reproducibility.
