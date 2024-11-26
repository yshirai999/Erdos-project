# Project summary

- This simple project was performed as part of the Data Science bootcamp of the Erdos Institute Fall 2024 program.
- The aim of this project is to test the capability of different machine learning methods in predicting the close of business (COB) mid price $S_t$ of $3$ stocks, given the previous 10 days of COB prices $S_{t−10}, ..., S_{t−1}$. The three stocks are SPY, AAPL and IBM.

## Source of data

- Our data source is the Wharton Research Dataset.
- This is a dataset often used in the academic literature and contains reliable stock market quotes, volume traded, and other information.

## Project plan and goal

- The ``BaseClass`` in the ``BaseEnv.py`` file reads the data and:
  - transforms prices into log-differeneces
  - standardizes it
  - and splits it into training, validation and testing sets.
- Regression and RNN classes inherit from the Basic Class, and their prediction methods train the models
- The Regression class includes Linear Regression and the autoregressive integrated moving average (ARIMA) model,
- The RNN class includes a Simple Recurrent neural networks (with possibly multiple layers and dropout and batch normalization options)
- It also includess a Long Short Term Memory architecture also with possibly multiple layers and dropout/batch normalization.
- Hyperparameter tuning was performed for each model on the validation set.

## Results

- ARIMA does not outperfom linear regression, possibly for lack of stationarity in time series of asset prices
- Simple RNN outperforms LR for AAPL, but is unable to explain the big variation in prices occurred during covid time for SPY and IBM and results in a biased estimate
- LSTM outperforms all the other methods in terms of MSE
- However, LSTM seem to achieve the lower MSE by reducing volaility in the price changes
- This opens the question of whether MSE is the correct performance measure for this task.

## Installation

- Download the repo locally
- To install the conda environment, run in the terminal `conda env create --name AI_Predictions --file=environment.yml`
- All the results can be obtained by running all the celss in the ``Main.ipynb` workbook.
