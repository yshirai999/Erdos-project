# AI based stock market predictions

- The aim of this project is to test the capability of different machine learning methods in predicting the close of business (COB) mid prices

$$\begin{align*} St = (S^1_t , ..., S^d_t )\end{align*}$$

-  of $d$ stocks, given the previous n days of COB prices $S_{t−n}, ..., S_{t−1}$ and other variables, including Treasury rates. The construction of reliable stock market predictions are useful for detecting interdependencies within different assets, and, ultimately exploit them for the construction of quantitative trading strategies.

# Source of data

- Our data source is the Wharton Research Dataset.
- We plan to build a python file that contains various SQL queries to automatically download the data into our Github repo.
- This is a dataset often used in the academic literature and contains reliable stock market quotes, volume traded, and other information.

# Project plan and goal
- We will focus on the 10 sector ETFs and the SPY fund tracking the S&P500 index.
- The current plan is to first split the dataset into training, validation and testing sets.
- We will then train some basic models from regression analysis, such as:
  - Linear Regression and the autoregressive integrated moving average (ARIMA) model,
  - Recurrent neural networks architectures,
  - Long Short Term Memory architecture.
- Hyperparameter tuning will be performed for each model on the validation set.
- Root mean square error in predictions will be estimated on the testing set for each model.
- The final goal is to understand which model achieves minimal RMSE on the validation test, which will be reported in our delivery.
