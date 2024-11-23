from BaseEnv import BaseClass
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class RegressionClass(BaseClass):

    def __init__(self,
        order: tuple[int] = (2,1,2),
        feature_steps: int = 10,
        target_steps: int = 1,
    ):
        BaseClass.__init__(feature_steps=feature_steps, target_steps=target_steps)
        self.models_name_str = {LinearRegression: "Linear Regression", ARIMA: "ARIMA"}
        self.models = {}
        self.train_series = {}
        self.train_errors = {}
        self.valid_errors = {}
        self.test_mse = {}
        self.y_pred_pcr = {}
        self.restored_prices = {}
        for name in self.tickers.groups.keys():
            self.train_series[name] = np.concatenate([self.y_train[name].flatten(), self.y_valid[name].flatten()])
            self.models[name] = {LinearRegression: LinearRegression, ARIMA: ARIMA(self.train_series[name], order = order)}
            self.train_errors[name] = {LinearRegression: [], ARIMA: []}
            self.valid_errors[name] = {LinearRegression: [], ARIMA: []}
            self.test_mse[name] = {LinearRegression: [], ARIMA: []}
            self.y_pred_pcr[name] = {LinearRegression: [], ARIMA: []}
            self.restored_prices[name] = {LinearRegression: [], ARIMA: []}

    def Prediction(self,
        Regression
    ):

        for name in self.tickers.groups.keys():
            model = self.models(Regression)

            if model == LinearRegression:
                result = model.fit(self.X_train[name], self.y_train[name].ravel())
                train_pred = result.predict(self.X_train[name])
                valid_pred = model.predict(self.X_valid[name])
            elif model == ARIMA:
                result = model.fit()
                train_pred = result.predict(start=0, end=len(self.y_train[name]) - 1, typ="levels")
                valid_pred = result.predict(start=len(self.y_train[name]), end=len(self.train_series[name]) - 1, typ="levels")
            else:
                raise TypeError("model must be a LinearRegression or ARIMA model")

            self.train_errors[name][Regression] = mean_squared_error(self.y_train[name], train_pred)
            self.valid_errors[name][Regression] = mean_squared_error(self.y_valid[name], valid_pred)

            diff_pred = result.forecast(steps=len(self.y_test[name]))
            self.restored_prices[name][Regression] = np.cumsum(diff_pred) + self.prc[name]
            self.y_pred_pcr[name][Regression] = self.restored_prices[name]

            diff_actual = self.y_test[name].flatten()
            self.y_test_pcr[name][Regression] = np.cumsum(diff_actual) + self.prc[name]

            self.test_mse[name][Regression] = mean_squared_error(self.y_test_pcr[name], self.restored_prices[name][Regression])
            self.test_dates[name][Regression] = self.dates[-len(self.y_test_pcr[name][Regression]):]

    def Visualization(self,
        Regression
    ):
        for name in self.tickers.groups.keys():
            plt.figure(figsize=(10, 5))
            plt.plot(self.test_dates[name][Regression], self.y_test_pcr[name][Regression], label="Actual")
            plt.plot(self.test_dates[name][Regression], self.restored_prices[name][Regression], label="Predicted")
            plt.title(f"{self.models_name_str[Regression]}: Predicted vs Actual for {name}")
            plt.xlabel("Time Steps")
            plt.ylabel("Price")
            plt.legend()
            plt.show()

        print("Mean Squared Error for each ticker:")
        for name in self.tickers.groups.keys():
            print(f"{name}: Train MSE = {self.train_errors[name]:.4f}, Valid MSE = {self.valid_errors[name]:.4f}, Test MSE = {self.test_mse[name]:.4f}")

