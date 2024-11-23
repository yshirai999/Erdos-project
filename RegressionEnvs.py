from BaseEnv import BaseClass
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class RegressionClass(BaseClass):

    def __init__(self,
        feature_steps: int = 10,
        target_steps: int = 1,
        order: tuple[int] = (2,1,2)
    ):  
        super().__init__(feature_steps = feature_steps, target_steps = target_steps)
        self.order = order
        self.models_name_str = {LinearRegression: "Linear Regression", ARIMA: "ARIMA"}
        self.train_series = {}
        self.train_errors = {}
        self.valid_errors = {}
        self.test_mse = {}
        self.y_pred_pcr = {}
        self.y_test_pcr = {}
        self.restored_prices = {}
        self.test_dates = {}
        for name in self.tickers.groups.keys():
            self.train_series[name] = np.concatenate([self.y_train[name].flatten(), self.y_valid[name].flatten()])
            self.train_errors[name] = {LinearRegression: [], ARIMA: []}
            self.valid_errors[name] = {LinearRegression: [], ARIMA: []}
            self.test_mse[name] = {LinearRegression: [], ARIMA: []}
            self.y_pred_pcr[name] = {LinearRegression: [], ARIMA: []}
            self.y_test_pcr[name] = {LinearRegression: [], ARIMA: []}
            self.restored_prices[name] = {LinearRegression: [], ARIMA: []}
            self.test_dates[name] = {LinearRegression: [], ARIMA: []}

    def Prediction(self,
        model
    ):

        for name in self.tickers.groups.keys():

            if model == LinearRegression:
                m = LinearRegression()
                m.fit(self.X_train[name], self.y_train[name].ravel())
                train_pred = m.predict(self.X_train[name])
                valid_pred = m.predict(self.X_valid[name])
                diff_pred = m.predict(self.X_test[name]).flatten()
            elif model == ARIMA:
                m = ARIMA(self.train_series[name], order = self.order)
                result = m.fit()
                train_pred = result.predict(start=0, end=len(self.y_train[name]) - 1, typ="levels")
                valid_pred = result.predict(start=len(self.y_train[name]), end=len(self.train_series[name]) - 1, typ="levels")
                diff_pred = result.forecast(steps=len(self.y_test[name]))
            else:
                raise TypeError("model must be a LinearRegression or ARIMA model")

            self.train_errors[name][model] = mean_squared_error(self.y_train[name], train_pred)
            self.valid_errors[name][model] = mean_squared_error(self.y_valid[name], valid_pred)

            self.restored_prices[name][model] = np.cumsum(diff_pred) + self.prc[name]
            self.y_pred_pcr[name][model] = self.restored_prices[name]

            diff_actual = self.y_test[name].flatten()
            self.y_test_pcr[name][model] = np.cumsum(diff_actual) + self.prc[name]

            self.test_mse[name][model] = mean_squared_error(self.y_test_pcr[name][model], self.restored_prices[name][model])
            self.test_dates[name][model] = self.dates[-len(self.y_test_pcr[name][model]):]

    def Visualization(self,
        model
    ):
        for name in self.tickers.groups.keys():
            plt.figure(figsize=(10, 5))
            plt.plot(self.test_dates[name][model], self.y_test_pcr[name][model], label="Actual")
            plt.plot(self.test_dates[name][model], self.restored_prices[name][model], label="Predicted")
            plt.title(f"{self.models_name_str[model]}: Predicted vs Actual for {name}")
            plt.xlabel("Time Steps")
            plt.ylabel("Price")
            plt.legend()
            plt.show()

        print("Mean Squared Error for each ticker:")
        for name in self.tickers.groups.keys():
            print(f"{name}: Train MSE = {self.train_errors[name][model]:.4f}, Valid MSE = {self.valid_errors[name][model]:.4f}, Test MSE = {self.test_mse[name][model]:.4f}")

