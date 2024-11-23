from BaseEnv import BaseClass
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


class ARIMAClass(BaseClass):

    def __init__(self,
        order: tuple[int] = (2,1,2),
        feature_steps: int = 10,
        target_steps: int = 1,
    ):
        BaseClass.__init__(feature_steps=feature_steps, target_steps=target_steps)
        self.order = order

    def Prediction(self
    ):

        self.train_errors = {}
        self.valid_errors = {}
        self.test_mse = {}
        self.y_pred_pcr = {}

        for name in self.tickers.groups.keys():

            train_series = np.concatenate([self.y_train[name].flatten(), self.y_valid[name].flatten()])
            model = ARIMA(train_series, order = self.order)  # (p, d, q)
            arima_result = model.fit()

            train_pred = arima_result.predict(start=0, end=len(self.y_train[name]) - 1, typ="levels")
            valid_pred = arima_result.predict(start=len(self.y_train[name]), end=len(train_series) - 1, typ="levels")
            train_error = mean_squared_error(self.y_train[name], train_pred)
            valid_error = mean_squared_error(self.y_valid[name], valid_pred)

            self.train_errors[name] = train_error
            self.valid_errors[name] = valid_error

            diff_pred = arima_result.forecast(steps=len(self.y_test[name]))
            self.restored_prices = np.cumsum(diff_pred) + self.prc[name]
            self.y_pred_pcr[name] = self.restored_prices

            diff_actual = self.y_test[name].flatten()
            self.y_test_pcr = np.cumsum(diff_actual) + self.prc[name]


            self.test_mse[name] = mean_squared_error(self.y_test_pcr, self.restored_prices)
            self.test_dates = self.dates[-len(self.y_test_pcr):]

    def Visualization(self
    ):

        plt.figure(figsize=(10, 5))
        plt.plot(self.test_dates, self.y_test_pcr, label="Actual")
        plt.plot(self.test_dates, self.restored_prices, label="Predicted")
        plt.title(f"Predicted vs Actual for {name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        print("Mean Squared Error for each ticker:")
        for name in self.tickers.groups.keys():
            print(f"{name}: Train MSE = {self.train_errors[name]:.4f}, Valid MSE = {self.valid_errors[name]:.4f}, Test MSE = {self.test_mse[name]:.4f}")


class LinearRegressionClass(BaseClass):

    def __init__(self,
        feature_steps: int = 10,
        target_steps: int = 1,
    ):
        BaseClass.__init__(feature_steps=feature_steps, target_steps=target_steps)

    def Prediction(self
    ):

        self.train_errors = {}
        self.valid_errors = {}
        self.test_mse = {}
        self.y_pred_pcr = {}

        for name in self.tickers.groups.keys():
            model = LinearRegression()

            model.fit(self.X_train[name], self.y_train[name].ravel())

            train_error = mean_squared_error(self.y_train[name], model.predict(self.X_train[name]))
            valid_error = mean_squared_error(self.y_valid[name], model.predict(self.X_valid[name]))

            self.train_errors[name] = train_error
            self.valid_errors[name] = valid_error

            diff_pred = model.predict(self.X_test[name]).flatten()
            self.restored_prices = np.cumsum(diff_pred) + self.prc[name]
            self.y_pred_pcr[name] = self.restored_prices

            diff_actual = self.y_test[name].flatten()
            self.y_test_pcr = np.cumsum(diff_actual) + self.prc[name]

            self.test_dates = self.dates[-len(self.y_test_pcr):]
            self.test_mse[name] = mean_squared_error(self.y_test_pcr, self.restored_prices)

    def Visualization(self
    ):

        plt.figure(figsize=(10, 5))
        plt.plot(self.test_dates, self.y_test_pcr, label="Actual")
        plt.plot(self.test_dates, self.restored_prices, label="Predicted")
        plt.title(f"Predicted vs Actual for {name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        print("Mean Squared Error for each ticker:")
        for name in self.tickers.groups.keys():
            print(f"{name}: Train MSE = {self.train_errors[name]:.4f}, Valid MSE = {self.valid_errors[name]:.4f}, Test MSE = {self.test_mse[name]:.4f}")