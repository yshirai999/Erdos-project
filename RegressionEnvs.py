from BaseEnv import BaseClass
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model as GARCH
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

class RegressionClass(BaseClass):

    def __init__(self,
        feature_steps: int = 10,
        target_steps: int = 1,
        order: tuple[int] = (10,0,1)
    ):  
        super().__init__(feature_steps = feature_steps, target_steps = target_steps)
        self.models = {}
        self.order = order
        self.models_name_str = {LinearRegression: "Linear Regression", ARIMA: "ARIMA", GARCH: "GARCH"}
        self.train_series = {}
        self.train_pred = {}
        self.valid_pred = {}
        self.test_pred = {}
        self.test_pred_rescaled = {}
        self.train_errors = {}
        self.valid_errors = {}
        self.test_errors = {}
        self.y_pred_prc = {}
        self.y_test_prc = {}
        self.restored_prices = {}
        self.test_dates = {}
        for name in self.tickers.groups.keys():
            self.models[name] = {}
            self.train_series[name] = np.concatenate( (self.y_train[name],self.y_valid[name],self.y_test[name]), axis=0)
            self.train_pred[name] = {LinearRegression: [], ARIMA: []}
            self.valid_pred[name] = {LinearRegression: [], ARIMA: []}
            self.test_pred[name] = {LinearRegression: [], ARIMA: []}
            self.test_pred_rescaled = {LinearRegression: [], ARIMA: []}
            self.train_errors[name] = {LinearRegression: [], ARIMA: []}
            self.valid_errors[name] = {LinearRegression: [], ARIMA: []}
            self.test_errors[name] = {LinearRegression: [], ARIMA: []}
            self.y_pred_prc[name] = {LinearRegression: [], ARIMA: []}
            self.y_test_prc[name] = {LinearRegression: [], ARIMA: []}
            self.restored_prices[name] = {LinearRegression: [], ARIMA: []}
            self.test_dates[name] = {LinearRegression: [], ARIMA: []}

    def Prediction(self,
        model
    ):

        for name in self.tickers.groups.keys():
            
            n_train = len(self.X_train[name])
            n_valid = len(self.X_valid[name])
            n_test = len(self.X_test[name])

            if model == LinearRegression:
                m = LinearRegression()
                m.fit(self.X_train[name], self.y_train[name].ravel())
                self.train_pred[name][model] = m.predict(self.X_train[name])
                self.valid_pred[name][model] = m.predict(self.X_valid[name])
                self.test_pred[name][model] = m.predict(self.X_test[name])
            elif model == ARIMA:
                m = ARIMA(self.train_series[name][:n_train], order = self.order).fit()
                self.train_pred[name][model] = m.predict(start = 0, end = n_train-1)
                pred = []
                for t in range(n_train,n_train+n_valid+n_test):
                    m = m.extend(self.train_series[name][t])
                    pred.append(m.forecast(1)[0])
                pred = np.array(pred)
                self.valid_pred[name][model] = pred[:n_valid]
                self.test_pred[name][model] = pred[n_valid:]
            else:
                raise TypeError("model must be a LinearRegression, ARIMA or GARCH model")

            self.test_pred[name][model] = self.test_pred[name][model][...,np.newaxis]

            self.train_errors[name][model] = mean_squared_error(self.y_train[name], self.train_pred[name][model])
            self.valid_errors[name][model] = mean_squared_error(self.y_valid[name], self.valid_pred[name][model])
            self.test_errors[name][model] = mean_squared_error(self.y_test[name], self.test_pred[name][model])

            self.y_predict_rescaled(model,name,n_test)

            self.test_dates[name][model] = self.dates[-n_test:]

            self.models[name][model] = m
    
    def VisualizationRegression(self,
        model,
        plot: bool = False,
        logdiff: bool = True
    ):
        if model not in [LinearRegression,ARIMA]:
            raise TypeError("model must be SimpleRNN or LSTM")
        else:
            self.Visualization(model=model,plot=plot,logdiff=logdiff)

