import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

class BaseClass:

    def __init__(self,
    feature_steps: int = 10,
    target_steps: int = 1,
    ):
        self.df = pd.read_csv("Dataset.csv")
        self.tickers = self.df.groupby('Ticker')
        self.dates = self.df.date.unique()
        self.dates = [dt.datetime.strptime(d,'%m/%d/%Y').date() for d in self.dates]
        self.feature_steps = feature_steps
        self.target_steps = target_steps
        self.ts = {}
        self.X = {}
        self.y = {}
        self.X_train_full = {}
        self.y_train_full = {}
        self.X_test = {}
        self.y_test = {}
        self.X_train = {}
        self.y_train = {}
        self.X_valid = {}
        self.y_valid = {}
        self.split_ind = {}
        self.split_ind_2 = {}
        self.prc = {}
        self.scalers = {}
        self.test_pred_rescaled = {}
        self.y_test_rescaled = {}
        for name, data in self.tickers:
            self.prc[name] = data['PRC'].values
            data = np.log(data[['PRC']]).diff().dropna()
            self.scalers[name] = MinMaxScaler()
            data = self.scalers[name].fit_transform(data)
            self.ts[name] = data#.values#.flatten()
            self.X[name], self.y[name] = self.ts_split(self.ts[name])
            self.split_ind[name] = int(self.X[name].shape[0]*0.8)
            self.X_train_full[name], self.y_train_full[name] = self.X[name][:self.split_ind[name]], self.y[name][:self.split_ind[name]]
            self.X_test[name], self.y_test[name] = self.X[name][self.split_ind[name]:], self.y[name][self.split_ind[name]:]
            self.split_ind_2[name] = int(self.X_train_full[name].shape[0]*0.8)
            self.X_train[name], self.y_train[name] = self.X_train_full[name][:self.split_ind_2[name]], self.y_train_full[name][:self.split_ind_2[name]]
            self.X_valid[name], self.y_valid[name] = self.X_train_full[name][self.split_ind_2[name]:], self.y_train_full[name][self.split_ind_2[name]:]
            self.y_test_rescaled[name] = self.scalers[name].inverse_transform(self.y_test[name]).flatten()
    
    def visualizedata(self
    ):
        print(self.df)
        for name, data in self.tickers:
            plt.plot(self.dates, data.PRC.values, '-', label = name)
        plt.legend()
        print(self.X_train_full['SPY'][0])
        print(self.X_train_full['SPY'][1])
        print(self.y_train_full['SPY'][0])

    def ts_split(self,
        ts
    ):
        feature_steps = self.feature_steps
        target_steps = self.target_steps
        n_obs = len(ts) - feature_steps - target_steps + 1
        X = np.array([ts[idx:idx + feature_steps].flatten() for idx in range(n_obs)])
        y = np.array([ts[idx + feature_steps:idx + feature_steps + target_steps][:, -1]
                    for idx in range(n_obs)])
        return X, y
    
    def Visualization(self,
        model,
        plot: bool = False,
        logdiff: bool = True
    ):
        try:
            if plot:
                for name in self.tickers.groups.keys():
                    plt.figure(figsize=(10, 5))
                    if logdiff:
                        plt.plot(self.test_dates[name][model], self.y_test_rescaled[name], label="Actual")
                        plt.plot(self.test_dates[name][model], self.test_pred[name][model], label="Predicted")
                    else:
                        plt.plot(self.test_dates[name][model], self.y_test_prc[name][model], label="Actual")
                        plt.plot(self.test_dates[name][model], self.y_pred_prc[name][model], label="Predicted")
                    plt.title(f"{self.models_name_str[model]}: Predicted vs Actual log difference on test dataset for {name}")
                    plt.xlabel("Time Steps")
                    plt.ylabel("Price")
                    plt.legend()
                    plt.show()

            print(f"{self.models_name_str[model]}: mean Squared Error for each ticker:")
            for name in self.tickers.groups.keys():
                print(f"{name}: Train MSE = {self.train_errors[name][model]:.4f}, Valid MSE = {self.valid_errors[name][model]:.4f}, Test MSE = {self.test_errors[name][model]:.4f}")
        
        except:
            raise TypeError("model not trained")
        
    def y_predict_rescaled(self,
        model,
        name,
        n_test                   
    ):
        try:
            self.test_pred[name][model] = self.scalers[name].inverse_transform(self.test_pred[name][model]).flatten()
            self.y_pred_prc[name][model] = [np.exp(self.test_pred[name][model][i])*self.prc[name][-n_test:][i] for i in range(n_test)]
            self.y_test_prc[name][model] = self.prc[name][-n_test:]
        except:
            raise TypeError("model not trained")