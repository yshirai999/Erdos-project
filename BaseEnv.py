import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt


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
        for name, data in self.tickers:
            self.prc[name] = data['PRC'].values
            data = np.log(data[['PRC']]).diff().dropna()
            self.ts[name] = data.values#.flatten()
            self.X[name], self.y[name] = self.ts_split(self.ts[name])
            self.split_ind[name] = int(self.X[name].shape[0]*0.8)
            self.X_train_full[name], self.y_train_full[name] = self.X[name][:self.split_ind[name]], self.y[name][:self.split_ind[name]]
            self.X_test[name], self.y_test[name] = self.X[name][self.split_ind[name]:], self.y[name][self.split_ind[name]:]
            self.split_ind_2[name] = int(self.X_train_full[name].shape[0]*0.8)
            self.X_train[name], self.y_train[name] = self.X_train_full[name][:self.split_ind_2[name]], self.y_train_full[name][:self.split_ind_2[name]]
            self.X_valid[name], self.y_valid[name] = self.X_train_full[name][self.split_ind_2[name]:], self.y_train_full[name][self.split_ind_2[name]:]
    
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