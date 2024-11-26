from BaseEnv import BaseClass
import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, LSTM
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


class RNNClass(BaseClass):

    def __init__(self,
        feature_steps: int = 10,
        target_steps: int = 1,
        scale: bool = False,
        batchnormalization: bool = False,
        dropout: bool = False
    ):  
        super().__init__(feature_steps = feature_steps, target_steps = target_steps)
        self.models_name_str = {SimpleRNN: "SimpleRNN", LSTM: "LSTM"}
        self.train_series = {}
        self.train_pred = {}
        self.valid_pred = {}
        self.test_pred = {}
        self.train_errors = {}
        self.valid_errors = {}
        self.test_errors = {}
        self.y_pred_prc = {}
        self.y_test_prc = {}
        self.restored_prices = {}
        self.test_dates = {}
        for name in self.tickers.groups.keys():
            self.train_series[name] = np.concatenate( (self.y_train[name],self.y_valid[name],self.y_test[name]), axis=0)
            self.train_pred[name] = {SimpleRNN: [], LSTM: []}
            self.valid_pred[name] = {SimpleRNN: [], LSTM: []}
            self.test_pred[name] = {SimpleRNN: [], LSTM: []}
            self.train_errors[name] = {SimpleRNN: [], LSTM: []}
            self.valid_errors[name] = {SimpleRNN: [], LSTM: []}
            self.test_errors[name] = {SimpleRNN: [], LSTM: []}
            self.y_pred_prc[name] = {SimpleRNN: [], LSTM: []}
            self.y_test_prc[name] = {SimpleRNN: [], LSTM: []}
            self.restored_prices[name] = {SimpleRNN: [], LSTM: []}
            self.test_dates[name] = {SimpleRNN: [], LSTM: []}

    def Prediction(self,
        model
    ):
        for name in self.tickers.groups.keys():
            self.reset_session()
            n_train = len(self.X_train[name])
            n_valid = len(self.X_valid[name])
            n_test = len(self.X_test[name])

            if model in [SimpleRNN, LSTM]:

                m = tf.keras.models.Sequential([
                     model(20, return_sequences=True,
                                            dropout=0.1, recurrent_dropout=0.1,
                                            input_shape=[None, 1]),
                     model(20, return_sequences=True,
                                           dropout=0.1, recurrent_dropout=0.1),
                     tf.keras.layers.BatchNormalization(),
                     tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
                     tf.keras.layers.Lambda(lambda Y_pred: Y_pred[:, -12:])
                 ])

                m.compile(loss="mse", optimizer="nadam")
            else:
                raise TypeError("model must be SimpleRNN or LSTM")
            
            early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=200,
                                                              min_delta=0.01,
                                                              restore_best_weights=True)

            run = m.fit(self.X_train[name][..., np.newaxis], self.y_train[name][..., np.newaxis], epochs=200,
                        validation_data=(self.X_valid[name][..., np.newaxis], self.y_valid[name][..., np.newaxis]),
                        callbacks=[early_stopping_cb], verbose=0)
            
            pd.DataFrame(run.history).iloc[-11:]

            #self.train_pred[name][model] = m.predict(self.X_train[name])
            #self.valid_pred[name][model] = m.predict(self.X_valid[name])
            #self.test_pred[name][model] = m.predict(self.X_test[name])
            #self.train_pred[name][model] = m.predict(self.X_train[name][..., np.newaxis]).flatten()
            #self.valid_pred[name][model] = m.predict(self.X_valid[name][..., np.newaxis]).flatten()
            #self.test_pred[name][model] = m.predict(self.X_test[name][..., np.newaxis]).flatten()
            
            train_pred = m.predict(self.X_train[name])
            valid_pred = m.predict(self.X_valid[name])
            test_pred = m.predict(self.X_test[name])


            self.train_pred[name][model] = train_pred[:, -1, 0].flatten()  
            self.valid_pred[name][model] = valid_pred[:, -1, 0].flatten()
            self.test_pred[name][model] = test_pred[:, -1, 0].flatten()

            self.train_errors[name][model] = mean_squared_error(self.y_train[name], self.train_pred[name][model])
            self.valid_errors[name][model] = mean_squared_error(self.y_valid[name], self.valid_pred[name][model])
            self.test_errors[name][model] = mean_squared_error(self.y_test[name], self.test_pred[name][model])

            self.y_pred_prc[name][model] = [np.exp(self.test_pred[name][model][i])*self.prc[name][-n_test:][i] for i in range(n_test)]
            self.y_test_prc[name][model] = self.prc[name][-n_test:]
            self.test_dates[name][model] = self.dates[-n_test:]

    def reset_session(self,
            seed=42
        ):
            tf.random.set_seed(seed)
            np.random.seed(seed)
            tf.keras.backend.clear_session()

    def Visualization(self,
        model,
        plot: bool = False
    ):
        if model not in [SimpleRNN,LSTM]:
            raise TypeError("model must be SimpleRNN or LSTM")
        else:
            if plot:
                for name in self.tickers.groups.keys():
                    plt.figure(figsize=(10, 5))
                    plt.plot(self.test_dates[name][model], self.y_test[name], label="Actual")
                    plt.plot(self.test_dates[name][model], self.test_pred[name][model], label="Predicted")
                    plt.title(f"{self.models_name_str[model]}: Predicted vs Actual log difference on test dataset for {name}")
                    plt.xlabel("Time Steps")
                    plt.ylabel("Price")
                    plt.legend()
                    plt.show()

            print(f"{self.models_name_str[model]}: mean Squared Error for each ticker:")
            for name in self.tickers.groups.keys():
                print(f"{name}: Train MSE = {self.train_errors[name][model]:.4f}, Valid MSE = {self.valid_errors[name][model]:.4f}, Test MSE = {self.test_errors[name][model]:.4f}")