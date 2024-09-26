import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
def download_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df
def prepare_data(df, feature_col='Close', look_back=60):
    data = df[[feature_col]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    X, y = np.array(X), np.array(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  
    model.add(LSTM(units=50, return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
if __name__ == "__main__":
    ticker = 'AAPL'  
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    look_back = 60
    df = download_stock_data(ticker, start_date, end_date)
    X, y, scaler = prepare_data(df, look_back=look_back)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = create_lstm_model(input_shape=(look_back, 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(y_test):], y_test, color='blue', label='Actual Price')
    plt.plot(df.index[-len(predictions):], predictions, color='red', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.legend()
    plt.show()
