#!/usr/bin/env python
# coding: utf-8

# In[24]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
#!pip install tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Tesla stock data
tesla_data = pd.read_csv("/Users/pranup/Downloads/Tesla.csv")
tesla_data['Date'] = pd.to_datetime(tesla_data['Date'])
tesla_data.set_index('Date', inplace=True)

# Summary statistics and visualization
print(tesla_data.describe())

# Plot Closing Price and Volume
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(tesla_data['Close'], label='Closing Price', color='blue')
plt.title('Tesla Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(tesla_data['Volume'], label='Trading Volume', color='orange')
plt.title('Tesla Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.tight_layout()
plt.show()

# Moving averages (10, 20, 50-day)
for ma in [10, 20, 50]:
    tesla_data[f'MA_{ma}'] = tesla_data['Close'].rolling(window=ma).mean()

plt.figure(figsize=(15, 8))
plt.plot(tesla_data['Close'], label='Close', color='blue')
plt.plot(tesla_data['MA_10'], label='10-Day MA', color='red', linestyle='--')
plt.plot(tesla_data['MA_20'], label='20-Day MA', color='green', linestyle='--')
plt.plot(tesla_data['MA_50'], label='50-Day MA', color='orange', linestyle='--')
plt.title('Tesla Stock Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate daily returns
tesla_data['Daily_Return'] = tesla_data['Adj Close'].pct_change()

plt.figure(figsize=(15, 8))
plt.plot(tesla_data['Daily_Return'], label='Daily Returns', color='purple', linestyle='--')
plt.title('Tesla Stock Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return')
plt.legend()
plt.tight_layout()
plt.show()

# Correlation matrix
correlation_matrix = tesla_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# LSTM Model Preparation
data = tesla_data[['Adj Close']].dropna()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(data, time_step=60):
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

x_train, y_train = create_dataset(train_data)
x_test, y_test = create_dataset(test_data)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Align the valid DataFrame with predictions
train = tesla_data[:train_size]
valid = tesla_data[train_size + 60:]  # Start from 60 rows into the test set
valid['Predictions'] = predictions

# Visualize Predictions
plt.figure(figsize=(15, 8))
plt.plot(train['Adj Close'], label='Training Data')
plt.plot(valid['Adj Close'], label='Testing Data')
plt.plot(valid['Predictions'], label='Predictions', linestyle='--')
plt.title('Tesla Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




