from model import *
from algorithm import *
from get_data import stock_prices, scaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define the optimizer
model.compile(optimizer='adam', loss=loss)

def create_dataset(data, timestep=1):
    X = []
    y = []

    for i in range(len(data) - timestep):
        X.append(data[i:i+timestep])
        y.append(data[i+timestep])
        print(i)

    return np.array(X), np.array(y)

X, y = create_dataset(stock_prices, timestep=timestep)

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

print(X_train.shape, y_train.shape)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Predict the future stock prices
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse transform the data
train_pred = scaler.inverse_transform(train_pred)
y_train = scaler.inverse_transform(y_train)
test_pred = scaler.inverse_transform(test_pred)
y_test = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(y_train, label='Actual Price')
plt.plot(train_pred, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price vs Time')
plt.legend()
plt.show()