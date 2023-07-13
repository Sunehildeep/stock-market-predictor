from model import *
from algorithm import *
from get_data import stock_prices
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices = scaler.fit_transform(stock_prices.reshape(-1, 1))

# Function to create the dataset
def create_dataset(dataset, past_history, forward_look):
    X, y = [], []
    for i in range(len(dataset) -past_history - 1):
        X.append(dataset[i:i+past_history, 0])
        y.append(dataset[i+past_history, 0])
    return np.array(X), np.array(y)

# Split the dataset into train and test sets
train, test = train_test_split(stock_prices, train_size=0.65, shuffle=False)

# Create the dataset
X_train, y_train = create_dataset(train, past_history, forward_look)
X_test, y_test = create_dataset(test, past_history, forward_look)

# Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Create the model
model = create_model((X_train.shape[1], 1))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

#Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Compile the model
model.compile(optimizer='adam', loss=loss)

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_schedule])

# Test the model
predictions = model.predict(X_test)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
plt.plot(scaler.inverse_transform(predictions.reshape(-1, 1)), label='Predicted')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price vs Time')
plt.legend()
plt.show()