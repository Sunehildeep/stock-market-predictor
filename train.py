from model import *
from algorithm import *
from get_data import stock_prices
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
stock_prices = scaler.fit_transform(stock_prices.reshape(-1, 1))

# Function to create the dataset
def create_dataset(dataset, past_history):
    X, y = [], []
    for i in range(len(dataset) -past_history - 1):
        X.append(dataset[i:i+past_history, 0])
        y.append(dataset[i+past_history, 0])
    return np.array(X), np.array(y)

# Split the dataset int
# o train and test sets
train_size = int(len(stock_prices) * 0.65)
test_size = len(stock_prices) - train_size
train, test = stock_prices[0:train_size,:], stock_prices[train_size:len(stock_prices),:]

# Create the dataset
X_train, y_train = create_dataset(train, past_history)
X_test, y_test = create_dataset(test, past_history)

# Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create the model
model = create_model((X_train.shape[1], 1))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

# Compile the model
model.compile(optimizer='adam', loss=loss)


# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

'''
Test if graph matches
'''
#Plot graph for x_test actual and predicted values
y_pred = model.predict(X_test)

#Dense gives out forward_look number of values
#Convert back to original scale
y_pred = scaler.inverse_transform(y_pred[:,:])

y_test = y_test[len(y_test)-forward_look:]

#Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(y_test)), scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual', color='blue')
plt.plot(np.arange(len(y_test)), y_pred, label='Predicted', color='orange')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price vs Time')
plt.legend()
plt.show()

'''
Future Prediction
'''
predictions = []

# Predict the next future 200 days
for i in range(forward_look):
    if i < X_test.shape[0]:
        y_pred = model.predict(X_test[i,:].reshape(1,X_test.shape[1],X_test.shape[2]))[0][0]
    else:
        # Use previous prediction as input for the next prediction
        y_pred = model.predict(X_test[-1,:].reshape(1,X_test.shape[1],X_test.shape[2]))[0][0]
        X_test = np.append(X_test, y_pred.reshape(1, 1, 1))
        X_test = X_test[1:].reshape(-1, past_history, 1)
    predictions.append(y_pred)

# Convert predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

#Get max and min values for the predictions
max_val = np.max(predictions)
min_val = np.min(predictions)

# Convert to /integer
max_val = np.round(max_val, 2)
min_val = np.round(min_val, 2)

max_val = int(max_val)
min_val = int(min_val)

print(f'\nMax value for the predictions: {max_val}')
print(f'Min value for the predictions: {min_val}')

decision = make_decision(predictions)

print(f'\nPredicted decision for {forward_look} days: {decision}')

# Reshape y_test to match predictions
y_test_reshaped = y_test.reshape(-1)
y_test_reshaped = y_test_reshaped[len(y_test_reshaped)-len(predictions):].reshape(-1, 1)

# Plot the predictions with previous forward_look days and next forward_look days
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(y_test_reshaped)), scaler.inverse_transform(y_test_reshaped).reshape(-1, 1), label='Actual', color='blue')
plt.plot(np.arange(len(y_test_reshaped), len(y_test_reshaped) + len(predictions)), predictions, label='Predicted', color='orange')
#Invert the predictions
# plt.plot(np.arange(len(y_test_reshaped), len(y_test_reshaped) + len(predictions)), -predictions, label='Inverted Predicted', color='green')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Stock Price vs Time')
plt.legend()
plt.show()
