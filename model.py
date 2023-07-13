import tensorflow as tf
from hyperparameters import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation

'''
Single-layer model implementation for predicting future stock prices
'''
# Input shape for Input layer: (None, 60, 2)
# Output shape for Input layer: (None, 2) 
# Next, LSTM layer input shape: (None, 60, 2)
# LSTM layer output shape: (None, 20)
# Dense layer input shape: (None, 20)
# Dense layer output shape: (None, 2)
model = Sequential()
model.add(LSTM(lstm_units,return_sequences=True,input_shape=(timestep,1)))
model.add(Dropout(dropout))
model.add(LSTM(lstm_units,return_sequences=True))
model.add(Dropout(dropout))
model.add(LSTM(lstm_units))
model.add(Dropout(dropout))
model.add(Dense(1))
    
'''
Loss Function - Mean Squared Error
'''
def loss(predictions, targets):
    # 1 + t/200
    # 1/m * sum(||y - y_hat||^2)
    # 1/m * sum(||w*(y - y_hat)||^2)
    # Here, w is weight, y is target, y_hat is prediction

    # Calculate the weighted mean squared error
    w = 1 + forward_look / 200
    squared_errors = tf.square(w * (targets - predictions))
    loss = tf.reduce_mean(squared_errors)

    # Normalize the loss by the number of samples in the batch
    batch_size = tf.shape(predictions)[0]
    loss = loss / tf.cast(batch_size, dtype=tf.float32)

    return loss
