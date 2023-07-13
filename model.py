import tensorflow as tf
from hyperparameters import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation
import numpy as np

'''
Single-layer model implementation for predicting future stock prices
'''

def create_model(shape):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=shape, return_sequences=True))
    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(LSTM(lstm_units))
    model.add(Dense(1))
    return model
    
'''
Loss Function - Mean Squared Error
'''
def loss(targets, predictions):
    # 1 + t/200
    # 1/m * sum(||y - y_hat||^2)
    # 1/m * sum(||w*(y - y_hat)||^2)
    # Here, w is weight, y is target, y_hat is prediction

    # Calculate the weighted mean squared error
    weights = np.float32(1.0 + 1.0*np.linspace(0,forward_look-1,forward_look)/200.0)
    return tf.math.reduce_mean(tf.math.square(tf.multiply(weights, targets - predictions)))
