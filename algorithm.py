import numpy as np

'''
StockBot decision making algorithm
Input: c - forecast from the model for forward_look days
'''
def make_decision(c):
    # Compute the nature of change
    # = sign(c(i+1) - c(i))
    change = np.sign(c[1:] - c[:-1])

    # Compute the curvature of the forecast
    # delta = change(i+1) - change(i)
    delta = change[1:] - change[:-1]

    # Compute the decision
    # delta = 2: sell
    # delta = 0: hold
    # delta = -2: buy
    decision = np.zeros(len(delta))

    # Change to string
    decision = decision.astype(str)
    decision[decision == '2'] = 'sell'
    decision[decision == '0'] = 'hold'
    decision[decision == '-2'] = 'buy'

    return decision