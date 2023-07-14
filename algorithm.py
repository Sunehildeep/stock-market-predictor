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
    if delta[-1] == 2:
        decision = 'sell'
    elif delta[-1] == 0:
        decision = 'hold'
    else:
        decision = 'buy'
    return decision
