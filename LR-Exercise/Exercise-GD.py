import numpy as np
import pandas as pd 
from sklearn import linear_model
import math 

import pandas as pd

df = pd.read_csv('LR-Exercise/dataset/test_scores.csv')

# Gradient Descent m and b 
# y = mx + b

# some parameters
x = np.array(df.math)
y = np.array(df.cs)

# normalize x and y
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

# model coef_ and intercept_
def linear(x, y):
    x = x.reshape(-1,1)
    # find coefficient and intercept
    model = linear_model.LinearRegression()
    model.fit(x, y)

    m_l = model.coef_
    b_l = model.intercept_

    print('m {} and b {}'.format(m_l, b_l))
    return m_l, b_l

# using gradient descent m and b
def descent(x, y):
    # parameters
    m = b = 0.0
    N = x.shape[0]
    cost_prv = 0

    # hyper parameters
    learning_rate = 0.1

    # iteration loop
    for i in range(1000): 
        y_pred = m * x + b

        # calculate cost function
        # cost = 1/N * np.sum((y - y_pred)**2)
        cost = 1/N * np.sum((y - y_pred)**2)

        # calculate derivative respect m and b
        md = (-2/N) * np.sum(x * (y - y_pred))
        bd = (-2/N) * np.sum(y - y_pred)

        # find new m and b values using derivative
        m = m - learning_rate * md 
        b = b - learning_rate * bd

        # isclose in math
        if math.isclose(cost, cost_prv, rel_tol=1e-20):    
            break

        # prv cost 
        cost_prv = cost
        print ("m {}, b {}, cost {}, iteration {}".format(m, b, cost, i))

    return m, b
    
m, b = descent(x, y)
m_l, b_l = linear(x, y)
