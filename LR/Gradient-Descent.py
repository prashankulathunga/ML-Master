import numpy as np
import random 

# yhat = mx + c
# loss = (y - yhat)**2 / N

# Initialize some parameters
x = np.random.randn(10,1)
y = 2*x + np.random.rand()

# Initialize parameters
m_curr = c_curr = 0.0
N = x.shape[0]

# Initialize hyperparameters 
learning_rate = 0.001


# create descent function
# need to calculate m and c new

def descend (x, y, N, m_curr, c_curr, learning_rate):
    # need to calculate cost function
    y_pred = m_curr * x + c_curr
    cost = np.mean((y - y_pred) ** 2)

    # Derivatives
    dm = (-2 / N) * np.sum(x * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)

    # Update parameters
    m_curr = m_curr - learning_rate * dm
    c_curr = c_curr - learning_rate * dc

    return cost, m_curr, c_curr

# create gradient descent function
for epoch in range(1000):
    cost, m_curr, c_curr = descend(x, y, N, m_curr, c_curr, learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Cost={cost:.4f}, m={m_curr:.4f}, c={c_curr:.4f}")

