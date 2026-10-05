import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input data (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target output
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Set seed
np.random.seed(1)

# Initialize weights and biases
wh = np.random.rand(2, 2)
bh = np.zeros((1, 2))

wo = np.random.rand(2, 1)
bo = np.zeros((1, 1))

# Training
for epoch in range(10000):

    # Forward Pass
    hidden_input = np.dot(X, wh) + bh
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, wo) + bo
    final_output = sigmoid(final_input)

    # Backward Pass
    error = y - final_output
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(wo.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    wo += hidden_output.T.dot(d_output)
    bo += np.sum(d_output, axis=0, keepdims=True)

    wh += X.T.dot(d_hidden)
    bh += np.sum(d_hidden, axis=0, keepdims=True)

# Display result
print("Output after training:")
print(np.round(final_output, 2))