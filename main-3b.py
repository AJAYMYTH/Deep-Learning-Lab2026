import numpy as np

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input and Output (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Seed for reproducibility
np.random.seed(0)

# Initialize weights and biases
wh = np.random.rand(2, 2)
bh = np.zeros((1, 2))

wo = np.random.rand(2, 1)
bo = np.zeros((1, 1))

# Forward Pass
print("=== Forward Pass ===")

hidden_input = np.dot(X, wh) + bh
hidden_output = sigmoid(hidden_input)

print("Hidden Layer Output:")
print(np.round(hidden_output, 3))

final_input = np.dot(hidden_output, wo) + bo
final_output = sigmoid(final_input)

print("Final Output:")
print(np.round(final_output, 3))

# Backward Pass
print("\n=== Backward Pass ===")

error = y - final_output
print("Error:")
print(np.round(error, 3))

d_output = error * sigmoid_derivative(final_output)

error_hidden = d_output.dot(wo.T)
d_hidden = error_hidden * sigmoid_derivative(hidden_output)

# Update weights and biases
learning_rate = 0.1

wo += hidden_output.T.dot(d_output) * learning_rate
bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate

wh += X.T.dot(d_hidden) * learning_rate
bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("Weights and biases updated.")