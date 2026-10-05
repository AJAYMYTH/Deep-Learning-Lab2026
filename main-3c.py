import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(0)

# Initialize weights and biases
wh = np.random.rand(2,2)
bh = np.zeros((1,2))
wo = np.random.rand(2,1)
bo = np.zeros((1,1))

learning_rate = 0.1
losses = []

# Train for multiple iterations
for epoch in range(1000):
    # Forward pass
    hidden = sigmoid(np.dot(X, wh) + bh)
    output = sigmoid(np.dot(hidden, wo) + bo)

    # Mean Squared Error
    loss = np.mean((y - output) ** 2)
    losses.append(loss)

    # Backward pass
    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(wo.T) * sigmoid_derivative(hidden)

    # Update weights
    wo += hidden.T.dot(d_output) * learning_rate
    bo += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    wh += X.T.dot(d_hidden) * learning_rate
    bh += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Plot loss
plt.figure(figsize=(8,5))
plt.plot(range(1, 1001), losses)
plt.xlabel("Iterations / Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Loss Function over Multiple Iterations")
plt.grid(True)
plt.show()

print("Initial Loss:", round(losses[0], 4))
print("Final Loss:", round(losses[-1], 4))
