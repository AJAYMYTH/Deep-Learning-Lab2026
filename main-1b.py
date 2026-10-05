import numpy as np

inputs = np.array([0.2,0.5,0.3])
weights = np.array([1.0,2.0,3.0])

result = np.dot(inputs, weights)
print("Inputs:", inputs)
print("Weights:", weights)
print("Dot product:", result)