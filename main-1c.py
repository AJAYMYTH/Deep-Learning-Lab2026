import matplotlib.pyplot as plt
import numpy as np

# 1. Create a random 5x5 matrix with values between 0 and 1
data = np.random.rand(5, 5)

# 2. Plot the heatmap
plt.imshow(data, cmap='viridis', interpolation='nearest')

# 3. Add a colorbar to show the scale
plt.colorbar()

# 4. Add titles and labels (optional but helpful)
plt.title('Random 5x5 Heatmap')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 5. Display the plot
plt.show()