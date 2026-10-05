import numpy as np 
import matplotlib.pyplot as plt

x = np.linspace(-10 , 10 , 100)

sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
ReLU = np.maximum(0, x)

plt.figure(figsize = (10,6))
plt.plot(x , sigmoid , label = 'Sigmoid' , color = 'blue')
plt.plot(x , tanh , label = 'Tanh' , color = 'green')
plt.plot(x , ReLU , label = 'ReLU' , color = 'red')
plt.title('Activation Functions : Sigmoid , Tanh , ReLU')
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.legend()
plt.grid()
plt.show()