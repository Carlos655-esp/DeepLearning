import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Generate dataset
x = np.linspace(-10, 10, 100)

# Compute outputs
sigmoid_output = sigmoid(x)
relu_output = relu(x)
tanh_output = tanh(x)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid_output, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, relu_output, label='ReLU', color='orange')
plt.title('ReLU Activation Function')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, tanh_output, label='Tanh', color='green')
plt.title('Tanh Activation Function')
plt.grid(True)

plt.tight_layout()
plt.show()