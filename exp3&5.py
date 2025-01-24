import numpy as np

class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.output_error = y - output
        self.output_delta = self.output_error * self.sigmoid_derivative(output)
        
        self.a1_error = self.output_delta.dot(self.W2.T)
        self.a1_delta = self.a1_error * self.sigmoid_derivative(self.a1)
        
        self.W2 += self.a1.T.dot(self.output_delta) * self.learning_rate
        self.b2 += np.sum(self.output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(self.a1_delta) * self.learning_rate
        self.b1 += np.sum(self.a1_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        output = self.forward(X)
        return np.round(output)

# Example dataset (XOR)
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

# Initialize the neural network
nn = TwoLayerNN(input_size=2, hidden_size=2, output_size=1)
# Train the neural network
nn.train(X, y)

# Test the neural network
print("Testing XOR")
for x in X:
    print(f"Input: {x}, Predicted Output: {nn.predict(x)}")