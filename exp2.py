import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        self.W = np.zeros(input_size + 1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.W.T.dot(np.insert(x, 0, 1))
        return self.activation_fn(z)

    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.activation_fn(self.W.T.dot(x))
                self.W = self.W + self.learning_rate * (d[i] - y) * x

# AND example dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
d = np.array([0, 0, 0, 1])

# Initialize the Perceptron
perceptron = Perceptron(input_size=2)
# Train the Perceptron
perceptron.fit(X, d)

# Test the Perceptron
print("Testing AND")
for x in X:
    print(f"Input: {x}, Predicted Output: {perceptron.predict(x)}")