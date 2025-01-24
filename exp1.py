class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.activation_function(weighted_sum)

# Ejemplo
inputs = [0.5, -1.5, 2.0, 1.0]
weights = [0.8, -0.2, 0.5, 0.7]
bias = 0.1

neuron = Neuron(weights, bias)
output = neuron.predict(inputs)
print(f"Output: {output}")