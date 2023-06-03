import numpy as np


class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        self.momentum_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.momentum_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        self.activation = activation

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x

    def activation_prime(self, x):
        if self.activation == 'sigmoid':
            return self.activate(x) * (1 - self.activate(x))
        elif self.activation == 'relu':
            return (x > 0) * 1.0
        else:
            return 1.0

    def forward(self, inputs):
        self.inputs = inputs
        self.raw_output = np.dot(inputs, self.weights) + self.biases
        self.output = self.activate(self.raw_output)
        return self.output

    def backward(self, grad_output, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grad_output *= self.activation_prime(self.raw_output)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)

        self.momentum_weights = beta1 * self.momentum_weights + (1 - beta1) * grad_weights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (grad_weights ** 2)

        self.momentum_biases = beta1 * self.momentum_biases + (1 - beta1) * grad_biases
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (grad_biases ** 2)

        m_weights_hat = self.momentum_weights / (1 - beta1 ** t)
        v_weights_hat = self.v_weights / (1 - beta2 ** t)

        m_biases_hat = self.momentum_biases / (1 - beta1 ** t)
        v_biases_hat = self.v_biases / (1 - beta2 ** t)

        self.weights -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
        self.biases -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input


class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad_output, learning_rate, t):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate, t)

#Testing

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# One hot encoding
encoder = OneHotEncoder(sparse=False)
Y = Y.reshape(-1, 1)
Y = encoder.fit_transform(Y)

# Split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize inputs
X_train = X_train / np.max(X_train, axis=0)
X_test = X_test / np.max(X_test, axis=0)

# Initialize network
network = DenseNetwork()
network.add_layer(DenseLayer(input_size=4, output_size=16, activation='relu'))
network.add_layer(DenseLayer(input_size=16, output_size=3, activation='sigmoid'))

# Training settings
epochs = 500
learning_rate = 0.01

# Training loop
for epoch in range(epochs):
    # Forward pass
    output = network.forward(X_train)

    # Compute error
    error = output - Y_train
    loss = np.mean(error ** 2)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    # Backward pass
    network.backward(error, learning_rate, epoch + 1)

# Test
output = network.forward(X_test)
predictions = np.argmax(output, axis=1)
true_labels = np.argmax(Y_test, axis=1)
accuracy = accuracy_score(true_labels, predictions)
print(f'Test Accuracy: {accuracy * 100}%')