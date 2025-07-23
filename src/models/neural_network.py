import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size=4):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, 1))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        dZ2 = self.A2 - y.reshape(-1, 1)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)
        
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, learning_rate)
    
    def predict(self, X):
        y_pred = self.forward(X)
        return (y_pred)