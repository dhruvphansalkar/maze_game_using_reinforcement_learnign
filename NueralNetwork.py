import numpy as np
# transposing -> p = p.T

class NeuralNetwork:
    def __init__(self, hidden_neuron = 128, activation='tanh', lr = 0.001):
        
        self.hidden_neuron = hidden_neuron
        self.lr = lr

        self.inputs = None
        self.w1 = np.random.randn(self.hidden_neuron, 8)
        self.b1 = np.random.randn(self.hidden_neuron, 1)

        self.w2 = np.random.randn(4, self.hidden_neuron)
        self.b2 = np.random.randn(4, 1)

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

        if activation == 'sigmoid':
            self.activation = self.sigmoid
        
        if activation == 'relu':
            self.activation = self.relu
        
        if activation == 'tanh':
            self.activation = self.htan

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        else:
            return 1/(1 + np.exp(-x))

    def relu(self, x, derivative=False) -> int:
        if not derivative:
            return np.maximum(x, 0)
        else:
            return (x > 0)

    def htan(self, x, derivative=False):
        if not derivative:
            return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        else:
            return 1 - self.htan(x, derivative=False) * self.htan(x, derivative=False)

    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    def forward_prop(self, input):
        self.inputs = input
        self.inputs = self.inputs.T
        self.z1 = self.w1.dot(self.inputs) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = self.activation(self.z2)
        return self.a2

    def forward_prop_1(self, input):
        inputs = input.T
        z1 = self.w1.dot(inputs) + self.b1
        a1 = self.activation(z1)
        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)
        return a2

    def back_prop(self, expected_op, output):
        m = expected_op.size
        dz2 = expected_op - output
        dw2 = 1/m * dz2.dot(self.a1.T)
        db2 = 1/m * np.sum(dz2)
        dz1 = self.w2.T.dot(dz2) * self.activation(self.z1, derivative=True)
        dw1 = 1 / m * dz1.dot(self.inputs.T)
        db1 = 1 / m * np.sum(dz1)
        self.update_params(dw1, db1, dw2, db2)
        
    def update_params(self, dw1, db1, dw2, db2):
        self.w1 = self.w1 + self.lr * dw1
        self.b1 = self.b1 + self.lr * db1    
        self.w2 = self.w2 + self.lr * dw2  
        self.b2 = self.b2 + self.lr * db2    
        