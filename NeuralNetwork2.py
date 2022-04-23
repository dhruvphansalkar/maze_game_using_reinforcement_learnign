#function to create a layer, mention that the layer is hidden/output layer

#all layers should have their weights initialized

#how to backprop with the weights????

#8,64,64,4

#w1,w2,w3

import numpy as np

class NeuralNetwork2:
    def __init__(self, hidden_neuron = [128,64], activation='tanh', lr = 0.0001):
        self.num_hiddenlayers = 1
        if isinstance(hidden_neuron, int) or len(hidden_neuron)==1:
            self.num_hiddenlayers = 1
        elif len(hidden_neuron)==2:
            self.num_hiddenlayers = 2
        self.hidden_neuron = hidden_neuron
        self.lr = lr

        self.inputs = None
        if self.num_hiddenlayers == 2:
            self.w1 = np.random.randn(self.hidden_neuron[0], 8)
            self.b1 = np.random.randn(self.hidden_neuron[0], 1)

            self.w2 = np.random.randn(self.hidden_neuron[1], self.hidden_neuron[0])
            self.b2 = np.random.randn(self.hidden_neuron[1], 1)

            self.w3 = np.random.randn(4, self.hidden_neuron[1])
            self.b3 = np.random.randn(4, 1)

            self.z1 = None
            self.a1 = None
            self.z2 = None
            self.a2 = None
            self.z3 = None
            self.a3 = None
        else:
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

    def get_no_of_hiddenlayers(self): 
        return self.num_hiddenlayers

    def forward_prop(self, input):
        if self.num_hiddenlayers == 2:
            self.inputs = input
            self.inputs = self.inputs.T
            self.z1 = self.w1.dot(self.inputs) + self.b1
            self.a1 = self.activation(self.z1)
            self.z2 = self.w2.dot(self.a1) + self.b2
            self.a2 = self.softmax(self.z2)
            self.z3 = self.w3.dot(self.a2) + self.b3
            self.a3 = self.softmax(self.z3)
            return self.a3
        else:
            self.inputs = input
            self.inputs = self.inputs.T
            self.z1 = self.w1.dot(self.inputs) + self.b1
            self.a1 = self.activation(self.z1)
            self.z2 = self.w2.dot(self.a1) + self.b2
            self.a2 = self.softmax(self.z2)
            return self.a2

    def forward_prop_1(self, input):
        if self.num_hiddenlayers == 2:
            inputs = input.T
            z1 = self.w1.dot(inputs) + self.b1
            a1 = self.activation(z1)
            z2 = self.w2.dot(a1) + self.b2
            a2 = self.softmax(z2)
            z3 = self.w3.dot(a2) + self.b3
            a3 = self.softmax(z3)
            return a3
        else:
            inputs = input.T
            z1 = self.w1.dot(inputs) + self.b1
            a1 = self.activation(z1)
            z2 = self.w2.dot(a1) + self.b2
            a2 = self.softmax(z2)
            return a2

    def back_prop(self, expected_op, output):

        #calculate gradients
        m = expected_op.size
        if self.num_hiddenlayers == 2:
            dz3 = expected_op - output
            dw3 = 1/m * dz3.dot(self.a2.T)
            db3 = 1/m * np.sum(dz3)
            #second layer
            dz2 = self.w3.T.dot(dz3) * self.activation(self.z2, derivative=True)
            dw2 = 1/m * dz2.dot(self.a1.T)
            db2 = 1/m * np.sum(dz2)
            #third layer
            dz1 = self.w2.T.dot(dz2) * self.activation(self.z1, derivative=True)
            dw1 = 1 / m * dz1.dot(self.inputs.T)
            db1 = 1 / m * np.sum(dz1)

            #updating weights
            self.w1 += self.lr * dw1
            self.b1 += self.lr * db1    
            self.w2 += self.lr * dw2  
            self.b2 += self.lr * db2
            self.w3 += self.lr * dw3  
            self.b3 += self.lr * db3
        else:
            dz2 = expected_op - output
            dw2 = 1/m * dz2.dot(self.a1.T)
            db2 = 1/m * np.sum(dz2)
            dz1 = self.w2.T.dot(dz2) * self.activation(self.z1, derivative=True)
            dw1 = 1 / m * dz1.dot(self.inputs.T)
            db1 = 1 / m * np.sum(dz1)

            #updating weights
            self.w1 += self.lr * dw1
            self.b1 += self.lr * db1    
            self.w2 += self.lr * dw2  
            self.b2 += self.lr * db2
            
        
