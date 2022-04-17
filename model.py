import torch
import torch.nn as nn #implement later yourself
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from NueralNetwork import NeuralNetwork

# our class impements the base class of all nerural netword module
'''class Linear_QNet(nn.Module):

    # initializing a feed forward nueral network with 1 hidden layer
    # for our game the input depends on the state which has 8 parameter
    # the output depends on the no of actions which has 4 parameters
    def __init__(self, nueron_tuple):
        super().__init__()

        # fixed earlier problem caused due to using regular list to store layers
        self.layers = nn.ModuleList()
        for layer_index in range(len(nueron_tuple) - 1): #0, 1, 2
            self.layers.append(nn.Linear(nueron_tuple[layer_index], nueron_tuple[layer_index + 1]))

    # implementation of the forwarding function for each nueron
    def forward(self, input_for_layer):
        for layer in self.layers[:-1]:
            input_for_layer = F.relu(layer(input_for_layer))
        output = self.layers[-1](input_for_layer)
        return output

    def save(self, file='saved_model.pth'):
        torch.save(self.state_dict(), file)'''


class trainer():
    def __init__(self, model: NeuralNetwork, learning_rate, gamma) -> None:
        self.learning_rate  = learning_rate
        self.gamma = gamma
        self.model = model
        #self.optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        #self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        #self.loss_function = nn.MSELoss()

    #MAIN algorithm which implements reinforcement learning
    def train_step(self, state, action, reward, new_state, game_over):
        #convert the data to array format since we need it to work it with both single values and 
        
        state = np.expand_dims(state, axis=0)
        action = np.expand_dims(np.array(action), axis=0)
        #reward = torch.tensor(reward, dtype=torch.float)
        new_state = np.expand_dims(new_state, axis=0)
        
        #if array already present in #(n, x) format

        # if only single value -> need to convert it to (n, x) format
        '''if len(state.shape) == 1:
            #appends one dimention at the begining
            state = np.expand_dims(state, axis=0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            new_state = np.expand_dims(new_state, axis=0)
            game_over = (game_over, )'''

        #Update rule implementation using bellman's equation
        #predicted Q value with current state
        prediction = self.model.forward_prop(state)
        print(prediction)

        prediction_clone = prediction.copy()

        #for i in range(len(state)):
        q_new = reward
        if not game_over:
        #new q value = r + gamma * max(next_predicted Q val) - bellman equation
            q_new = reward + (self.gamma * np.max(self.model.forward_prop_1(new_state)))

        prediction_clone[np.argmax(action).item()] = q_new
        

        #self.optimizer.zero_grad()

        #difined the error as the squared difference between the new and the old q values
        #loss = self.loss_function(prediction_clone, prediction)

        #applies the backpropogation to update the weights
        self.model.back_prop(prediction_clone, prediction)
        #self.optimizer.step()










