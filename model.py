import torch
import torch.nn as nn #implement later yourself
import torch.optim as optim
import torch.nn.functional as F

# our class impements the base class of all nerural netword module
class Linear_QNet(nn.Module):

    # initializing a feed forward nueral network with 1 hidden layer
    # for our game the input depends on the state which has 8 parameter
    # the output depends on the no of actions which has 4 parameters
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, output_size)
        #self.linear3 = nn.Linear(hidden_size_2, output_size)

    # implementation of the forwarding function for each nueron
    def forward(self, input_for_input_layer):
        # applying the activation funtion on input
        activated_input_for_hidden_1 = F.relu(self.linear1(input_for_input_layer))
        #activated_input_for_hidden_2 = F.relu(self.linear2(activated_input_for_hidden_1))
        activated_input_for_output = self.linear2(activated_input_for_hidden_1)
        return activated_input_for_output

    def save(self, file='saved_model.pth'):
        torch.save(self.state_dict(), file)


class trainer():
    def __init__(self, model: Linear_QNet, learning_rate, gamma) -> None:
        self.learning_rate  = learning_rate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()



    #MAIN algorithm which implements reinforcement learning
    def train_step(self, state, action, reward, new_state, game_over):

        #convert the data to array format since we need it to work it with both since values and 
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        #if array already present in #(n, x) format

        # if only single value -> need to convert it to (n, x) format
        if len(state.shape) == 1:
            #appends one dimention at the begining
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            new_state = torch.unsqueeze(new_state,0)
            game_over = (game_over, )

        #Update rule implementation using bellman's equation

        #predicted Q value with current state
        predition = self.model(state)

        prediction_clone = predition.clone()

        for i in range(len(state)):
            q_new = reward[i]
            if not game_over[i]:
                q_new = reward[i] + (self.gamma * torch.max(self.model(new_state[i])))

            #TODO understand this logic
            prediction_clone[i][torch.argmax(action).item()] = q_new
        
        
        # TODO empties the gradients????
        self.optimizer.zero_grad()

        #difined the error as the squared difference between the new and the old q values
        loss = self.loss_function(prediction_clone, predition)
        
        #applies the backpropogation to update the weights
        loss.backward()


        self.optimizer.step()

        #new q value = r + gamma * max(next_predicted Q val)


        #










