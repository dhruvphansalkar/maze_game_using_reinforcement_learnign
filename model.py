import numpy as np
from NueralNetwork import NeuralNetwork

class trainer():
    def __init__(self, model: NeuralNetwork, gamma) -> None:
        self.gamma = gamma
        self.model = model

    #MAIN algorithm which implements reinforcement learning
    def train_step(self, state, action, reward, new_state, game_over):
        #convert the data to array format since we need it to work it with both single values and 
        
        state = np.expand_dims(state, axis=0)
        action = np.expand_dims(np.array(action), axis=0)
        #reward = torch.tensor(reward, dtype=torch.float)
        new_state = np.expand_dims(new_state, axis=0)
        
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
        
        # back propogation and weight updation
        self.model.back_prop(prediction_clone, prediction)










