import numpy as np
from NueralNetwork import NeuralNetwork

class trainer():
    def __init__(self, model: NeuralNetwork, gamma) -> None:
        self.gamma = gamma
        self.model = model
        self.error = None
        self.a = []
    #MAIN algorithm which implements reinforcement learning
    def train_step(self, state, action, reward, new_state, game_over):
        
        #convert the data to array format since we need it to work it with single values
        state = np.expand_dims(state, axis=0)
        action = np.expand_dims(np.array(action), axis=0)
        new_state = np.expand_dims(new_state, axis=0)
        
        #predicted Q value with current state
        prediction = self.model.forward_prop(state)
        print(prediction)
        #modify it to get target value
        prediction_clone = prediction.copy()

        if game_over:
            q_new = reward
        else:
            #new q value = r + gamma * max(next_predicted Q val) - bellman equation
            max_future_q_value = np.max(self.model.forward_prop_1(new_state))
            q_new = reward + (self.gamma * max_future_q_value)

        index = np.argmax(action).item()
        prediction_clone[index] = q_new
        
        # back propogation and weight updation
        self.model.back_prop(prediction_clone, prediction)










