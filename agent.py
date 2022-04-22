import random
import numpy as np
from maze_game_ml import maze_game, Point, BLOCK
from plotter import plot
from NueralNetwork import NeuralNetwork
from NeuralNetwork2 import NeuralNetwork2

from collections import deque


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MEMORY_SIZE = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.01
RANDOM_GAME_THRESHOLD = 200
TRAINING_THRESHOLD = 0 # incease this if training is set 

class Agent:

    def __init__(self) -> None:
        #self.model = NeuralNetwork(lr= LEARNING_RATE, hidden_neuron=256, activation='tanh')
        self.model = NeuralNetwork2(lr= LEARNING_RATE, hidden_neuron=[256,128], activation='tanh')
        self.no_of_games = 0
        self.random_action_flag = 0 # controls the random behaviour
        self.gamma = 0.8 # discount rate
        self.memory = deque()
        self.trainer = trainer(self.model, gamma=self.gamma)

    
    def get_state(self, game: maze_game):
        """
        gets the current state of the game
        contains danger areas with reference to the position of the protagonist
        containd location of the treasure with reference to the protagonist

        :param game: pass instance of maze_game
        :return: returns a numpy array of the state of the game consisting of 0 or 1 values
        """ 

        protagonist = game.protagonist
        # this checks the position of the treasure with respect to the protagonist
        if game.treasure.x < protagonist.x:
            treasure_left = 1
        else:
            treasure_left = 0

        if game.treasure.x > protagonist.x:
            treasure_right = 1
        else:
            treasure_right = 0

        if game.treasure.y < protagonist.y:
            treasure_above = 1
        else:
            treasure_above = 0

        if game.treasure.y > protagonist.y:
            treasure_below = 1
        else:
            treasure_below = 0

        # this checks if there are danger areas in the immidiate vicinity of the protagonist
        if game.collision(Point(protagonist.x, protagonist.y - BLOCK)):
            danger_above = 1
        else:
            danger_above = 0

        if game.collision(Point(protagonist.x + BLOCK, protagonist.y)):
            danger_right = 1
        else:
            danger_right = 0

        if game.collision(Point(protagonist.x, protagonist.y + BLOCK)):
            danger_below = 1
        else:
            danger_below = 0

        if game.collision(Point(protagonist.x - BLOCK, protagonist.y)):
            danger_Left = 1
        else:
            danger_Left = 0

        state = [
            # state of danger areas with respect to protagonist
            danger_above,
            danger_right,
            danger_below,
            danger_Left,
            #position of treasure with respect to the protagonist
            treasure_left,  #treasure to the left
            treasure_right,  #treasure to the right
            treasure_above,  #treasure above
            treasure_below   #treasure below

        ]

        # this is the input of the 
        return np.array(state)

    def LM_train(self):
        """
        trains with all the past states and actions
        """ 
        if len(self.memory) <= BATCH_SIZE:
            sample = self.memory
        else:
            sample = random.sample(self.memory, BATCH_SIZE)
        for i in range(len(sample)):
            state, action, reward, new_state, game_over = sample[i]
            self.trainer.training(state, action, reward, new_state, game_over)

    def train_and_store(self, state, action, reward, new_state, game_over):
        """
         trains with the last states and action and stores result for LM_train
        """
        self.trainer.training(state, action, reward, new_state, game_over)
        if len(self.memory) > MEMORY_SIZE:
            self.memory.popleft()
        self.memory.append((state, action, reward, new_state, game_over))

    def get_action(self, state):
        """
        gets the action to be performed by the protagonist
        """
        action = [0,0,0,0]

        #when no of games exceeds RANDOM_GAME_THRESHOLD -> 0 chance of random action
        # first we do some random moves - exploration v exploitation
        # the more games we play -> chances of random action decreases
        if random.randint(0,100) < RANDOM_GAME_THRESHOLD - self.no_of_games:
            action[random.randint(0,3)] = 1
        else:
            model_action = self.model.forward_prop(np.expand_dims(state, axis=0))
            action[np.argmax(model_action).item()] = 1
        return action


class trainer():
    def __init__(self, model: NeuralNetwork, gamma) -> None:
        self.gamma = gamma
        self.model = model
        self.error = None
        self.a = []
    #MAIN algorithm which implements reinforcement learning
    def training(self, state, action, reward, new_state, game_over):
        
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



def start_training():
    scores = []
    avg_scores = []
    total_score = 0
    max_score = 0

    #10000 games will be played and the best model will be created
    while(agent.no_of_games <= 500):
        
        state = agent.get_state(game)
        print(state)
        # decide action
        action = agent.get_action(state)
        print(action)
        #preform move
        game_output = game.play_step(action)
        game_over = game_output[0]
        score = game_output[1]
        no_of_moves = game_output[2]
        reward = game_output[3]
        
        new_state = agent.get_state(game)
        #train with state and store result in memory
        agent.train_and_store(state, action, reward, new_state, game_over)

        #if game is over train long memory, ie trains again on all the moves and games it has played before
        if game_over:
            game.restart()
            agent.no_of_games = agent.no_of_games + 1
            agent.LM_train()

            if score >= max_score and agent.no_of_games > TRAINING_THRESHOLD:
                max_score = score
                # saving the model which gets us the best performance
                w1 = agent.model.w1.copy()
                w2 = agent.model.w2.copy()
                b1 = agent.model.b1.copy()
                b2 = agent.model.b2.copy()

            scores.append(score)
            total_score += score

            avg_scores.append(total_score/agent.no_of_games)
            plot(scores, avg_scores, 'AI TRAINING')
    
    agent.model.w1 = w1
    agent.model.w2 = w2
    agent.model.b1 = b1
    agent.model.b2 = b2

def start_test():
    scores = []
    total_score = 0
    while(agent.no_of_games <= 1000):
        state = agent.get_state(game)
        action = agent.get_action(state)
        game_over, score, _, _ = game.play_step(action)
        if game_over:
            game.restart()
            agent.no_of_games += 1

            scores.append(score)
            total_score += score
            plot(scores, [], 'AI Playing Using Fixed Model')
    print(total_score/500)

if __name__ == '__main__':
    agent = Agent()
    game = maze_game()
    start_training()
    start_test()