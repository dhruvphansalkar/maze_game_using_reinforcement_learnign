import random
import numpy as np
from maze_game_ml import maze_game, Point, BLOCK_SIZE
from model import trainer
from plotter import plot
from NueralNetwork import NeuralNetwork


from collections import deque


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MEMORY_SIZE = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.00025
RANDOM_GAME_THRESHOLD = 200

class Agent:

    def __init__(self) -> None:
        self.model = NeuralNetwork(lr= LEARNING_RATE, hidden_neuron=256, activation='tanh')
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
            treasure_left = True
        else:
            treasure_left = False

        if game.treasure.x > protagonist.x:
            treasure_right = True
        else:
            treasure_right = False

        if game.treasure.y < protagonist.y:
            treasure_above = True
        else:
            treasure_above = False

        if game.treasure.y > protagonist.y:
            treasure_below = True
        else:
            treasure_below = False

        # this checks if there are danger areas in the immidiate vicinity of the protagonist
        if game.collision(Point(protagonist.x, protagonist.y - BLOCK_SIZE)):
            danger_above = True
        else:
            danger_above = False

        if game.collision(Point(protagonist.x + BLOCK_SIZE, protagonist.y)):
            danger_right = True
        else:
            danger_right = False

        if game.collision(Point(protagonist.x, protagonist.y + BLOCK_SIZE)):
            danger_below = True
        else:
            danger_below = False

        if game.collision(Point(protagonist.x - BLOCK_SIZE, protagonist.y)):
            danger_Left = True
        else:
            danger_Left = False

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
        return np.array(state, dtype=int)

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
            self.trainer.train_step(state, action, reward, new_state, game_over)

    def train_and_store(self, state, action, reward, new_state, game_over):
        """
         trains with the last states and action and stores result for LM_train
        """
        self.trainer.train_step(state, action, reward, new_state, game_over)
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
        randomVal = random.randint(0,100)
        if randomVal < RANDOM_GAME_THRESHOLD - self.no_of_games:
            action[random.randint(0,3)] = 1
        else:
            model_action = self.model.forward_prop(np.expand_dims(state, axis=0))
            action[np.argmax(model_action).item()] = 1
        return action



def start_training():
    scores = []
    avg_scores = []
    total_score = 0
    max_score = 0

    #10000 games will be played and the best model will be created
    while(agent.no_of_games <= 500):
        
        state = agent.get_state(game)
        # decide action
        action = agent.get_action(state)
        #preform move
        game_over, score, no_of_moves, reward = game.play_step(action)
        new_state = agent.get_state(game)
        #train with state and store result in memory
        agent.train_and_store(state, action, reward, new_state, game_over)

        #if game is over train long memory, ie trains again on all the moves and games it has played before
        if game_over:
            game.restart()
            agent.no_of_games += 1
            agent.LM_train()

            if score >= max_score:
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