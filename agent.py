import torch
import random
import numpy as np
from maze_game_ml import maze_game, Point, BLOCK_SIZE
from model import Linear_QNet, trainer
from plotter import plot


# can also be done using lists
from collections import deque


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MEMORY_SIZE = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.01
RANDOM_GAME_THRESHOLD = 100

class Agent:

    def __init__(self) -> None:
        self.no_of_games = 0
        self.random_action_flag = 0 # controls the random behaviour
        self.gamma = 0.8 # discount rate
        self.memory = deque(maxlen= MEMORY_SIZE)

        self.model = Linear_QNet((8, 128, 4,))
        self.trainer = trainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    
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
        


    def store_in_memory(self, state, action, reward, new_state, game_over):
        """
        stores the action performed in deque memory
        if MEMORY becomes full it automatically pops the oldest value
        :param state: state before the action was performed
        :param action: action that was performed based on current state
        :param reward: reward associated with the action
        :param new_state: state after performing the action
        :game_over: protagonist dead
        """ 
        self.memory.append((state, action, reward, new_state, game_over))

    def LM_train(self):
        """
        trains with all the past states and actions
        """ 
        if len(self.memory) > BATCH_SIZE:
            sample = random.sample(self.memory, BATCH_SIZE)
        else:
            sample = self.memory

        states, actions, rewards, new_states, game_overs = zip(*sample)
        self.trainer.train_step(states, actions, rewards, new_states, game_overs)



    def SM_train(self, state, action, reward, new_state, game_over):
        """
         trains with all the past states and actions
        """
        self.trainer.train_step(state, action, reward, new_state, game_over)

    def get_action(self, state):
        """
        gets the action to be performed by the protagonist
        """
        # first we do some random moves - exploration v exploitation
        # the more games we play -> chances of random action decreases
        # inccrease RANDOM_GAME_THRESHOLD if model is not getting sufficiently trained in those number of games
        self.random_action_flag = RANDOM_GAME_THRESHOLD - self.no_of_games
        action = [0,0,0,0]

        #when no of games exceeds 50 -> 0 chance of random action
        randomVal = random.randint(0,100)
        if randomVal < self.random_action_flag:
            index = random.randint(0,3)
            action[index] = 1
        else:
            temp = torch.tensor(state, dtype=torch.float)
            model_action = self.model(temp)
            index = torch.argmax(model_action).item()
            action[index] = 1
        return action



def start_training():
    scores = []
    avg_scores = []
    total_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = maze_game()

    #1000 games will be played and the best model will be created
    for i in range(1000):
        
        state = agent.get_state(game)
        # decide action
        action = agent.get_action(state)
        #preform move
        game_over, score, no_of_moves, reward = game.play_step(action)
        new_state = agent.get_state(game)
        
        #train short memory
        agent.SM_train(state, action, reward, new_state, game_over)

        #strore in memory
        agent.store_in_memory(state, action, reward, new_state, game_over)

        #if game is over train long memory, ie trains again on all the moves and games it has played before
        if game_over:
            game.restart()
            agent.no_of_games += 1
            agent.LM_train()

            # can be removed if not required
            if score > record:
                record = score
                agent.model.save()

            scores.append(score)
            total_score += score
            avg_scores.append(total_score/agent.no_of_games)
            plot(scores, avg_scores)



    

if __name__ == '__main__':
    start_training()