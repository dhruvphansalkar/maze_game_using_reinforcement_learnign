import pygame
import random
from collections import namedtuple
import numpy as np
import os

#named tuple to represent location
Point = namedtuple('Point', 'x, y')

BLOCK = 20

MAX_MOVES_WITHOUT_REWARD = 100;

SPEED = 60 #reduce to make slower

pygame.init()

#get font to display info
font = pygame.font.Font(None, 20)

mario = pygame.image.load(os.path.join('mario.png'))
mario = pygame.transform.scale(mario, (BLOCK, BLOCK))

fire = pygame.image.load(os.path.join('fire.png'))
fire = pygame.transform.scale(fire, (BLOCK, BLOCK))

treasure = pygame.image.load(os.path.join('treasure.png'))
treasure = pygame.transform.scale(treasure, (BLOCK, BLOCK))

class maze_game:

    # constructor
    def __init__(self) -> None:
        self.width = 240
        self.height = 200

        #create the screen
        self.display = pygame.display.set_mode((self.width, self.height))

        #create clock to control the speed
        self.clock = pygame.time.Clock()
        self.restart()

    
    def restart(self) :
        #initialize the game state
        self.fire_pits = []
        self.create_firepits()
        
        self.protagonist = Point(self.width/2, self.height/2)

        self.moves = 0
        self.score = 0


        self.treasure = None
        self.place_treasure()


    #create firepits
    def create_firepits(self):

        y = BLOCK * 3
        for x_index in range(4) :
            self.fire_pits.append(Point(x_index*BLOCK, y))

        x = BLOCK * 8
        for y_index in range(5, 10) :
            self.fire_pits.append(Point(x, y_index*BLOCK))


    
    #randomly places treasure at the start of the game
    def place_treasure(self):
        self.treasure = Point(random.randint(0, 11) * BLOCK,random.randint(0, 9) * BLOCK)
        
        if(self.treasure == self.protagonist or self.treasure in self.fire_pits) :
            self.place_treasure()
        
    # this is the main method controlling the functionality of the game
    def play_step(self, action):

        # move the character based on action   
        #1,0,0,0 - up
        #0,1,0,0, - right
        #0,0,1,0 - left
        #0,0,0,1 - down
        x = self.protagonist.x
        y = self.protagonist.y
        if np.array_equal(action, [1,0,0,0]):
            y = y - BLOCK
        elif np.array_equal(action, [0,1,0,0]):
            x = x + BLOCK
        elif np.array_equal(action, [0,0,1,0]):
            y = y + BLOCK
        elif np.array_equal(action, [0,0,0,1]):
            x = x - BLOCK
        self.protagonist = Point(x,y)
        self.moves += 1
        
        #check if game over
        reward = 0
        game_over = False

        #this prevents a lot of steps being taken without reaching the reward
        if self.collision() or self.moves > MAX_MOVES_WITHOUT_REWARD:
            reward = -10
            game_over = True
            return game_over, self.score, self.moves, reward

        #check if treasure found
        if self.protagonist == self.treasure:
            self.score += 1
            reward = 10
            self.moves = 0
            self.place_treasure()


        #update score and clock
        self.update()
        self.clock.tick(SPEED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # return game over
        return game_over, self.score, self.moves, reward

    #checks for collisions
    def collision(self, pt = None):
        pt = self.protagonist
        #check if it hits the boundry
        if pt.x > self.width - BLOCK:
            return True
        elif pt.x < 0:
            return True
        elif pt.y > self.height - BLOCK:
            return True
        elif pt.y < 0:
            return True

        #check if it hits the fire pits
        if pt in self.fire_pits:
            return True
        return False

    # draws the protagonist at the new location at the begining after user input
    def update(self):
        # set the backround
        self.display.fill((0,0,0))
        #draw the protagonist
        self.display.blit(mario, (self.protagonist.x,self.protagonist.y))
        #draw treasure
        self.display.blit(treasure, (self.treasure.x, self.treasure.y))
        #draw firepits
        for point in self.fire_pits:
            #pygame.draw.rect(self.display, RED, pygame.Rect(point.x, point.y, BLOCK, BLOCK))
            self.display.blit(fire, (point.x, point.y))
        #display information
        text = font.render('Moves: ' + str(self.moves) + ' Score: ' + str(self.score), True, (255,255,255))
        self.display.blit(text, [0,0])

        # visually updates the changes
        pygame.display.flip()
