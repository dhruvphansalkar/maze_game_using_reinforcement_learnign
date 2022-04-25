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

    
    def __init__(self) -> None:
        self.width = 240
        self.height = 200

        
        self.display = pygame.display.set_mode((self.width, self.height))

        
        self.clock = pygame.time.Clock()
        self.restart()

    
    def restart(self) :
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


    
    
    def place_treasure(self):
        self.treasure = Point(random.randint(0, 11) * BLOCK,random.randint(0, 9) * BLOCK)
        
        if(self.treasure == self.protagonist or self.treasure in self.fire_pits) :
            self.place_treasure()
        
    def play_step(self, action):

        
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
        
        
        reward = 0
        game_over = False

        #this prevents a lot of steps being taken without reaching the reward
        if self.collision() or self.moves > MAX_MOVES_WITHOUT_REWARD:
            reward = -10
            game_over = True
            return game_over, self.score, self.moves, reward

        
        if self.protagonist == self.treasure:
            self.score += 1
            reward = 10
            self.moves = 0
            self.place_treasure()


        
        self.update()
        self.clock.tick(SPEED)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        
        return game_over, self.score, self.moves, reward

    
    def collision(self, pt = None):
        pt = self.protagonist
        
        if pt.x > self.width - BLOCK:
            return True
        elif pt.x < 0:
            return True
        elif pt.y > self.height - BLOCK:
            return True
        elif pt.y < 0:
            return True

        
        if pt in self.fire_pits:
            return True
        return False

    
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
        
        text = font.render('Moves: ' + str(self.moves) + ' Score: ' + str(self.score), True, (255,255,255))
        self.display.blit(text, [0,0])

        
        pygame.display.flip()
