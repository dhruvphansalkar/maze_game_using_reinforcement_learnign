import os
import pygame
import random
from collections import namedtuple
from enum import Enum

class DIRECTION(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

#named tuple to represent location
Point = namedtuple('Point', 'x, y')

BLOCK_SIZE = 20

RED = (255,0,0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0,0,0)

pygame.init()

#get font to display info
font = pygame.font.Font(None, 20)

mario = pygame.image.load(os.path.join('mario.png'))
mario = pygame.transform.scale(mario, (BLOCK_SIZE, BLOCK_SIZE))

fire = pygame.image.load(os.path.join('fire.png'))
fire = pygame.transform.scale(fire, (BLOCK_SIZE, BLOCK_SIZE))

treasure = pygame.image.load(os.path.join('treasure.png'))
treasure = pygame.transform.scale(treasure, (BLOCK_SIZE, BLOCK_SIZE))


class maze_game:

    # constructor
    def __init__(self, width = 240, height = 200) -> None:

        self.width = width
        self.height = height

        #create the screen
        self.display = pygame.display.set_mode((self.width, self.height))

        #create clock to control the speed
        self.clock = pygame.time.Clock()

        self.fire_pits = []
        self.create_firepits()
        
        self.protagonist = Point(self.width/2, self.height/2)

        self.moves = 0
        self.score = 0


        self.treasure = None

        self.place_treasure()


    #create firepits
    def create_firepits(self):

        y = BLOCK_SIZE * 3
        for x_index in range(4) :
            self.fire_pits.append(Point(x_index*BLOCK_SIZE, y))

        x = BLOCK_SIZE * 8
        for y_index in range(5, 10) :
            self.fire_pits.append(Point(x, y_index*BLOCK_SIZE))


    
    #randomly places treasure at the start of the game
    def place_treasure(self):
        x = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.treasure = Point(x,y)
        if(self.treasure == self.protagonist or self.treasure in self.fire_pits) :
            self.place_treasure()
        
    # this is the main method controlling the functionality of the game
    def play_step(self):
        # get input from user
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.move(DIRECTION.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.move(DIRECTION.RIGHT)
                elif event.key == pygame.K_UP:
                    self.move(DIRECTION.UP)
                elif event.key == pygame.K_DOWN:
                    self.move(DIRECTION.DOWN)

        #check if game over
        game_over = False
        if self.collision():
            game_over = True
            return game_over, self.score, self.moves

        #check if treasure found
        if self.protagonist == self.treasure:
            self.score += 1
            self.place_treasure()
        #place new treasure (optional)


        #update score and clock
        self.update()
        self.clock.tick(30)

        # return game over
        return game_over, self.score, self.moves

    #checks for collisions
    def collision(self):
        #check if it hits the boundry
        if self.protagonist.x > self.width - BLOCK_SIZE or self.protagonist.x < 0 or self.protagonist.y > self.height - BLOCK_SIZE or self.protagonist.y < 0:
            return True

        #check if it hits the fire pits
        if self.protagonist in self.fire_pits:
            return True
        return False
    
    # moves the protagonist based on user input
    def move(self, direction):

        x = self.protagonist.x
        y = self.protagonist.y

        if direction == DIRECTION.RIGHT:
            x += BLOCK_SIZE
        elif direction == DIRECTION.LEFT:
            x -= BLOCK_SIZE
        elif direction == DIRECTION.UP:
            y -= BLOCK_SIZE
        elif direction == DIRECTION.DOWN:
            y += BLOCK_SIZE

        self.protagonist = Point(x,y)
        self.moves += 1


    # draws the protagonist at the new location at the begining after user input
    def update(self):
        # set the backround
        self.display.fill(BLACK)

        #draw the protagonist
        #pygame.draw.rect(self.display, GREEN, pygame.Rect(self.protagonist.x, self.protagonist.y, BLOCK_SIZE, BLOCK_SIZE))
        self.display.blit(mario, (self.protagonist.x,self.protagonist.y))
        
        #draw treasure
        #pygame.draw.rect(self.display, BLUE, pygame.Rect(self.treasure.x, self.treasure.y, BLOCK_SIZE, BLOCK_SIZE))
        self.display.blit(treasure, (self.treasure.x, self.treasure.y))

        #draw firepits
        for point in self.fire_pits:
            #pygame.draw.rect(self.display, RED, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            self.display.blit(fire, (point.x, point.y))

        #display information
        text = font.render('Moves: ' + str(self.moves) + ' Score: ' + str(self.score), True, WHITE)
        self.display.blit(text, [0,0])

        # visually updates the changes
        pygame.display.flip()

if __name__ == '__main__':
    
    game = maze_game()
    # check for game over
    while True:
        game.play_step()
        game_over, score, moves = game.play_step()
        #break if game over
        if game_over == True:
            break
    pygame.quit()