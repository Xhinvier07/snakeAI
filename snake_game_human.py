import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 5

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # Load images
        self.head_img = pygame.image.load('Graphics/head_right.png')
        self.head_img = pygame.transform.scale(self.head_img, (BLOCK_SIZE, BLOCK_SIZE))
        
        # Load body sprites
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png')
        self.body_vertical = pygame.image.load('Graphics/body_vertical.png')
        self.body_tl = pygame.image.load('Graphics/body_topleft.png')
        self.body_tr = pygame.image.load('Graphics/body_topright.png')
        self.body_bl = pygame.image.load('Graphics/body_bottomleft.png')
        self.body_br = pygame.image.load('Graphics/body_bottomright.png')
        
        # Scale body sprites
        self.body_horizontal = pygame.transform.scale(self.body_horizontal, (BLOCK_SIZE, BLOCK_SIZE))
        self.body_vertical = pygame.transform.scale(self.body_vertical, (BLOCK_SIZE, BLOCK_SIZE))
        self.body_tl = pygame.transform.scale(self.body_tl, (BLOCK_SIZE, BLOCK_SIZE))
        self.body_tr = pygame.transform.scale(self.body_tr, (BLOCK_SIZE, BLOCK_SIZE))
        self.body_bl = pygame.transform.scale(self.body_bl, (BLOCK_SIZE, BLOCK_SIZE))
        self.body_br = pygame.transform.scale(self.body_br, (BLOCK_SIZE, BLOCK_SIZE))
        
        # Load tail sprites
        self.tail_up = pygame.image.load('Graphics/tail_down.png')
        self.tail_down = pygame.image.load('Graphics/tail_up.png')
        self.tail_left = pygame.image.load('Graphics/tail_right.png')
        self.tail_right = pygame.image.load('Graphics/tail_left.png')
        
        # Scale tail sprites
        self.tail_up = pygame.transform.scale(self.tail_up, (BLOCK_SIZE, BLOCK_SIZE))
        self.tail_down = pygame.transform.scale(self.tail_down, (BLOCK_SIZE, BLOCK_SIZE))
        self.tail_left = pygame.transform.scale(self.tail_left, (BLOCK_SIZE, BLOCK_SIZE))
        self.tail_right = pygame.transform.scale(self.tail_right, (BLOCK_SIZE, BLOCK_SIZE))
        self.tail_img = pygame.image.load('Graphics/tail_right.png')
        self.tail_img = pygame.transform.scale(self.tail_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.apple_img = pygame.image.load('Graphics/apple.png')
        self.apple_img = pygame.transform.scale(self.apple_img, (BLOCK_SIZE, BLOCK_SIZE))
        
        # Store rotated versions of head
        self.head_rotations = {
            Direction.RIGHT: self.head_img,
            Direction.LEFT: pygame.transform.rotate(self.head_img, 180),
            Direction.UP: pygame.transform.rotate(self.head_img, 90),
            Direction.DOWN: pygame.transform.rotate(self.head_img, 270)
        }
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself
        if self.head in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        # Draw snake
        for i, pt in enumerate(self.snake):
            if i == 0:  # Head
                rotated_head = self.head_rotations[self.direction]
                self.display.blit(rotated_head, (pt.x, pt.y))
            elif i == len(self.snake) - 1:  # Tail
                self._draw_tail(pt, self.snake[-2])
            else:  # Body
                self._draw_body_segment(i, pt)
        
        # Draw apple
        self.display.blit(self.apple_img, (self.food.x, self.food.y))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _draw_tail(self, tail_pt, before_tail_pt):
        # Determine tail direction based on the segment before it
        if before_tail_pt.x < tail_pt.x:
            self.display.blit(self.tail_left, (tail_pt.x, tail_pt.y))
        elif before_tail_pt.x > tail_pt.x:
            self.display.blit(self.tail_right, (tail_pt.x, tail_pt.y))
        elif before_tail_pt.y < tail_pt.y:
            self.display.blit(self.tail_up, (tail_pt.x, tail_pt.y))
        elif before_tail_pt.y > tail_pt.y:
            self.display.blit(self.tail_down, (tail_pt.x, tail_pt.y))

    def _draw_body_segment(self, i, pt):
        prev_pt = self.snake[i - 1]
        next_pt = self.snake[i + 1]
        
        # Determine if it's a corner or straight segment
        if prev_pt.x == next_pt.x:  # Vertical
            self.display.blit(self.body_vertical, (pt.x, pt.y))
        elif prev_pt.y == next_pt.y:  # Horizontal
            self.display.blit(self.body_horizontal, (pt.x, pt.y))
        else:  # Corner
            if prev_pt.y > pt.y:  # Coming from bottom (moving up)
                if next_pt.x < pt.x:  # Going left
                    self.display.blit(self.body_bl, (pt.x, pt.y))  # moving left down
                else:  # Going right
                    self.display.blit(self.body_br, (pt.x, pt.y))  # moving right down
            elif prev_pt.y < pt.y:  # Coming from top (moving down)
                if next_pt.x < pt.x:  # Going left
                    self.display.blit(self.body_tl, (pt.x, pt.y))  # going left up
                else:  # Going right
                    self.display.blit(self.body_tr, (pt.x, pt.y))  # going right up
            elif prev_pt.x < pt.x:  # Coming from left
                if next_pt.y < pt.y:  # Going up
                    self.display.blit(self.body_tl, (pt.x, pt.y))  # moving down left
                else:  # Going down
                    self.display.blit(self.body_bl, (pt.x, pt.y))  # moving up left
            else:  # Coming from right
                if next_pt.y < pt.y:  # Going up
                    self.display.blit(self.body_tr, (pt.x, pt.y))  # moving down right
                else:  # Going down
                    self.display.blit(self.body_br, (pt.x, pt.y))  # bottomleft when moving left then down
        
        # Draw apple
        self.display.blit(self.apple_img, (self.food.x, self.food.y))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _get_rotation(self, current, previous):
        # Calculate the rotation angle based on the direction
        if previous.x < current.x:  # Moving left
            return 0
        elif previous.x > current.x:  # Moving right
            return 180
        elif previous.y < current.y:  # Moving up
            return 90
        elif previous.y > current.y:  # Moving down
            return 270
        return 0
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()