import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math

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
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

BLOCK_SIZE = 20
SPEED = 120  # Increased speed for faster training

# Add these colors after the existing color definitions
GRID_COLOR = (16, 52, 10)  # Darker gray for grid
SCORE_BG = (1, 64, 5)    # Background for score
SCORE_TEXT = (255, 255, 255)  # White text

# Max steps without eating before timeout
MAX_STEPS_WITHOUT_FOOD = 300

class SnakeGameAI:

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
        self.apple_img = pygame.image.load('Graphics/apple.png')
        self.apple_img = pygame.transform.scale(self.apple_img, (BLOCK_SIZE, BLOCK_SIZE))
        
        # Store rotated versions of head
        self.head_rotations = {
            Direction.RIGHT: self.head_img,
            Direction.LEFT: pygame.transform.rotate(self.head_img, 180),
            Direction.UP: pygame.transform.rotate(self.head_img, 90),
            Direction.DOWN: pygame.transform.rotate(self.head_img, 270)
        }
        
        self.reset()
        
        # Add background grid texture
        self.grid_surface = pygame.Surface((self.w, self.h))
        self._create_grid_background()
        
    def _create_grid_background(self):
        self.grid_surface.fill(BLACK)
        
        # Draw vertical lines
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.grid_surface, GRID_COLOR, (x, 0), (x, self.h))
            
        # Draw horizontal lines
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.grid_surface, GRID_COLOR, (0, y), (self.w, y))
            
    def _draw_score(self):
        # Create score panel
        score_text = f'Score: {self.score}'
        text_surface = font.render(score_text, True, SCORE_TEXT)
        text_rect = text_surface.get_rect()
        
        # Create background panel for score
        panel_width = text_rect.width + 20
        panel_height = text_rect.height + 10
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.fill(SCORE_BG)
        
        # Add a border to the panel
        pygame.draw.rect(panel_surface, GRID_COLOR, (0, 0, panel_width, panel_height), 1)
        
        # Position text in panel
        text_rect.center = (panel_width // 2, panel_height // 2)
        panel_surface.blit(text_surface, text_rect)
        
        # Position panel in top-right corner with padding
        panel_x = self.w - panel_width - 10
        panel_y = 10
        self.display.blit(panel_surface, (panel_x, panel_y))

    def _update_ui(self):
        # Draw background grid
        self.display.blit(self.grid_surface, (0, 0))
        
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
        
        # Draw score
        self._draw_score()
        
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
                    self.display.blit(self.body_bl, (pt.x, pt.y))
                else:  # Going right
                    self.display.blit(self.body_br, (pt.x, pt.y))
            elif prev_pt.y < pt.y:  # Coming from top (moving down)
                if next_pt.x < pt.x:  # Going left
                    self.display.blit(self.body_tl, (pt.x, pt.y))
                else:  # Going right
                    self.display.blit(self.body_tr, (pt.x, pt.y))
            elif prev_pt.x < pt.x:  # Coming from left
                if next_pt.y < pt.y:  # Going up
                    self.display.blit(self.body_tl, (pt.x, pt.y))
                else:  # Going down
                    self.display.blit(self.body_bl, (pt.x, pt.y))
            else:  # Coming from right
                if next_pt.y < pt.y:  # Going up
                    self.display.blit(self.body_tr, (pt.x, pt.y))
                else:  # Going down
                    self.display.blit(self.body_br, (pt.x, pt.y))


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.food_iterations = 0  # Counter for steps without eating
        self.last_food_distance = 0  # To track if getting closer to food
        self.previous_positions = []  # Track recent positions for loop detection
        self.last_distances = []  # Track distance history for smarter rewards


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
        # Calculate initial distance to food for progress tracking
        self.last_food_distance = self._get_food_distance()
        self.food_iterations = 0  # Reset counter when placing new food


    def _get_food_distance(self):
        """Calculate the Manhattan distance to food"""
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    

    def _get_euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)


    def play_step(self, action):
        self.frame_iteration += 1
        self.food_iterations += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Track recent positions to detect loops (store only every few steps to save memory)
        if self.frame_iteration % 5 == 0:
            pos_hash = (self.head.x, self.head.y, self.direction.value)
            self.previous_positions.append(pos_hash)
            if len(self.previous_positions) > 20:  # Keep only recent positions
                self.previous_positions.pop(0)
        
        # 3. check if game over
        reward = 0
        game_over = False
        
        # Game over conditions
        if self.is_collision():
            game_over = True
            reward = -15  # Significant negative reward for collision
            return reward, game_over, self.score
        
        # Timeout condition - avoid endless games
        if self.food_iterations > MAX_STEPS_WITHOUT_FOOD:
            game_over = True
            reward = -10  # Penalize for not finding food in reasonable time
            return reward, game_over, self.score
            
        # Detect if snake is going in circles without progress
        if len(self.previous_positions) >= 20:
            unique_positions = len(set(self.previous_positions))
            if unique_positions <= 4:  # Snake is stuck in a small loop
                reward -= 1  # Small penalty for repetitive movement
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            
            # Progressive reward system - bigger reward for longer snake
            base_reward = 20  # Base reward for eating
            
            # Add bonus for faster food consumption
            time_bonus = max(0, 100 - self.food_iterations) / 10
            
            # Length bonus - more reward as snake gets longer
            length_bonus = min(self.score, 10)  # Cap at 10
            
            # Calculate total reward
            reward = base_reward + time_bonus + length_bonus
            
            # Place new food and reset counters
            self._place_food()
            self.previous_positions = []  # Reset position history after eating
            
        else:
            # Remove the tail
            self.snake.pop()
            
            # Calculate current distance to food
            current_distance = self._get_food_distance()
            
            # Track recent distances
            self.last_distances.append(current_distance)
            if len(self.last_distances) > 10:
                self.last_distances.pop(0)
            
            # Reward for moving toward food (using trend, not just single step)
            if len(self.last_distances) >= 3:
                avg_prev = sum(self.last_distances[:-3]) / max(1, len(self.last_distances[:-3]))
                avg_recent = sum(self.last_distances[-3:]) / 3
                
                if avg_recent < avg_prev:  # Trending closer to food
                    reward += 0.5
                elif avg_recent > avg_prev:  # Trending away from food
                    reward -= 0.2
            
            # Check for near-collision conditions - encourage path planning
            head_dangers = self._check_surrounding_dangers()
            if head_dangers >= 3:  # Snake has limited movement options
                reward -= 0.2  # Small penalty to encourage better planning
            
            # Progressive penalty for taking too long to find food
            if self.food_iterations > 50 and self.food_iterations % 10 == 0:
                reward -= 0.1 * (self.food_iterations // 50)  # Increasing penalty
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. return game over and score
        return reward, game_over, self.score


    def _check_surrounding_dangers(self):
        """Count how many adjacent cells contain danger (wall or body)"""
        directions = [
            Point(self.head.x + BLOCK_SIZE, self.head.y),  # right
            Point(self.head.x - BLOCK_SIZE, self.head.y),  # left
            Point(self.head.x, self.head.y + BLOCK_SIZE),  # down
            Point(self.head.x, self.head.y - BLOCK_SIZE)   # up
        ]
        
        danger_count = 0
        for point in directions:
            if self.is_collision(point):
                danger_count += 1
                
        return danger_count


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
        
    # New function to check proximity to danger
    def is_near_collision(self, pt, distance=1):
        # Check if point is near a wall
        if (pt.x < distance*BLOCK_SIZE or pt.x > self.w - distance*BLOCK_SIZE or 
            pt.y < distance*BLOCK_SIZE or pt.y > self.h - distance*BLOCK_SIZE):
            return True
            
        # Check if point is near snake body
        for segment in self.snake[1:]:
            if (abs(pt.x - segment.x) < distance*BLOCK_SIZE and 
                abs(pt.y - segment.y) < distance*BLOCK_SIZE):
                return True
                
        return False


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)