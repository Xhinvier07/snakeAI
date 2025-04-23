import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from game import Direction, Point, BLOCK_SIZE, GRID_COLOR, font

pygame.init()

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

SCORE_BG = (1, 64, 5)
SCORE_TEXT = (255, 255, 255)

SPEED = 100

class SnakeTwoPlayerAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Battle')
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
        
        # Create two snake colors
        self.snake1_color = GREEN   # First snake
        self.snake2_color = PURPLE  # Second snake
        
        # Create head rotation dictionaries for both snakes
        self.head_rotations = {
            Direction.RIGHT: self.head_img,
            Direction.LEFT: pygame.transform.rotate(self.head_img, 180),
            Direction.UP: pygame.transform.rotate(self.head_img, 90),
            Direction.DOWN: pygame.transform.rotate(self.head_img, 270)
        }
        
        # Create colored versions of the head for snake2
        self.head_img2 = pygame.image.load('Graphics/head_right.png')
        self.head_img2 = pygame.transform.scale(self.head_img2, (BLOCK_SIZE, BLOCK_SIZE))
        # Apply tint to the second snake's head
        self.head_img2.fill((150, 0, 150), special_flags=pygame.BLEND_MULT)
        
        self.head_rotations2 = {
            Direction.RIGHT: self.head_img2,
            Direction.LEFT: pygame.transform.rotate(self.head_img2, 180),
            Direction.UP: pygame.transform.rotate(self.head_img2, 90),
            Direction.DOWN: pygame.transform.rotate(self.head_img2, 270)
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
            
    def _draw_scores(self):
        # Draw scores for both snakes
        # Snake 1 score (left)
        score_text1 = f'Snake 1: {self.score1}'
        text_surface1 = font.render(score_text1, True, SCORE_TEXT)
        text_rect1 = text_surface1.get_rect()
        
        panel_width1 = text_rect1.width + 20
        panel_height1 = text_rect1.height + 10
        panel_surface1 = pygame.Surface((panel_width1, panel_height1))
        panel_surface1.fill(self.snake1_color)
        
        pygame.draw.rect(panel_surface1, GRID_COLOR, (0, 0, panel_width1, panel_height1), 1)
        text_rect1.center = (panel_width1 // 2, panel_height1 // 2)
        panel_surface1.blit(text_surface1, text_rect1)
        
        # Snake 2 score (right)
        score_text2 = f'Snake 2: {self.score2}'
        text_surface2 = font.render(score_text2, True, SCORE_TEXT)
        text_rect2 = text_surface2.get_rect()
        
        panel_width2 = text_rect2.width + 20
        panel_height2 = text_rect2.height + 10
        panel_surface2 = pygame.Surface((panel_width2, panel_height2))
        panel_surface2.fill(self.snake2_color)
        
        pygame.draw.rect(panel_surface2, GRID_COLOR, (0, 0, panel_width2, panel_height2), 1)
        text_rect2.center = (panel_width2 // 2, panel_height2 // 2)
        panel_surface2.blit(text_surface2, text_rect2)
        
        # Position panels at top
        self.display.blit(panel_surface1, (10, 10))
        self.display.blit(panel_surface2, (self.w - panel_width2 - 10, 10))

    def _update_ui(self):
        # Draw background grid
        self.display.blit(self.grid_surface, (0, 0))
        
        # Draw snake 1
        self._draw_snake(self.snake1, self.direction1, self.head_rotations, 0)
        
        # Draw snake 2
        self._draw_snake(self.snake2, self.direction2, self.head_rotations2, 1)
        
        # Draw food
        self.display.blit(self.apple_img, (self.food.x, self.food.y))
        
        # Draw scores
        self._draw_scores()
        
        pygame.display.flip()
    
    def _draw_snake(self, snake, direction, head_rotations, snake_id):
        for i, pt in enumerate(snake):
            if i == 0:  # Head
                rotated_head = head_rotations[direction]
                self.display.blit(rotated_head, (pt.x, pt.y))
            elif i == len(snake) - 1:  # Tail
                if len(snake) > 1:
                    self._draw_tail(pt, snake[-2], snake_id)
            else:  # Body
                if len(snake) > 2:
                    self._draw_body_segment(i, pt, snake, snake_id)

    def _draw_tail(self, tail_pt, before_tail_pt, snake_id):
        # Choose the correct tail image based on direction
        if before_tail_pt.x < tail_pt.x:
            img = self.tail_left
        elif before_tail_pt.x > tail_pt.x:
            img = self.tail_right
        elif before_tail_pt.y < tail_pt.y:
            img = self.tail_up
        elif before_tail_pt.y > tail_pt.y:
            img = self.tail_down
            
        # Apply color tint for snake 2
        if snake_id == 1:
            img_copy = img.copy()
            img_copy.fill((150, 0, 150), special_flags=pygame.BLEND_MULT)
            self.display.blit(img_copy, (tail_pt.x, tail_pt.y))
        else:
            self.display.blit(img, (tail_pt.x, tail_pt.y))

    def _draw_body_segment(self, i, pt, snake, snake_id):
        prev_pt = snake[i - 1]
        next_pt = snake[i + 1]
        
        # Determine correct body segment image
        if prev_pt.x == next_pt.x:  # Vertical
            img = self.body_vertical
        elif prev_pt.y == next_pt.y:  # Horizontal
            img = self.body_horizontal
        else:  # Corner
            if prev_pt.y > pt.y:  # Coming from bottom (moving up)
                if next_pt.x < pt.x:  # Going left
                    img = self.body_bl
                else:  # Going right
                    img = self.body_br
            elif prev_pt.y < pt.y:  # Coming from top (moving down)
                if next_pt.x < pt.x:  # Going left
                    img = self.body_tl
                else:  # Going right
                    img = self.body_tr
            elif prev_pt.x < pt.x:  # Coming from left
                if next_pt.y < pt.y:  # Going up
                    img = self.body_tl
                else:  # Going down
                    img = self.body_bl
            else:  # Coming from right
                if next_pt.y < pt.y:  # Going up
                    img = self.body_tr
                else:  # Going down
                    img = self.body_br
        
        # Apply color tint for snake 2
        if snake_id == 1:
            img_copy = img.copy()
            img_copy.fill((150, 0, 150), special_flags=pygame.BLEND_MULT)
            self.display.blit(img_copy, (pt.x, pt.y))
        else:
            self.display.blit(img, (pt.x, pt.y))

    def reset(self):
        # Initialize snake 1 (left side)
        self.direction1 = Direction.RIGHT
        self.head1 = Point(self.w/4, self.h/2)
        self.snake1 = [self.head1,
                       Point(self.head1.x-BLOCK_SIZE, self.head1.y),
                       Point(self.head1.x-(2*BLOCK_SIZE), self.head1.y)]
        self.score1 = 0
        
        # Initialize snake 2 (right side)
        self.direction2 = Direction.LEFT
        self.head2 = Point(3*self.w/4, self.h/2)
        self.snake2 = [self.head2,
                       Point(self.head2.x+BLOCK_SIZE, self.head2.y),
                       Point(self.head2.x+(2*BLOCK_SIZE), self.head2.y)]
        self.score2 = 0
        
        # Place food
        self.food = None
        self._place_food()
        
        # Game state
        self.frame_iteration = 0
        self.food_iterations = 0
        self.game_over1 = False
        self.game_over2 = False

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x, y)
        
        # Make sure food isn't on either snake
        if self.food in self.snake1 or self.food in self.snake2:
            self._place_food()
        
        self.food_iterations = 0

    def play_step(self, action1, action2):
        self.frame_iteration += 1
        self.food_iterations += 1
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move both snakes
        reward1 = 0
        reward2 = 0
        
        # Move snake 1 if not game over
        if not self.game_over1:
            self._move(action1, 1)
            self.snake1.insert(0, self.head1)
        
        # Move snake 2 if not game over
        if not self.game_over2:
            self._move(action2, 2)
            self.snake2.insert(0, self.head2)
        
        # 3. check for collisions
        
        # Check snake 1 collisions
        if not self.game_over1:
            # Check if snake 1 hit wall or itself
            if self.is_collision(self.head1, self.snake1[1:]):
                self.game_over1 = True
                reward1 = -10
            
            # Check if snake 1 hit snake 2
            elif self.head1 in self.snake2:
                self.game_over1 = True
                reward1 = -10
                reward2 = 5  # Reward for snake 2 if snake 1 hits it
        
        # Check snake 2 collisions
        if not self.game_over2:
            # Check if snake 2 hit wall or itself
            if self.is_collision(self.head2, self.snake2[1:]):
                self.game_over2 = True
                reward2 = -10
            
            # Check if snake 2 hit snake 1
            elif self.head2 in self.snake1:
                self.game_over2 = True
                reward2 = -10
                reward1 = 5  # Reward for snake 1 if snake 2 hits it
        
        # Check for timeouts
        max_iterations = 100 * max(len(self.snake1), len(self.snake2))
        if self.frame_iteration > max_iterations:
            # Both snakes timeout
            if not self.game_over1:
                self.game_over1 = True
                reward1 = -5
            
            if not self.game_over2:
                self.game_over2 = True
                reward2 = -5
        
        # 4. Check food collisions
        
        # Snake 1 gets food
        if not self.game_over1 and self.head1 == self.food:
            self.score1 += 1
            reward1 = 10
            
            # Extra reward based on snake length and speed
            reward1 += len(self.snake1)
            if self.food_iterations < 50:
                reward1 += 5
                
            # Penalty for snake 2 for losing the food
            if not self.game_over2:
                reward2 = -2
                
            self._place_food()
        else:
            # Remove tail for snake 1 if didn't eat
            if not self.game_over1 and len(self.snake1) > 0:
                self.snake1.pop()
        
        # Snake 2 gets food
        if not self.game_over2 and self.head2 == self.food:
            self.score2 += 1
            reward2 = 10
            
            # Extra reward based on snake length and speed
            reward2 += len(self.snake2)
            if self.food_iterations < 50:
                reward2 += 5
                
            # Penalty for snake 1 for losing the food
            if not self.game_over1:
                reward1 = -2
                
            self._place_food()
        else:
            # Remove tail for snake 2 if didn't eat
            if not self.game_over2 and len(self.snake2) > 0:
                self.snake2.pop()
        
        # Distance-based rewards for active snakes
        if not self.game_over1:
            old_distance1 = abs(self.snake1[1].x - self.food.x) + abs(self.snake1[1].y - self.food.y) if len(self.snake1) > 1 else 0
            new_distance1 = abs(self.head1.x - self.food.x) + abs(self.head1.y - self.food.y)
            
            if old_distance1 > 0 and new_distance1 < old_distance1:
                reward1 += 0.5  # Moving towards food
            else:
                reward1 -= 0.2  # Moving away from food
                
        if not self.game_over2:
            old_distance2 = abs(self.snake2[1].x - self.food.x) + abs(self.snake2[1].y - self.food.y) if len(self.snake2) > 1 else 0
            new_distance2 = abs(self.head2.x - self.food.x) + abs(self.head2.y - self.food.y)
            
            if old_distance2 > 0 and new_distance2 < old_distance2:
                reward2 += 0.5  # Moving towards food
            else:
                reward2 -= 0.2  # Moving away from food
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. Check if game is completely over
        game_over = self.game_over1 and self.game_over2
        
        # 7. Return rewards and scores
        return reward1, reward2, game_over, self.score1, self.score2

    def is_collision(self, pt, snake_body=None):
        # Check for wall collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            
        # Check for body collision if body segments provided
        if snake_body and pt in snake_body:
            return True
            
        return False

    def _move(self, action, snake_id):
        # Determine which snake to move
        if snake_id == 1:
            direction = self.direction1
            head = self.head1
        else:
            direction = self.direction2
            head = self.head2
            
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        # Update direction
        if snake_id == 1:
            self.direction1 = new_dir
        else:
            self.direction2 = new_dir

        # Calculate new head position
        x = head.x
        y = head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE

        # Update head
        if snake_id == 1:
            self.head1 = Point(x, y)
        else:
            self.head2 = Point(x, y)

# Get state for two player game
def get_state_2p(game, snake_id):
    if snake_id == 1:
        head = game.snake1[0]
        snake = game.snake1
        direction = game.direction1
        opponent = game.snake2
    else:
        head = game.snake2[0]
        snake = game.snake2
        direction = game.direction2
        opponent = game.snake1
        
    point_l = Point(head.x - BLOCK_SIZE, head.y)
    point_r = Point(head.x + BLOCK_SIZE, head.y)
    point_u = Point(head.x, head.y - BLOCK_SIZE)
    point_d = Point(head.x, head.y + BLOCK_SIZE)
    
    # Diagonal points
    point_ul = Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)
    point_ur = Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE)
    point_dl = Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE)
    point_dr = Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE)
    
    dir_l = direction == Direction.LEFT
    dir_r = direction == Direction.RIGHT
    dir_u = direction == Direction.UP
    dir_d = direction == Direction.DOWN

    # Check collisions with both own body and opponent
    def check_collision(pt):
        # Wall collision
        if pt.x > game.w - BLOCK_SIZE or pt.x < 0 or pt.y > game.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Own body collision
        if pt in snake[1:]:
            return True
        # Opponent collision
        if pt in opponent:
            return True
        return False

    # Calculate distance to food
    food_dist_x = game.food.x - head.x
    food_dist_y = game.food.y - head.y
    
    # Calculate distance to opponent's head
    opponent_head = opponent[0] if opponent else Point(0, 0)
    opponent_dist_x = opponent_head.x - head.x
    opponent_dist_y = opponent_head.y - head.y

    state = [
        # Danger straight
        (dir_r and check_collision(point_r)) or 
        (dir_l and check_collision(point_l)) or 
        (dir_u and check_collision(point_u)) or 
        (dir_d and check_collision(point_d)),

        # Danger right
        (dir_u and check_collision(point_r)) or 
        (dir_d and check_collision(point_l)) or 
        (dir_l and check_collision(point_u)) or 
        (dir_r and check_collision(point_d)),

        # Danger left
        (dir_d and check_collision(point_r)) or 
        (dir_u and check_collision(point_l)) or 
        (dir_r and check_collision(point_u)) or 
        (dir_l and check_collision(point_d)),
        
        # Diagonal dangers
        check_collision(point_ul),
        check_collision(point_ur),
        check_collision(point_dl),
        check_collision(point_dr),
        
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # Food location 
        game.food.x < head.x,  # food left
        game.food.x > head.x,  # food right
        game.food.y < head.y,  # food up
        game.food.y > head.y,  # food down
        
        # Additional state information
        food_dist_x / game.w,  # Normalized distance to food X
        food_dist_y / game.h,  # Normalized distance to food Y
        
        # Opponent information
        opponent_dist_x / game.w,  # Normalized distance to opponent X
        opponent_dist_y / game.h,  # Normalized distance to opponent Y
        len(opponent) / (game.w * game.h / (BLOCK_SIZE * BLOCK_SIZE)),  # Opponent length
        len(snake) / (game.w * game.h / (BLOCK_SIZE * BLOCK_SIZE)),  # Own length
        ]

    return np.array(state, dtype=float) 