import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer, ConvQNet
from helper import plot
import os
import math

# Increased memory for better experience replay
MAX_MEMORY = 200_000
BATCH_SIZE = 2000
LR = 0.0005  # Reduced learning rate for more stable learning

class Agent:

    def __init__(self, model_name='model.pth', use_conv_net=False):
        self.n_games = 0
        self.epsilon = 100  # Start with high exploration
        self.gamma = 0.99  # Higher discount rate for better long-term planning
        self.memory = deque(maxlen=MAX_MEMORY)
        self.use_conv_net = use_conv_net
        
        # Choose network architecture
        if use_conv_net:
            self.model = ConvQNet(18, 3)  # Using the convolutional network
            print("Using Convolutional Network")
        else:
            # Deeper network with wider hidden layers
            self.model = Linear_QNet(18, 1024, 3)
            print("Using Linear Network")
            
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.model_name = model_name
        self._load_model()
        
        # Track performance metrics
        self.last_scores = deque(maxlen=100)  # Recent scores for adaptive exploration
        self.total_rewards = 0

    def _load_model(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        model_file = os.path.join(model_folder_path, self.model_name)
        if os.path.exists(model_file):
            print(f"Loading existing model: {self.model_name}")
            self.model.load_state_dict(torch.load(model_file))
            self.model.eval()

    def get_state(self, game):
        head = game.snake[0]
        
        # Check in all 8 directions (N, NE, E, SE, S, SW, W, NW)
        # Create points for all directions
        points = [
            Point(head.x, head.y - BLOCK_SIZE),                # North
            Point(head.x + BLOCK_SIZE, head.y - BLOCK_SIZE),   # Northeast
            Point(head.x + BLOCK_SIZE, head.y),                # East
            Point(head.x + BLOCK_SIZE, head.y + BLOCK_SIZE),   # Southeast
            Point(head.x, head.y + BLOCK_SIZE),                # South
            Point(head.x - BLOCK_SIZE, head.y + BLOCK_SIZE),   # Southwest
            Point(head.x - BLOCK_SIZE, head.y),                # West
            Point(head.x - BLOCK_SIZE, head.y - BLOCK_SIZE)    # Northwest
        ]
        
        # Current direction one-hot encoding
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        # Calculate distance to food using Euclidean distance
        food_dist_x = game.food.x - head.x
        food_dist_y = game.food.y - head.y
        food_euclidean = math.sqrt(food_dist_x**2 + food_dist_y**2) / math.sqrt(game.w**2 + game.h**2)
        
        # Normalized coordinates of head and food
        head_x_norm = head.x / game.w
        head_y_norm = head.y / game.h
        food_x_norm = game.food.x / game.w
        food_y_norm = game.food.y / game.h
        
        # Create a comprehensive state vector
        state = []
        
        # Danger detection in all 8 directions (1 if danger, 0 if safe)
        for point in points:
            state.append(game.is_collision(point))
        
        # Current direction
        state.extend([dir_l, dir_r, dir_u, dir_d])
        
        # Food direction relative to head (binary signals)
        state.extend([
            food_dist_x < 0,  # food is to the left
            food_dist_x > 0,  # food is to the right
            food_dist_y < 0,  # food is above
            food_dist_y > 0   # food is below
        ])
        
        # Normalized position data
        state.extend([
            head_x_norm,      # normalized head x position
            head_y_norm,      # normalized head y position
            food_x_norm,      # normalized food x position
            food_y_norm,      # normalized food y position
            food_euclidean    # normalized euclidean distance to food
        ])
        
        # Snake information
        state.append(len(game.snake) / (game.w * game.h / (BLOCK_SIZE * BLOCK_SIZE)))  # Normalized snake length

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.total_rewards += reward

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            # Prioritize more recent experiences
            recent_size = min(BATCH_SIZE // 2, len(self.memory) // 4)
            recent_sample = list(self.memory)[-recent_size:]
            
            # Randomly sample from the rest of memory
            old_sample = random.sample(list(self.memory)[:-recent_size], BATCH_SIZE - recent_size)
            
            # Combine samples
            mini_sample = recent_sample + old_sample
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Adaptive exploration based on recent performance
        if len(self.last_scores) > 10:
            avg_score = sum(self.last_scores) / len(self.last_scores)
            # Increase exploration if performance is stagnating
            if avg_score < 5:
                self.epsilon = max(min(self.epsilon + 5, 100), 20)
            else:
                # Epsilon decay - gradually reduce exploration as we learn
                self.epsilon = max(self.epsilon * 0.99, 5)
        else:
            # Initial epsilon decay
            self.epsilon = max(100 - self.n_games * 0.5, 5)
        
        # Action decision
        final_move = [0, 0, 0]
        
        # Exploration phase - higher chance early on
        if random.random() < self.epsilon / 100:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploitation phase - use the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            
            # Add a small amount of noise to predictions for exploration
            noise = torch.randn_like(prediction) * 0.05
            prediction = prediction + noise
            
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(agent_name='model.pth', use_conv_net=False):
    plot_scores = []
    plot_mean_scores = []
    plot_rewards = []
    total_score = 0
    record = 0
    agent = Agent(model_name=agent_name, use_conv_net=use_conv_net)
    game = SnakeGameAI()
    
    print("Starting training with enhanced learning...")
    print("Press Ctrl+C to stop training at any time. Model will be saved.")
    
    try:
        while True:
            # Get old state
            state_old = agent.get_state(game)

            # Get move
            final_move = agent.get_action(state_old)

            # Perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.last_scores.append(score)
                agent.train_long_memory()
                
                avg_reward = agent.total_rewards / max(1, agent.n_games)
                plot_rewards.append(avg_reward)

                if score > record:
                    record = score
                    agent.model.save(file_name=agent.model_name)
                    print(f'New high score! {score} - Saving model...')

                print(f'Game {agent.n_games}, Score {score}, Record {record}, Epsilon {agent.epsilon:.1f}, Avg Reward {avg_reward:.2f}')

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
                
                # Save periodically regardless of record
                if agent.n_games % 50 == 0:
                    backup_name = f"{agent.model_name.split('.')[0]}_backup.pth"
                    agent.model.save(file_name=backup_name)
                    print(f"Periodic save: {backup_name}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save the model when interrupted
        agent.model.save(file_name=agent.model_name)
        print(f"Model saved as {agent.model_name}")


if __name__ == '__main__':
    use_conv = input("Use convolutional network? (y/n): ").lower() == 'y'
    model_name = input("Enter model name (default: model.pth): ")
    if not model_name:
        model_name = "model.pth"
    
    train(model_name, use_conv_net=use_conv)