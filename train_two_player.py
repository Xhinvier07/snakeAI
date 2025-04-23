import torch
import random
import numpy as np
from collections import deque
from two_player_game import SnakeTwoPlayerAI, get_state_2p
from model import Linear_QNet, QTrainer
from helper import plot
import os
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class CompetitiveAgent:
    def __init__(self, model_name='snake_agent.pth', is_competitor=False):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.95  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        # Larger state size for competitive play (includes opponent info)
        self.model = Linear_QNet(21, 512, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.model_name = model_name
        self.is_competitor = is_competitor
        self._load_model()
        
    def _load_model(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        model_file = os.path.join(model_folder_path, self.model_name)
        if os.path.exists(model_file):
            print(f"Loading existing model: {self.model_name}")
            self.model.load_state_dict(torch.load(model_file))
            self.model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
    def get_action(self, state):
        # Dynamic epsilon calculation for exploration vs exploitation
        if not self.is_competitor:
            self.epsilon = max(80 - self.n_games, 0)
        else:
            # Competitor can use a fixed strategy or a smaller epsilon
            self.epsilon = max(20 - self.n_games, 0)
            
        final_move = [0, 0, 0]
        
        # Random moves for exploration
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Predicted moves based on model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move
        
    def save_model(self):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
            
        file_name = os.path.join(model_folder_path, self.model_name)
        torch.save(self.model.state_dict(), file_name)


def train_competitive():
    # Create two agents
    agent1 = CompetitiveAgent(model_name='snake_agent1.pth')
    agent2 = CompetitiveAgent(model_name='snake_agent2.pth')
    
    # Create the game
    game = SnakeTwoPlayerAI()
    
    # Training metrics
    plot_scores1 = []
    plot_scores2 = []
    plot_mean_scores1 = []
    plot_mean_scores2 = []
    total_score1 = 0
    total_score2 = 0
    record1 = 0
    record2 = 0
    games_played = 0
    
    # Main training loop
    while True:
        # Get current states for both agents
        state1 = get_state_2p(game, 1)
        state2 = get_state_2p(game, 2)
        
        # Get actions based on current states
        action1 = agent1.get_action(state1)
        action2 = agent2.get_action(state2)
        
        # Execute actions and get rewards
        reward1, reward2, done, score1, score2 = game.play_step(action1, action2)
        
        # Get new states after actions
        new_state1 = get_state_2p(game, 1)
        new_state2 = get_state_2p(game, 2)
        
        # Train both agents on short memory
        agent1.train_short_memory(state1, action1, reward1, new_state1, done)
        agent2.train_short_memory(state2, action2, reward2, new_state2, done)
        
        # Remember experiences for replay memory
        agent1.remember(state1, action1, reward1, new_state1, done)
        agent2.remember(state2, action2, reward2, new_state2, done)
        
        # Game over - train on long memory and reset
        if done:
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1
            games_played += 1
            
            # Train long memory (experience replay)
            agent1.train_long_memory()
            agent2.train_long_memory()
            
            # Update records and save models if improved
            if score1 > record1:
                record1 = score1
                agent1.save_model()
                
            if score2 > record2:
                record2 = score2
                agent2.save_model()
                
            print(f'Game {games_played}, Agent1: {score1}, Agent2: {score2}, Records: [{record1}, {record2}]')
            
            # Update metrics
            plot_scores1.append(score1)
            plot_scores2.append(score2)
            total_score1 += score1
            total_score2 += score2
            mean_score1 = total_score1 / agent1.n_games
            mean_score2 = total_score2 / agent2.n_games
            plot_mean_scores1.append(mean_score1)
            plot_mean_scores2.append(mean_score2)
            
            # Plot training progress
            plot_competitive(plot_scores1, plot_scores2, plot_mean_scores1, plot_mean_scores2)


def plot_competitive(scores1, scores2, mean_scores1, mean_scores2):
    plt.clf()
    plt.title('Training Two Snake Agents...')
    plt.xlabel('Games')
    plt.ylabel('Score')
    plt.plot(scores1, color='green', label='Snake 1')
    plt.plot(scores2, color='purple', label='Snake 2')
    plt.plot(mean_scores1, color='blue', label='Mean Snake 1')
    plt.plot(mean_scores2, color='red', label='Mean Snake 2')
    plt.legend()
    plt.ylim(ymin=0)
    
    if len(scores1) > 0:
        plt.text(len(scores1)-1, scores1[-1], str(scores1[-1]))
    if len(scores2) > 0:
        plt.text(len(scores2)-1, scores2[-1], str(scores2[-1]))
    if len(mean_scores1) > 0:
        plt.text(len(mean_scores1)-1, mean_scores1[-1], str(round(mean_scores1[-1], 2)))
    if len(mean_scores2) > 0:
        plt.text(len(mean_scores2)-1, mean_scores2[-1], str(round(mean_scores2[-1], 2)))
        
    plt.savefig('./model/competitive_training_plot.png')
    plt.pause(.1)


def self_play_training(iterations=1000):
    """Training against a copy of itself to improve gameplay"""
    # Create the learning agent and a copy that follows the same policy
    main_agent = CompetitiveAgent(model_name='self_play_agent.pth')
    
    for i in range(iterations):
        # For each iteration, create a copy of the main agent
        copy_agent = CompetitiveAgent(model_name='self_play_agent.pth', is_competitor=True)
        
        # Create game instance
        game = SnakeTwoPlayerAI()
        done = False
        
        while not done:
            # Get states for both agents
            state1 = get_state_2p(game, 1)
            state2 = get_state_2p(game, 2)
            
            # Get actions from both agents
            action1 = main_agent.get_action(state1)
            action2 = copy_agent.get_action(state2)
            
            # Play game step
            reward1, reward2, done, score1, score2 = game.play_step(action1, action2)
            
            # Get new states
            new_state1 = get_state_2p(game, 1)
            new_state2 = get_state_2p(game, 2)
            
            # Train only the main agent
            main_agent.train_short_memory(state1, action1, reward1, new_state1, done)
            main_agent.remember(state1, action1, reward1, new_state1, done)
            
        # After each game, train long memory and save model
        print(f'Self-play iteration {i+1}/{iterations}, Score: {score1}')
        main_agent.train_long_memory()
        main_agent.save_model()
        
    print("Self-play training complete")


if __name__ == '__main__':
    # Choose which training mode to run
    training_mode = input("Select training mode (1=Competitive, 2=Self-play): ")
    
    if training_mode == "1":
        train_competitive()
    elif training_mode == "2":
        iterations = int(input("Enter number of self-play iterations: "))
        self_play_training(iterations)
    else:
        print("Invalid option, defaulting to competitive training")
        train_competitive()
 