import os
import sys
import time
from agent import train as train_single_player
from train_two_player import train_competitive, self_play_training

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_ascii_snake():
    snake = """
    _    _            _____             _        
   | |  | |          /  ___|           | |       
   | |  | | ___  _ __\ `--. _ __   __ _| | _____ 
   | |/\| |/ _ \| '_ \`--. \ '_ \ / _` | |/ / _ \\
   \  /\  / (_) | | | /\__/ / | | | (_| |   <  __/
    \/  \/ \___/|_| |_\____/|_| |_|\__,_|_|\_\___|
    """
    print(snake)

def print_menu():
    print_ascii_snake()
    print("\nWelcome to AI Snake Training Platform!\n")
    print("1. Train Single Player Snake AI")
    print("2. Train Two Competitive Snakes")
    print("3. Self-Play Training")
    print("4. View Training Documentation")
    print("5. Exit")

def single_player_menu():
    clear_screen()
    print("\n=== Single Player Training ===\n")
    print("This will train a snake to play by itself and avoid collisions.")
    print("The AI will learn through reinforcement learning.\n")
    
    model_name = input("Enter a model name (or press Enter for default 'model.pth'): ").strip()
    if not model_name:
        model_name = "model.pth"
    
    print(f"\nStarting training with model: {model_name}")
    print("Press Ctrl+C to stop training at any time.")
    time.sleep(2)
    
    try:
        train_single_player(model_name)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        time.sleep(1)

def two_player_menu():
    clear_screen()
    print("\n=== Two Player Competitive Training ===\n")
    print("This will train two snake AIs competing against each other.")
    print("They will learn to collect food and block each other.\n")
    
    print("Starting competitive training...")
    print("Press Ctrl+C to stop training at any time.")
    time.sleep(2)
    
    try:
        train_competitive()
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        time.sleep(1)

def self_play_menu():
    clear_screen()
    print("\n=== Self-Play Training ===\n")
    print("This will train a snake against copies of itself to improve strategy.")
    print("Self-play is an advanced technique for reinforcement learning.\n")
    
    iterations = input("Enter number of iterations (default: 100): ").strip()
    if not iterations:
        iterations = 100
    else:
        try:
            iterations = int(iterations)
        except ValueError:
            print("Invalid input. Using default 100 iterations.")
            iterations = 100
    
    print(f"\nStarting self-play training for {iterations} iterations.")
    print("Press Ctrl+C to stop training at any time.")
    time.sleep(2)
    
    try:
        self_play_training(iterations)
    except KeyboardInterrupt:
        print("\nTraining stopped by user.")
        time.sleep(1)

def view_documentation():
    clear_screen()
    print("\n=== Snake AI Training Documentation ===\n")
    
    docs = """
    SNAKE AI TRAINING SYSTEM DOCUMENTATION
    --------------------------------------
    
    This system uses reinforcement learning with Deep Q-Learning to train snake AIs.
    
    1. SINGLE PLAYER MODE:
       - Snake learns to navigate, find food, and avoid collisions
       - Model is saved automatically when performance improves
       - State includes danger detection, food location, and snake direction
       - Best for learning basic navigation and survival skills
    
    2. COMPETITIVE MODE:
       - Two snakes compete for food and territory
       - Each snake has its own model that learns independently
       - State includes opponent position and movement
       - Great for developing more advanced strategies
    
    3. SELF-PLAY MODE:
       - Snake plays against copies of itself
       - Similar to how AlphaGo was trained
       - Each iteration creates a copy with the latest knowledge
       - Excellent for mastering competitive strategies
    
    TRAINING NOTES:
    - Models are saved in the './model' directory
    - Training plots show progress over time
    - Epsilon controls exploration vs. exploitation
    - You can adjust learning parameters in the agent files
    
    ADVANCED USAGE:
    - Modify reward functions in game files for different behaviors
    - Add more features to state representations for better learning
    - Experiment with network architectures in model.py
    - Try different hyperparameters like learning rate and batch size
    """
    
    print(docs)
    input("\nPress Enter to return to main menu...")

def main():
    while True:
        clear_screen()
        print_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            single_player_menu()
        elif choice == '2':
            two_player_menu()
        elif choice == '3':
            self_play_menu()
        elif choice == '4':
            view_documentation()
        elif choice == '5':
            clear_screen()
            print("Thank you for using the Snake AI Training Platform!")
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\nProgram terminated by user.")
        sys.exit(0) 