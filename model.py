import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)  # Apply dropout after first activation
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)  # Add weight decay
        self.criterion = nn.MSELoss()
        # Track loss history for debugging and analysis
        self.loss_history = []

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Add noise to next state prediction for exploration
                noise = torch.randn(self.model(next_state[idx]).shape) * 0.01
                next_q_values = self.model(next_state[idx]) + noise
                Q_new = reward[idx] + self.gamma * torch.max(next_q_values)

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # Implement gradient clipping to prevent exploding gradients
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        self.loss_history.append(loss.item())
        loss.backward()
        
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()

# A more advanced model for handling complex situations
class ConvQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Convert 1D state to 2D grid representation for convolution
        self.input_size = input_size
        # First convolution layer
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Calculate flattened size after convolutions
        self.flattened_size = 64 * input_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Reshape input for 1D convolution (batch_size, channels, input_size)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.input_size)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 1)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
    def save(self, file_name='conv_model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


# Utility function to visualize model performance
def plot_loss(trainer, filename="training_loss.png"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(trainer.loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    model_folder_path = './model'
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
        
    plt.savefig(os.path.join(model_folder_path, filename))
    plt.close()



