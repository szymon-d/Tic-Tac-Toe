import torch
from torch import nn
import numpy as np
from torch.optim import Adam


class QNetwork(nn.Module):
    """
    A neural network representing a Q-learning agent, which learns
    optimal action-value functions (Q-values) for decision-making.

    Attributes:
    - epsilon (float): exploration rate for epsilon-greedy action selection.
    - gamma (float): discount factor for future rewards.
    - optimizer (torch.optim.Adam): optimizer for updating network weights.
    - MSE (torch.nn.MSELoss): mean squared error loss function.
    """
    # Defining the loss function (Mean Squared Error) for the Q-learning update.
    MSE = torch.nn.MSELoss()

    def __init__(self, epsilon: float, lr: float, gamma: float):
        """
        Initialize the QNetwork.

        Args:
        - epsilon (float): exploration rate for epsilon-greedy policy.
        - lr (float): learning rate for the optimizer.
        - gamma (float): discount factor for future rewards.
        """
        super().__init__()

        # Defining fully connected layers for the network.
        # Input layer: takes 9 input features and maps to 256 hidden neurons.
        self.fc1 = nn.Linear(9, 256)
        # Hidden layer: 256 neurons fully connected to another 256 neurons.
        self.fc2 = nn.Linear(256, 256)
        # Output layer: maps the hidden representation to 9 output Q-values (one for each action).
        self.fc3 = nn.Linear(256, 9)

        # Store epsilon and gamma parameters.
        self.epsilon = epsilon
        self.gamma = gamma

        # Adam optimizer for updating the network's weights with the specified learning rate.
        self.optimizer = Adam(params=self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Q-network to compute Q-values for a given state.

        Args:
        - x (torch.Tensor): input tensor representing the state.

        Returns:
        - torch.Tensor: output tensor of Q-values for each possible action.
        """
        # Apply ReLU activation after the first and second layers.
        x = torch.relu(self.fc1(x))  # Pass input through the first layer.
        x = torch.relu(self.fc2(x))  # Pass through the second layer.
        return self.fc3(x)  # Output from the third layer (no activation, raw Q-values).

    def update_q_values(self,
                        board: np.ndarray,
                        next_board: np.ndarray,
                        action: int,
                        reward: float) -> float:
        """
        Update the Q-values based on the given transition using the Q-learning algorithm.

        Args:
        - board (np.ndarray): current state of the environment (as a numpy array).
        - next_board (np.ndarray): next state after taking the action (as a numpy array).
        - action (int): the action taken in the current state.
        - reward (float): the reward received after taking the action.

        Returns:
        - float: the computed loss value after the update.
        """
        # Convert numpy arrays (state representations) to PyTorch tensors.
        board = torch.from_numpy(board).float()
        next_board = torch.from_numpy(next_board).float()

        # Perform a forward pass to get Q-values for the current and next states.
        q_values = self(board)  # Current state's Q-values.
        next_q_values = self(next_board)  # Next state's Q-values.

        # Compute the target Q-value using the Bellman equation.
        # target = reward + gamma * max(Q(s', a')) (where s' is the next state)
        target = reward + self.gamma * torch.max(next_q_values).detach()

        # Compute the loss between the predicted Q-value for the chosen action and the target.
        loss = self.MSE(q_values[action].unsqueeze(0), target.unsqueeze(0))

        # Zero the gradients before performing backpropagation.
        self.optimizer.zero_grad()

        # Backpropagation: compute the gradient of the loss with respect to the network parameters.
        loss.backward()

        # Update the network weights based on the computed gradients.
        self.optimizer.step()

        # Return the scalar loss value.
        return loss.item()
