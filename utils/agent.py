from .network import QNetwork
from settings import *  # Import settings for neural network configuration.
from .board import *  # Import necessary board functions.
import numpy as np
import random
import os


class Agent:
    """
    This class represents a reinforcement learning agent that interacts with the environment (the game board),
    decides its moves based on a neural network, and learns through reward-based feedback.
    """

    def __init__(self, player: int):
        """
        Initialize the Agent with a player number and create its QNetwork.

        Args:
        - player (int): Player number, either 1 (for player 1) or -1 (for player 2).
        """
        self.player = player  # Store the player identifier.
        self.network = QNetwork(**settings[self.player])  # Initialize the QNetwork based on settings.
        self.model_path = os.path.join(MODELS_DIR,
                                       f'agent{mapper[self.player]}.pth')  # Define path to save/load the model.

    def __str__(self):
        """
        String representation of the agent.

        Returns:
        - str: "Agent X" or "Agent O", depending on the player number.
        """
        return f"Agent {mapper[self.player]}"

    def get_reward(self, board: np.ndarray, board_before_action: np.ndarray, action: int) -> float:
        """
        Compute the reward for the agent based on the current state of the board and the action taken.

        Args:
        - board (np.ndarray): The game board after the agent's action.
        - board_before_action (np.ndarray): The game board before the agent's action.
        - action (int): The action (position) taken by the agent on the board.

        Returns:
        - float: The reward value based on the game state and action.
        """
        # Check if there is a winner after the agent's action.
        winner = check_winner(board=board)

        if winner:
            return winner * self.player  # Reward if the agent won (or punishment if the opponent won).

        # Reward for starting from the middle (favorable strategy).
        if np.sum(board == 0) == 8 and board.flatten().tolist()[4] == self.player:
            return 0.9

        # Punishment for not starting from the middle.
        if np.sum(board == 0) == 8 and board.flatten().tolist()[4] != self.player and np.sum(board == self.player * -1) == 0:
            return -0.9

        # Punishment if the agent missed a chance to win.
        if if_player_win_in_next_turn(board=board_before_action, player=self.player, amount_of_winnig_configs=1):
            return -1

        # Punishment if the agent failed to block an opponent's winning move.
        if if_player_lost_in_next_turn(board=board, player=self.player):
            return -1

        # Reward if the agent can win in the next turn (two possible configurations).
        if if_player_win_in_next_turn(board=board, player=self.player, amount_of_winnig_configs=2):
            return 0.6

        # Smaller reward if the agent can win in the next turn (one possible configuration).
        if if_player_win_in_next_turn(board=board, player=self.player, amount_of_winnig_configs=1):
            return 0.3

        # If the board is full, it’s a draw.
        if np.sum(board == 0) == 0:
            return 0

        # Small reward for making a move in a corner position (strategically important).
        if action in (0, 2, 6, 8):
            return 0.1

        # Default no reward for a neutral move.
        return 0

    def choose_action(self, board: torch.Tensor, random_state=True) -> int:
        """
        Choose the agent's next action, either randomly (exploration) or based on the neural network (exploitation).

        Args:
        - board (torch.Tensor): The current game board as a tensor.
        - random_state (bool): If True, allow random exploration based on the epsilon value of the QNetwork.

        Returns:
        - int: The chosen action (index of the board position).
        """
        # Get available moves (positions that are empty).
        available_choices = [idx for idx, value in enumerate(board.flatten().tolist()) if value == 0]

        # If random exploration is allowed and epsilon condition is met, pick a random action.
        if random_state:
            if np.random.uniform(0, 1) < self.network.epsilon:
                return random.choice([i for i in range(9) if i in available_choices])

        # Use the neural network to predict Q-values for the current board state.
        state = torch.tensor(board, dtype=torch.float32)
        q_values = self.network(state).detach().flatten().tolist()

        # Choose the action with the highest predicted Q-value from the available choices.
        return max(available_choices, key=lambda idx: q_values[idx])

    def save(self):
        """
        Save the agent's Q-network model to the specified file path.
        """
        torch.save(self.network.state_dict(), self.model_path)

    def load_model(self, training_mode: bool = False):
        """
        Load the agent's Q-network model from the specified file path.

        Args:
        - training_mode (bool): If True, the network is set to training mode; otherwise, it's set to evaluation mode.
        """
        # Reinitialize the network and load the saved model.
        self.network = QNetwork(**settings[self.player])
        self.network.load_state_dict(torch.load(self.model_path))

        # Set the network to evaluation mode unless training mode is explicitly requested.
        if not training_mode:
            self.network.eval()


# Initialize two agents for the game: agent1 for player 1 (X) and agent2 for player -1 (O).
agent1 = Agent(player=1)
agent2 = Agent(player=-1)
