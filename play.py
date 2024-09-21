from utils.board import *  # Import board utilities
from utils.agent import Agent, agent1  # Import agent1 (AI player)
from settings import mapper  # Import the mapper for player display
import random

# Initialize the AI agent as player 1
agent1.load_model()  # Load the pretrained model for agent1

# Initialize the human player as player -1
human = Agent(player=-1)

def play():
    """
    Function to play a game between a human player and an AI agent (agent1).
    The game alternates between players until a winner is determined or the game ends in a draw.
    """
    # Initialize the game board
    board = reset_board()

    # Randomly decide who starts the game
    if random.uniform(0, 1) > 0.5:
        player = agent1  # AI starts the game
        # AI selects its first move
        action = agent1.choose_action(board, False)
    else:
        player = human  # Human starts the game
        # Display the current board state
        print('Current board:')
        print(display_board(board))
        print('-------------')
        # Prompt human for their move
        action = int(input("Provide your choice:  ")) - 1

    # Update the board with the first move
    board[action] = player.player

    # Display the board after the first move
    print('Current board:')
    print(display_board(board))
    print('-------------')

    # Main game loop
    while True:
        # Switch between the AI and human player
        player = agent1 if player is human else human

        # Check if there is a winner
        winner = check_winner(board)
        if winner:
            # Announce the winner and display the final board
            print(f'Game is end. Won player {mapper[winner]}')
            print(display_board(board))
            break

        # Check if no more moves are possible (draw)
        if not is_any_possible_move(board):
            print(f'Game is end. Draw!')
            break

        # If it's the AI's turn, select an action based on the model
        if player is agent1:
            action = agent1.choose_action(board, False)
        else:
            # Prompt human for their next move
            action = int(input("Provide your choice:  ")) - 1

        # Update the board with the selected move
        board[action] = player.player
        # Display the current board after the move
        print('Current board:')
        print(display_board(board))
        print('-------------')

# Start the game
play()
