# Tic-Tac-Toe with AI

Welcome to the **Tic-Tac-Toe** game built with a custom AI using reinforcement learning! This project allows you to play the classic game of Tic-Tac-Toe against an AI agent. The AI has been trained using Q-learning, making it capable of learning from its mistakes and improving over time. 

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [How to Play](#how-to-play)
- [Game Rules](#game-rules)
- [AI Agent](#ai-agent)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Play Tic-Tac-Toe**: Classic 3x3 Tic-Tac-Toe board.
- **AI Opponent**: Play against an AI agent trained using Q-learning.
- **Human vs AI**: Face off against a competitive AI.
- **Reinforcement Learning**: The AI gets better over time by adjusting its strategy based on rewards and punishments.
- **Save/Load Models**: The AI's learned strategy can be saved and reloaded for further training or testing.

## Installation

1. **Clone the repository**:
    ```bash
    git https://github.com/szymon-d/Tic-Tac-Toe.git
    cd Tic-Tac-Toe
    ```

2. **Create a Python virtual environment (optional)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Optional: Pre-trained model**:
   - If you want to use a pre-trained model for the AI, you can place the model file (`.pth`) in the `models` directory.
   - If you don't have one, the AI will start training from scratch and save the model after training.

## How to Play

1. **Run the game**:
    To start the game, simply run:
    ```bash
    python play.py
    ```

2. **Gameplay**:
    - At the beginning of the game, a random draw will decide whether the AI or the human player starts.
    - When it's your turn, you will be asked to provide your move by entering a number corresponding to a board position (1-9).

3. **Board Layout**:
    The Tic-Tac-Toe board is represented as follows:
    ```
    1 | 2 | 3
    ---------
    4 | 5 | 6
    ---------
    7 | 8 | 9
    ```
    When prompted, choose a number to place your marker (X or O) in the corresponding cell.

4. **End of Game**:
    - The game ends when there is a winner or the board is full (draw).
    - The board will display the final state, and you will be informed whether you won, lost, or if the game was a draw.

## Game Rules

- **Objective**: The objective is to get three of your marks in a row (horizontally, vertically, or diagonally).
- **Game Alternates**: Players take turns to place their mark on an empty spot.
- **Winning**: The first player to get three in a row wins.
- **Draw**: If the board is full and no player has won, the game ends in a draw.

## AI Agent

This project implements a Q-learning agent to play Tic-Tac-Toe. The AI learns through experience, adjusting its Q-values based on rewards. Here’s a brief overview of how it works:

- **Q-learning**: The AI uses a neural network to estimate Q-values (action-value pairs) and updates its strategy based on the outcome of each game.
- **Rewards and Punishments**: The AI receives positive rewards for winning and penalties for making poor moves, especially if they lead to losing or missed opportunities.
- **Model Training**: After each game, the AI updates its Q-network using the feedback from the outcome. The more it plays, the better it becomes.

### AI Strategy
The AI uses a neural network to predict the best moves based on the current state of the board. Over time, it learns the most optimal strategies to win.

## File Structure

