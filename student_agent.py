# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict
import time
import json


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self, board = None, score = None):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        if board is not None:
            self.board = copy.deepcopy(board)
        self.score = 0
        if score is not None:
            self.score = score

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        score_before_move = self.score

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        board_before_spawn = self.board.copy()

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}, board_before_spawn, score_before_move

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)
    

def load_weights(approx, path):
    with open(path, "r") as f:
        raw = json.load(f)
    weights = [
        defaultdict(float, {tuple(map(int, k.split(","))): v for k, v in d.items()})
        for d in raw
    ]
    approx.weights = weights

def sysmetric_generator(pattern):
    #returna list 8(maybe less) symmetries of a pattern
    pass

def rotate90(coord):
    return tuple((y, 3 - x) for (x, y) in coord)

def rotate180(coord):
    return tuple((3 - x, 3 - y) for (x, y) in coord)

def rotate270(coord):
    return tuple((3 - y, x) for (x, y) in coord)

def flip_horizontal(coord):
    return tuple((x, 3 - y) for (x, y) in coord)

def flip_vertical(coord):
    return tuple((3 - x, y) for (x, y) in coord)

def flip_diag(coord):
    return tuple((y, x) for (x, y) in coord)

def flip_anti_diag(coord):
    return tuple((3 - y, 3 - x) for (x, y) in coord)

def normalize(coord):
    return tuple(sorted(coord))

def symmetric_generator(pattern):
    #pattern = normalize(pattern)
    transformations = [
        lambda x: x,
        rotate90,
        rotate180,
        rotate270,
        flip_horizontal,
        flip_vertical,
        flip_diag,
        flip_anti_diag,
    ]
    seen = set()
    result = []
    for transform in transformations:
        transformed = transform(pattern)
        normalized = normalize(transformed)
        if normalized not in seen:
            seen.add(normalized)
            result.append(transformed)
    return result

def load_weights(approx, path):
    with open(path, "r") as f:
        raw = json.load(f)
    weights = [
        defaultdict(float, {tuple(map(int, k.split(","))): v for k, v in d.items()})
        for d in raw
    ]
    approx.weights = weights

class NTupleApproximator:
    def __init__(self, board_size, patterns, trained_weights_path = None):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        if trained_weights_path is not None:
            print("loading")
            load_weights(self, trained_weights_path)
            #print(self.weights)
            print("loaded")
        #    with open(trained_weights_path, "rb") as f:
        #        loaded_weights = pickle.load(f)
        #    for i, pattern in enumerate(patterns):
        #        self.weights[i] = loaded_weights[i]
            
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = [symmetric_generator(p) for p in patterns]
        for i, p_list in enumerate(self.symmetry_patterns):
            print(p_list)

        self.tile_index_map = [0 for _ in range(32768 * 4)]
        for i in range(1, 16):
          self.tile_index_map[pow(2, i)] = i

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return self.tile_index_map[tile]

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        feature_vector = tuple(self.tile_to_index(board[x, y]) for x, y in coords)
        return feature_vector

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        v = 0.0
        for i, p_list in enumerate(self.symmetry_patterns):
            for p in p_list:
                v += self.weights[i][self.get_feature(board, p)]
        return v / len(self.symmetry_patterns)
    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, p_list in enumerate(self.symmetry_patterns):
            for p in p_list:
                self.weights[i][self.get_feature(board, p)] += alpha * delta / len(p_list)

def select_action(env, approximator, epsilon):
    #print("2", env.board, end = '\n\n')
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]

    if random.random() < epsilon:
        return random.choice(legal_moves)  # Exploration (random move)
    if not legal_moves:
        return -1
    # Exploitation: Simulate each move and pick the best one
    best_action = max(legal_moves, key=lambda a: simulate_and_evaluate(env, approximator, a))
    return best_action

def simulate_and_evaluate(env, approximator, action):
    """Simulates taking an action in a copied environment and evaluates the resulting board state."""
    env_copy = copy.deepcopy(env)
    next_state, new_score, _, _, state_before_spawn, score_before_move = env_copy.step(action)
    reward = new_score - score_before_move
    #print(reward + approximator.value(state_before_spawn))
    return reward + approximator.value(state_before_spawn)


approximator = None

def get_action(state, score):
    patterns = [
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)),
        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)),
        ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1))
    ]
    global approximator 
    if approximator is None:
        approximator = NTupleApproximator(board_size=4, patterns=patterns, trained_weights_path = "TD-weights_30000.json")
    env = Game2048Env()
    #print("3",state,end="\n\n")
    env.board = copy.deepcopy(state)
    env.score = score
    action = select_action(env, approximator, 0)


    
    return action


