import copy
import random
import math
import numpy as np
import student_agent 
import time
import pickle
from collections import defaultdict

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
            with open(trained_weights_path, "rb") as f:
                loaded_weights = pickle.load(f)
            for i, pattern in enumerate(patterns):
                self.weights[i] = loaded_weights[i]
            
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
    return reward + approximator.value(state_before_spawn)

def td_learning(env, approximator, num_episodes=50000, alpha=0.02, gamma=0.99, epsilon_end=0.1, decay_rate = 0.99999, save_dir = "TD6_saves/"):

    final_scores = []
    success_flags = []
    num_steps = []
    episode_times = []
    epsilon = 0

    for episode in range(num_episodes):
        _ = env.reset()
        trajectory = []  # Store trajectory data if needed
        done = False
        max_tile = np.max(_)
        cnt = 0
        start_time = time.time()

        while not done and cnt < 1000:
            #cnt+=1
            legal_moves = [a for a in range(4) if env.is_move_legal(a)] # game ends when no legal moves or board is full
            if not legal_moves:
                break
            action = select_action(env, approximator, epsilon)
            next_state, new_score, done, _, state_before_spawn, score_before_move = env.step(action)
            #incremental_reward = new_score - score_before_move
            max_tile = max(max_tile, np.max(next_state))
            trajectory.append((next_state, state_before_spawn, copy.deepcopy(env)))

        reverse_trajectory = trajectory[:][::-1]
        for idx, transition in enumerate(reverse_trajectory):
            next_state, state_before_spawn, env_copy = transition
            a_best = select_action(env_copy, approximator, 0.0)
            if a_best == -1:
              r_next = 0
              v_after_next = 0
            else:
              next_next_state, new_score, done, _, next_state_before_spawn, score_before_move = env_copy.step(a_best)
              r_next = new_score - score_before_move
              v_after_next = approximator.value(next_state_before_spawn)
            delta = r_next + v_after_next - approximator.value(state_before_spawn)
            approximator.update(state_before_spawn, delta, alpha)

        episode_times.append(time.time() - start_time)
        epsilon = max(epsilon * decay_rate, epsilon_end)
        final_scores.append(env.score)
        num_steps.append(len(trajectory))
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            avg_steps = np.mean(num_steps[-100:])
            avg_train_time = np.mean(episode_times[-100:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f} | epsilon: {epsilon} | Steps: {avg_steps:.2f} | Time: {avg_train_time:.2f}")
        if (episode + 1) %  100 == 0:
            with open(save_dir + f"TD-weights_{((episode+1) // 10000 + 1 ) * 10000}.pkl", "wb") as f:
                pickle.dump(approximator.weights, f)
    return final_scores

# TODO: Define your own n-tuple patterns

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

approximator = NTupleApproximator(board_size=4, patterns=patterns, trained_weights_path = "TD-weights_30000.pkl")

if __name__ == "__main__":
    env  = student_agent.Game2048Env()
    final_scores = td_learning(env, approximator, num_episodes = 0, alpha=0.32, gamma=0.99, epsilon_end=0.00, decay_rate=0.999)
    scores = []
    for _ in range(10):
        env.reset()
        done = False
        while not done:
            action = select_action(env, approximator, 0)
            next_state, new_score, done, _, state_before_spawn, score_before_move = env.step(action)

        print(env.score)
        scores.append(env.score)
    print("average score: ", np.mean(scores))
            
            

    
#nohup python3 TD-training-4t.py > TD4.log 2>&1 &