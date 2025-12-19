import gymnasium as gym
from gymnasium import spaces
import numpy as np
from physics import TrussPhysics

class TrussEnv(gym.Env):
    def __init__(self, config_path=None):
        super(TrussEnv, self).__init__()
        self.physics = TrussPhysics(config_path=config_path)
        self.n_bars = len(self.physics.all_possible_bars)
        # 0 to n_bars-1: Toggle that specific bar
        # n_bars: No-Op (Do nothing)
        self.action_space = spaces.Discrete(self.n_bars + 1)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.n_bars,), dtype=np.int8
        )
        self.current_structure = np.ones(self.n_bars, dtype=np.int8)
        w_pen = self.physics.config.get('weight_penalty', 0.5)
        starting_score = 10.0 - (np.sum(self.n_bars) * w_pen)
        if starting_score < -50.0:
            self.failure_threshold = starting_score * 1.5
        else:
            self.failure_threshold = -50.0
        print(
            f"[Env] Start Score: {starting_score:.2f} | Failure Threshold: {self.failure_threshold:.2f}"
            )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_structure = np.ones(self.n_bars, dtype=np.int8)
        return self.current_structure, {}

    def step(self, action):
        # 1. action
        if action < self.n_bars:
            self.current_structure[action] = 1 - self.current_structure[action]
        # sync: commit physics cleaning to agent state for more efficient learning
        self.current_structure = self.physics._get_cleaned_structure(
            self.current_structure
        )
        # 2. solve
        result = self.physics.solve(self.current_structure)
        terminated = False
        truncated = False
        # 3. reward logic adn termination
        if not result.valid:
            reward = self.failure_threshold - (result.max_displacement * 0.1)
            terminated = True # False allows continued exploration, True for simple config
        else:
            w_pen = self.physics.config.get('weight_penalty')
            s_pen = self.physics.config.get('stiffness_penalty')
            disp_cost = min(result.max_displacement * s_pen, 20.0)
            reward = 10.0 - (result.weight * w_pen) - disp_cost
        info = {
            "displacement": result.max_displacement,
            "weight": result.weight,
            "valid": result.valid
        }
        return self.current_structure.copy(), reward, terminated, truncated, info