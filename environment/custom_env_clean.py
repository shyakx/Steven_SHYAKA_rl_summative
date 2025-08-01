"""
AgriTech Precision Farming Drone Environment

Mission: Using AI to build impactful solutions in agricultural sectors
through precision farming and resource optimization.

A custom RL environment where an autonomous drone navigates a farm grid
to treat diseased crops while managing limited resources (battery, treatment capacity).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import random


class PrecisionFarmingEnv(gym.Env):
    """
    🌾 AgriTech Precision Farming Drone Environment
    
    SCENARIO:
    An autonomous farming drone must navigate a 15x15 farm grid to locate and treat
    diseased crops while managing battery and treatment resources efficiently.
    
    MISSION ALIGNMENT:
    This simulates real-world agricultural AI challenges - resource management,
    spatial navigation, and optimization under constraints.
    
    AGENT OBJECTIVE:
    - Navigate the farm grid efficiently
    - Locate diseased crops (red cells)
    - Treat them with limited treatment capacity
    - Manage battery by visiting charging stations
    - Avoid obstacles (rocks, trees)
    - Maximize crop health while minimizing resource waste
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Environment Configuration
    GRID_SIZE = 15
    MAX_STEPS = 200
    INITIAL_BATTERY = 100
    INITIAL_TREATMENT = 20
    
    # Cell Types (State Representation)
    EMPTY = 0           # Brown soil - empty farmland
    HEALTHY_CROP = 1    # Green - healthy crops (no action needed)
    DISEASED_CROP = 2   # Red - diseased crops (need treatment)
    OBSTACLE = 3        # Gray - rocks/trees (impassable)
    CHARGING_STATION = 4 # Blue - battery charging stations
    TREATED_CROP = 5    # Light green - successfully treated crops
    
    # Action Space (6 discrete actions)
    ACTION_MOVE_UP = 0
    ACTION_MOVE_DOWN = 1
    ACTION_MOVE_LEFT = 2
    ACTION_MOVE_RIGHT = 3
    ACTION_TREAT_CROP = 4
    ACTION_CHARGE_BATTERY = 5
    
    ACTIONS = {
        0: "MOVE_UP",
        1: "MOVE_DOWN", 
        2: "MOVE_LEFT",
        3: "MOVE_RIGHT",
        4: "TREAT_CROP",
        5: "CHARGE_BATTERY"
    }
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Observation space: Dictionary with grid, agent position, and resources
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=5, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32
            ),
            "agent_pos": spaces.Box(
                low=0, high=self.GRID_SIZE-1, shape=(2,), dtype=np.int32
            ),
            "resources": spaces.Box(
                low=0, high=200, shape=(3,), dtype=np.int32  # battery, treatment, steps
            )
        })
        
        # Initialize state variables
        self.grid = None
        self.agent_pos = None
        self.battery_level = self.INITIAL_BATTERY
        self.treatment_capacity = self.INITIAL_TREATMENT
        self.steps_taken = 0
        self.initial_diseased_count = 0
        self.current_diseased_count = 0
        self.renderer = None
        
    def _generate_farm_layout(self) -> np.ndarray:
        """
        Generate a realistic farm layout with strategic placement:
        - 60% healthy crops (main farmland)
        - 15% of crops become diseased (challenge)
        - 10% obstacles (realistic farm hazards)
        - 2-3 charging stations (strategic placement)
        - Rest empty soil for navigation
        """
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int32)
        
        # Start with healthy crops covering most farmland
        crop_probability = 0.6
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if random.random() < crop_probability:
                    grid[i, j] = self.HEALTHY_CROP
        
        # Add diseased crops (15% of healthy crops become diseased)
        healthy_positions = np.where(grid == self.HEALTHY_CROP)
        healthy_indices = list(zip(healthy_positions[0], healthy_positions[1]))
        disease_count = max(1, int(len(healthy_indices) * 0.15))
        
        diseased_positions = random.sample(healthy_indices, min(disease_count, len(healthy_indices)))
        for pos in diseased_positions:
            grid[pos[0], pos[1]] = self.DISEASED_CROP
        
        # Add obstacles (rocks, trees) - 10% of grid
        obstacle_count = int(self.GRID_SIZE * self.GRID_SIZE * 0.1)
        empty_positions = np.where(grid == self.EMPTY)
        empty_indices = list(zip(empty_positions[0], empty_positions[1]))
        
        if len(empty_indices) >= obstacle_count:
            obstacle_positions = random.sample(empty_indices, obstacle_count)
            for pos in obstacle_positions:
                grid[pos[0], pos[1]] = self.OBSTACLE
        
        # Add 2-3 charging stations in strategic locations
        remaining_empty = np.where(grid == self.EMPTY)
        empty_indices = list(zip(remaining_empty[0], remaining_empty[1]))
        
        if len(empty_indices) >= 3:
            charging_positions = random.sample(empty_indices, 3)
            for pos in charging_positions:
                grid[pos[0], pos[1]] = self.CHARGING_STATION
        
        return grid
    
    def _get_valid_starting_position(self) -> Tuple[int, int]:
        """Find a valid starting position (empty cell or charging station)."""
        valid_positions = []
        for i in range(self.GRID_SIZE):
            for j in range(self.GRID_SIZE):
                if self.grid[i, j] in [self.EMPTY, self.CHARGING_STATION]:
                    valid_positions.append((i, j))
        
        if valid_positions:
            return random.choice(valid_positions)
        else:
            # Fallback: place on any non-obstacle cell
            for i in range(self.GRID_SIZE):
                for j in range(self.GRID_SIZE):
                    if self.grid[i, j] != self.OBSTACLE:
                        return (i, j)
        
        return (0, 0)  # Last resort
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Generate new farm layout
        self.grid = self._generate_farm_layout()
        
        # Place agent at valid starting position
        self.agent_pos = self._get_valid_starting_position()
        
        # Reset resources
        self.battery_level = self.INITIAL_BATTERY
        self.treatment_capacity = self.INITIAL_TREATMENT
        self.steps_taken = 0
        
        # Count initial diseased crops
        self.initial_diseased_count = np.sum(self.grid == self.DISEASED_CROP)
        self.current_diseased_count = self.initial_diseased_count
        
        # Create observation
        obs = self._get_observation()
        
        # Info for monitoring
        info = {
            "agent_pos": self.agent_pos,
            "battery_level": self.battery_level,
            "treatment_capacity": self.treatment_capacity,
            "initial_diseased_count": self.initial_diseased_count,
            "current_diseased_count": self.current_diseased_count,
            "completion_rate": 0.0
        }
        
        return obs, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        return {
            "grid": self.grid.copy(),
            "agent_pos": np.array(self.agent_pos, dtype=np.int32),
            "resources": np.array([
                self.battery_level, 
                self.treatment_capacity,
                self.steps_taken
            ], dtype=np.int32)
        }
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid (within bounds and not obstacle)."""
        x, y = pos
        if x < 0 or x >= self.GRID_SIZE or y < 0 or y >= self.GRID_SIZE:
            return False
        if self.grid[x, y] == self.OBSTACLE:
            return False
        return True
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        self.steps_taken += 1
        reward = 0.0
        
        # Movement actions
        if action in [self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN, 
                     self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT]:
            
            # Calculate new position
            x, y = self.agent_pos
            if action == self.ACTION_MOVE_UP:
                new_pos = (x - 1, y)
            elif action == self.ACTION_MOVE_DOWN:
                new_pos = (x + 1, y)
            elif action == self.ACTION_MOVE_LEFT:
                new_pos = (x, y - 1)
            elif action == self.ACTION_MOVE_RIGHT:
                new_pos = (x, y + 1)
            
            # Validate movement
            if self._is_valid_position(new_pos):
                self.agent_pos = new_pos
                self.battery_level -= 1  # Battery cost for movement
                reward -= 0.01  # Small penalty for movement to encourage efficiency
            else:
                reward -= 0.1  # Penalty for invalid movement
        
        # Treatment action
        elif action == self.ACTION_TREAT_CROP:
            x, y = self.agent_pos
            if self.grid[x, y] == self.DISEASED_CROP and self.treatment_capacity > 0:
                # Successful treatment
                self.grid[x, y] = self.TREATED_CROP
                self.treatment_capacity -= 1
                self.current_diseased_count -= 1
                reward += 10.0  # Large reward for treating diseased crop
            elif self.grid[x, y] == self.DISEASED_CROP and self.treatment_capacity == 0:
                reward -= 1.0  # Penalty for trying to treat without capacity
            else:
                reward -= 0.5  # Penalty for treating non-diseased cell
        
        # Charging action
        elif action == self.ACTION_CHARGE_BATTERY:
            x, y = self.agent_pos
            if self.grid[x, y] == self.CHARGING_STATION:
                self.battery_level = min(100, self.battery_level + 20)  # Charge 20%
                reward += 1.0  # Small reward for smart charging
            else:
                reward -= 0.5  # Penalty for charging at wrong location
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Success: All diseased crops treated
        if self.current_diseased_count == 0:
            terminated = True
            reward += 50.0  # Large completion bonus
        
        # Failure: Battery depleted
        elif self.battery_level <= 0:
            terminated = True
            reward -= 20.0  # Large penalty for battery depletion
        
        # Truncation: Max steps reached
        elif self.steps_taken >= self.MAX_STEPS:
            truncated = True
            reward -= 5.0  # Penalty for not completing in time
        
        # Additional rewards for efficiency
        if self.initial_diseased_count > 0:
            completion_rate = (self.initial_diseased_count - self.current_diseased_count) / self.initial_diseased_count
            reward += completion_rate * 0.1  # Incremental completion reward
        
        # Create observation and info
        obs = self._get_observation()
        info = {
            "agent_pos": self.agent_pos,
            "battery_level": self.battery_level,
            "treatment_capacity": self.treatment_capacity,
            "current_diseased_count": self.current_diseased_count,
            "completion_rate": (self.initial_diseased_count - self.current_diseased_count) / max(1, self.initial_diseased_count) * 100,
            "steps_taken": self.steps_taken,
            "action_taken": self.ACTIONS[action]
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render environment in human-readable format (console)."""
        print(f"\n🌾 Farm Status - Step {self.steps_taken}")
        print(f"🔋 Battery: {self.battery_level}% | 💉 Treatment: {self.treatment_capacity} | 🎯 Diseased: {self.current_diseased_count}")
        print("─" * (self.GRID_SIZE * 2 + 1))
        
        symbols = {
            self.EMPTY: "·",           # Empty soil
            self.HEALTHY_CROP: "🌱",   # Healthy crop
            self.DISEASED_CROP: "🔴",  # Diseased crop
            self.OBSTACLE: "🪨",       # Obstacle
            self.CHARGING_STATION: "⚡", # Charging station
            self.TREATED_CROP: "✅"    # Treated crop
        }
        
        for i in range(self.GRID_SIZE):
            row = ""
            for j in range(self.GRID_SIZE):
                if (i, j) == self.agent_pos:
                    row += "🚁"  # Drone agent
                else:
                    row += symbols.get(self.grid[i, j], "?")
                row += " "
            print(row)
        
        print("─" * (self.GRID_SIZE * 2 + 1))
    
    def _render_rgb_array(self):
        """Render environment as RGB array (for video recording)."""
        # This would require pygame or similar graphics library
        # For now, return a placeholder
        return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        if self.renderer:
            self.renderer.close()


def make_precision_farming_env(**kwargs):
    """Factory function to create the environment."""
    return PrecisionFarmingEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    print("🌾 Testing AgriTech Precision Farming Drone Environment")
    print("=" * 55)
    
    env = PrecisionFarmingEnv(render_mode="human")
    
    print(f"Environment Details:")
    print(f"  Grid Size: {env.GRID_SIZE}x{env.GRID_SIZE}")
    print(f"  Action Space: {env.action_space}")
    print(f"  Max Steps: {env.MAX_STEPS}")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial State:")
    print(f"  Diseased crops to treat: {info['initial_diseased_count']}")
    print(f"  Agent starting position: {info['agent_pos']}")
    print(f"  Battery: {info['battery_level']}%")
    print(f"  Treatment capacity: {info['treatment_capacity']}")
    
    # Test a few actions
    actions_to_test = [
        (env.ACTION_MOVE_RIGHT, "Move Right"),
        (env.ACTION_MOVE_DOWN, "Move Down"),
        (env.ACTION_TREAT_CROP, "Treat Crop"),
        (env.ACTION_MOVE_UP, "Move Up"),
        (env.ACTION_CHARGE_BATTERY, "Charge Battery")
    ]
    
    print(f"\nTesting Actions:")
    for i, (action, name) in enumerate(actions_to_test):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: {name} -> Reward: {reward:.2f}, Battery: {info['battery_level']}%")
        
        if terminated or truncated:
            print(f"  Episode ended: Completion rate {info['completion_rate']:.1f}%")
            break
    
    # Show final state
    env.render()
    
    env.close()
    print("\n✅ Environment test completed successfully!")
    print("\nKey Features:")
    print("  🎯 Mission-aligned agricultural scenario")
    print("  🔋 Resource management (battery + treatment)")
    print("  🚁 Drone navigation with obstacles")
    print("  🌾 Diseased crop treatment objective")
    print("  📊 Rich observation space and rewards")
