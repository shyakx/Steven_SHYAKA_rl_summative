"""
Base Agent Interface for AgriTech RL Agents

Abstract base class defining the interface that all RL agents must implement.
Provides common functionality and ensures consistent API across algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
import os
import sys

# Add environment to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from custom_env import PrecisionFarmingEnv


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""
    
    def __init__(self, 
                 state_size: int,
                 action_size: int, 
                 config: Dict[str, Any],
                 agent_name: str = "BaseAgent"):
        """
        Initialize the agent.
        
        Args:
            state_size: Size of the state space
            action_size: Size of the action space
            config: Configuration dictionary
            agent_name: Name of the agent
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.agent_name = agent_name
        
        # Training tracking
        self.training_step = 0
        self.episode_count = 0
        
        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ {agent_name} using device: {self.device}")
    
    @abstractmethod
    def act(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current environment state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def learn(self, experience: Tuple) -> float:
        """
        Learn from experience.
        
        Args:
            experience: Experience tuple (state, action, reward, next_state, done)
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """Save the agent's model."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """Load the agent's model."""
        pass
    
    def preprocess_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess state dictionary into flat numpy array.
        
        Args:
            state: State dictionary from environment
            
        Returns:
            Flattened state array
        """
        # Extract components
        grid = state['grid'].flatten()  # 15x15 = 225
        agent_pos = state['agent_pos']  # 2
        resources = state['resources']  # 3
        
        # Normalize grid values to [0, 1]
        grid_normalized = grid / 5.0  # Max cell type value is 5
        
        # Normalize position to [0, 1]
        pos_normalized = agent_pos / 14.0  # Grid is 15x15, so max index is 14
        
        # Normalize resources
        battery = resources[0] / 100.0  # Battery percentage
        treatment = resources[1] / 20.0  # Treatment capacity
        steps_remaining = resources[2] / 200.0  # Steps remaining
        
        # Combine all features
        combined_state = np.concatenate([
            grid_normalized,  # 225 features
            pos_normalized,   # 2 features  
            [battery, treatment, steps_remaining]  # 3 features
        ])
        
        return combined_state.astype(np.float32)
    
    def get_state_size(self) -> int:
        """Get the size of preprocessed state."""
        return 225 + 2 + 3  # grid + position + resources = 230
    
    def update_training_info(self):
        """Update training step counter."""
        self.training_step += 1
    
    def update_episode_info(self):
        """Update episode counter."""
        self.episode_count += 1


class DQNNetwork(nn.Module):
    """Neural network for DQN agent."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Neural network for policy-based agents (REINFORCE, PPO)."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(PolicyNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build shared layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head (outputs action probabilities)
        self.policy_head = nn.Linear(hidden_size, action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        shared = self.shared_layers(state)
        policy_logits = self.policy_head(shared)
        return F.softmax(policy_logits, dim=-1)


class ActorCriticNetwork(nn.Module):
    """Neural network for Actor-Critic agents."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, num_layers: int = 2):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build shared layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(state_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor head (outputs action probabilities)
        self.actor_head = nn.Linear(hidden_size, action_size)
        
        # Critic head (outputs state value)
        self.critic_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        """Forward pass through the network."""
        shared = self.shared_layers(state)
        
        # Actor output (action probabilities)
        policy_logits = self.actor_head(shared)
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # Critic output (state value)
        state_value = self.critic_head(shared)
        
        return action_probs, state_value


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


def create_environment():
    """Create and return the AgriTech environment."""
    return PrecisionFarmingEnv()


def get_state_action_sizes():
    """Get state and action space sizes for the environment."""
    env = create_environment()
    
    # Get a sample state to determine size
    obs, _ = env.reset()
    dummy_agent = type('DummyAgent', (BaseAgent,), {
        'act': lambda self, state, training=True: 0,
        'learn': lambda self, experience: 0.0,
        'save_model': lambda self, filepath: None,
        'load_model': lambda self, filepath: None
    })(0, 0, {})
    
    state_size = len(dummy_agent.preprocess_state(obs))
    action_size = env.action_space.n
    
    env.close()
    
    return state_size, action_size


if __name__ == "__main__":
    # Test the base agent infrastructure
    print("🧪 Testing Base Agent Infrastructure")
    print("=" * 40)
    
    # Test state and action size calculation
    state_size, action_size = get_state_action_sizes()
    print(f"✅ State size: {state_size}")
    print(f"✅ Action size: {action_size}")
    
    # Test neural networks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Device: {device}")
    
    # Test DQN network
    dqn_net = DQNNetwork(state_size, action_size)
    test_input = torch.randn(1, state_size)
    output = dqn_net(test_input)
    print(f"✅ DQN Network output shape: {output.shape}")
    
    # Test Policy network
    policy_net = PolicyNetwork(state_size, action_size)
    policy_output = policy_net(test_input)
    print(f"✅ Policy Network output shape: {policy_output.shape}")
    
    # Test Actor-Critic network
    ac_net = ActorCriticNetwork(state_size, action_size)
    actor_output, critic_output = ac_net(test_input)
    print(f"✅ Actor-Critic Network outputs: {actor_output.shape}, {critic_output.shape}")
    
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some dummy experiences
    for i in range(10):
        state = np.random.randn(state_size)
        action = np.random.randint(action_size)
        reward = np.random.randn()
        next_state = np.random.randn(state_size)
        done = np.random.choice([True, False])
        buffer.push(state, action, reward, next_state, done)
    
    # Sample batch
    if len(buffer) >= 5:
        batch = buffer.sample(5)
        print(f"✅ Replay buffer batch shapes: {[arr.shape for arr in batch]}")
    
    print("✅ Base agent infrastructure ready!")
    print("\n🚀 Ready to implement specific RL algorithms!")
