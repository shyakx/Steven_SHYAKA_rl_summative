"""
DQN Agent for AgriTech Precision Farming

Deep Q-Network implementation with experience replay and target network
for stable training on the agricultural drone environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, Any, Tuple
import os

try:
    from .base_agent import BaseAgent, DQNNetwork, ReplayBuffer
except ImportError:
    from base_agent import BaseAgent, DQNNetwork, ReplayBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        super().__init__(state_size, action_size, config, "DQN")
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_frequency = config.get('target_update_frequency', 100)
        self.memory_size = config.get('memory_size', 10000)
        
        # Neural networks
        self.q_network = DQNNetwork(
            state_size, 
            action_size, 
            config.get('hidden_size', 128),
            config.get('num_layers', 2)
        ).to(self.device)
        
        self.target_network = DQNNetwork(
            state_size, 
            action_size, 
            config.get('hidden_size', 128),
            config.get('num_layers', 2)
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(self.memory_size)
        
        # Training tracking
        self.steps_since_target_update = 0
        self.total_training_steps = 0
        
        print(f"🤖 DQN Agent initialized with {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def act(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current environment state
            training: Whether agent is in training mode
            
        Returns:
            Selected action
        """
        # Preprocess state
        state_array = self.preprocess_state(state)
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Get Q-values from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        
        return action
    
    def remember(self, state: Dict[str, Any], action: int, reward: float, 
                next_state: Dict[str, Any], done: bool):
        """Store experience in replay buffer."""
        state_array = self.preprocess_state(state)
        next_state_array = self.preprocess_state(next_state)
        
        self.memory.push(state_array, action, reward, next_state_array, done)
    
    def learn(self, experience: Tuple = None) -> float:
        """
        Train the Q-network using experience replay.
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        self.steps_since_target_update += 1
        if self.steps_since_target_update >= self.target_update_frequency:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.steps_since_target_update = 0
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        self.total_training_steps += 1
        self.update_training_info()
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save the agent's model."""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the agent's model."""
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        print(f"📂 DQN model loaded from {filepath}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get current training information."""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_steps': self.total_training_steps,
            'target_updates': self.total_training_steps // self.target_update_frequency
        }


def train_dqn_agent(config: Dict[str, Any], save_dir: str = "models") -> DQNAgent:
    """
    Train a DQN agent on the AgriTech environment.
    
    Args:
        config: Training configuration
        save_dir: Directory to save models
        
    Returns:
        Trained DQN agent
    """
    try:
        from trainer import TrainingLogger, ModelCheckpoint, evaluate_agent
        from custom_env import PrecisionFarmingEnv
    except ImportError:
        from training.trainer import TrainingLogger, ModelCheckpoint, evaluate_agent
        from environment.custom_env import PrecisionFarmingEnv
    
    # Create environment
    env = PrecisionFarmingEnv()
    
    # Get state and action sizes
    obs, _ = env.reset()
    dummy_agent = DQNAgent(0, 0, {})
    state_size = len(dummy_agent.preprocess_state(obs))
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size, config)
    
    # Training infrastructure
    logger = TrainingLogger("DQN", config.get('log_dir', 'logs'))
    checkpoint = ModelCheckpoint(save_dir, "DQN")
    
    # Training loop
    max_episodes = config.get('max_episodes', 1000)
    eval_interval = config.get('eval_interval', 50)
    save_interval = config.get('save_interval', 100)
    
    print(f"🚀 Starting DQN training for {max_episodes} episodes")
    print(f"📊 State size: {state_size}, Action size: {action_size}")
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        total_reward = 0
        episode_loss = 0
        loss_count = 0
        
        for step in range(env.MAX_STEPS):
            # Select action
            action = agent.act(obs, training=True)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(obs, action, reward, next_obs, done)
            
            # Learn from experience
            loss = agent.learn()
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            total_reward += reward
            obs = next_obs
            info = next_info
            
            if done:
                break
        
        # Calculate metrics
        success_rate = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
        avg_loss = episode_loss / max(1, loss_count)
        
        # Log episode
        logger.log_episode(episode, total_reward, step + 1, success_rate, 
                          avg_loss, agent.epsilon)
        
        # Print progress
        if episode % 10 == 0:
            training_info = agent.get_training_info()
            print(f"Episode {episode:4d}: Reward={total_reward:7.2f}, "
                  f"Steps={step+1:3d}, Success={success_rate:.0%}, "
                  f"Loss={avg_loss:.4f}, ε={agent.epsilon:.3f}, "
                  f"Memory={training_info['memory_size']}")
        
        # Evaluation
        if episode % eval_interval == 0 and episode > 0:
            avg_reward, success_rate, avg_steps = evaluate_agent(env, agent, num_episodes=5)
            logger.log_evaluation(episode, avg_reward, success_rate, avg_steps)
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            model_path = checkpoint.save_model({
                'agent_state': agent.q_network.state_dict(),
                'target_state': agent.target_network.state_dict(),
                'optimizer_state': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon
            }, episode, {'avg_reward': total_reward})
            
            # Also save using agent's method
            agent.save_model(os.path.join(save_dir, f"dqn_episode_{episode}.pth"))
        
        agent.update_episode_info()
    
    # Final evaluation and save
    final_avg_reward, final_success_rate, final_avg_steps = evaluate_agent(env, agent, num_episodes=20)
    logger.log_evaluation(max_episodes, final_avg_reward, final_success_rate, final_avg_steps)
    
    # Save final model
    final_model_path = os.path.join(save_dir, "dqn_final.pth")
    agent.save_model(final_model_path)
    
    # Generate training plots
    plot_path = os.path.join(logger.log_dir, "dqn_training_plots.png")
    fig = logger.plot_training_progress(plot_path)
    
    # Save metrics
    metrics_path = os.path.join(logger.log_dir, "dqn_metrics.json")
    logger.save_metrics(metrics_path)
    
    print(f"🎉 DQN training completed!")
    print(f"📊 Final performance: Reward={final_avg_reward:.2f}, "
          f"Success Rate={final_success_rate:.1%}, Steps={final_avg_steps:.1f}")
    
    env.close()
    return agent


if __name__ == "__main__":
    # Test DQN agent
    print("🧪 Testing DQN Agent")
    print("=" * 30)
    
    # Test configuration
    test_config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'hidden_size': 128,
        'num_layers': 2,
        'memory_size': 5000,
        'target_update_frequency': 100
    }
    
    # Create test agent
    try:
        from base_agent import get_state_action_sizes
    except ImportError:
        from agents.base_agent import get_state_action_sizes
    
    state_size, action_size = get_state_action_sizes()
    
    agent = DQNAgent(state_size, action_size, test_config)
    print(f"✅ DQN agent created successfully")
    
    # Test action selection
    try:
        from custom_env import PrecisionFarmingEnv
    except ImportError:
        from environment.custom_env import PrecisionFarmingEnv
    env = PrecisionFarmingEnv()
    obs, _ = env.reset()
    
    action = agent.act(obs, training=True)
    print(f"✅ Action selection test: {action}")
    
    # Test experience storage and learning
    next_obs, reward, terminated, truncated, _ = env.step(action)
    agent.remember(obs, action, reward, next_obs, terminated or truncated)
    
    # Add more experiences to test learning
    for _ in range(100):
        obs, _ = env.reset()
        action = agent.act(obs, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.remember(obs, action, reward, next_obs, terminated or truncated)
    
    loss = agent.learn()
    print(f"✅ Learning test: Loss = {loss:.4f}")
    
    # Test model saving and loading
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    try:
        agent.save_model(temp_path)
        agent.load_model(temp_path)
        print(f"✅ Model save/load test passed")
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass  # Ignore cleanup error
    
    env.close()
    print("✅ DQN agent ready for training!")
    
    # Uncomment to run a short training session
    # print("\n🚀 Running short training session...")
    # train_config = test_config.copy()
    # train_config.update({'max_episodes': 50, 'eval_interval': 20, 'save_interval': 25})
    # trained_agent = train_dqn_agent(train_config)
