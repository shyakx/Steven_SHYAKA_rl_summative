"""
PPO Agent for AgriTech Precision Farming

Proximal Policy Optimization with clipped objective and value function learning.
State-of-the-art policy gradient method with improved stability.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Any, Tuple, List
import os
from collections import deque

try:
    from .base_agent import BaseAgent, PolicyNetwork
except ImportError:
    from base_agent import BaseAgent, PolicyNetwork


class PPOBuffer:
    """Experience buffer for PPO training."""
    
    def __init__(self, size: int, state_size: int):
        self.size = size
        self.states = np.zeros((size, state_size), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        
        self.ptr = 0
        self.trajectory_start = 0
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a transition."""
        idx = self.ptr % self.size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        self.ptr += 1
    
    def finish_trajectory(self, last_value=0, gamma=0.99, lam=0.95):
        """Compute advantages and returns for the current trajectory."""
        trajectory_slice = slice(self.trajectory_start, self.ptr)
        
        rewards = np.append(self.rewards[trajectory_slice], last_value)
        values = np.append(self.values[trajectory_slice], last_value)
        dones = np.append(self.dones[trajectory_slice], 0)
        
        # Compute GAE (Generalized Advantage Estimation)
        deltas = rewards[:-1] + gamma * values[1:] * (1 - dones[1:]) - values[:-1]
        advantages = np.zeros_like(deltas)
        
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + gamma * lam * (1 - dones[t + 1]) * advantage
            advantages[t] = advantage
        
        # Compute returns
        returns = advantages + values[:-1]
        
        # Store computed values
        self.advantages[trajectory_slice] = advantages
        self.returns[trajectory_slice] = returns
        
        self.trajectory_start = self.ptr
    
    def get_batch(self, batch_size=None):
        """Get a batch of experiences."""
        valid_data = min(self.ptr, self.size)
        if batch_size is None:
            batch_size = valid_data
        
        indices = np.random.choice(valid_data, batch_size, replace=False)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'log_probs': torch.FloatTensor(self.log_probs[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices]),
            'values': torch.FloatTensor(self.values[indices])
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.trajectory_start = 0


class PPOAgent(BaseAgent):
    """PPO agent with clipped objective and value function learning."""
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        super().__init__(state_size, action_size, config, "PPO")
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lam', 0.95)  # GAE parameter
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_loss_coeff = config.get('value_loss_coeff', 0.5)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.buffer_size = config.get('buffer_size', 2048)
        
        # Networks
        hidden_size = config.get('hidden_size', 128)
        num_layers = config.get('num_layers', 2)
        
        # Actor network (policy)
        self.actor = PolicyNetwork(
            state_size, action_size, hidden_size, num_layers
        ).to(self.device)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate
        )
        
        # Experience buffer
        self.buffer = PPOBuffer(self.buffer_size, state_size)
        
        # Tracking
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
        print(f"🤖 PPO Agent initialized with {sum(p.numel() for p in self.actor.parameters()) + sum(p.numel() for p in self.critic.parameters())} parameters")
    
    def act(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action using the current policy.
        
        Args:
            state: Current environment state
            training: Whether agent is in training mode
            
        Returns:
            Selected action and additional info if training
        """
        # Preprocess state
        state_array = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get action probabilities and state value
            action_probs = self.actor(state_tensor)
            value = self.critic(state_tensor)
            
            # Create distribution and sample action
            dist = Categorical(action_probs)
            
            if training:
                action = dist.sample()
                log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), value.item(), state_array
            else:
                # Greedy action for evaluation
                action = action_probs.argmax()
                return action.item()
    
    def store_transition(self, state_array, action, reward, log_prob, value, done):
        """Store a transition in the buffer."""
        self.buffer.store(state_array, action, reward, value, log_prob, done)
    
    def finish_trajectory(self, last_value=0):
        """Finish the current trajectory and compute advantages."""
        self.buffer.finish_trajectory(last_value, self.gamma, self.lam)
    
    def learn(self, experience: Tuple = None) -> float:
        """
        Learn from collected experiences using PPO algorithm.
        
        Returns:
            Average loss
        """
        if self.buffer.ptr < self.batch_size:
            return 0.0
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        
        # Multiple epochs of updates
        for epoch in range(self.n_epochs):
            # Get batch of experiences
            batch = self.buffer.get_batch(self.batch_size)
            
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass
            action_probs = self.actor(batch['states'])
            values = self.critic(batch['states']).squeeze()
            
            # Calculate current log probabilities
            dist = Categorical(action_probs)
            current_log_probs = dist.log_prob(batch['actions'])
            entropy = dist.entropy().mean()
            
            # Probability ratios
            ratio = torch.exp(current_log_probs - batch['log_probs'])
            
            # Normalize advantages
            advantages = batch['advantages']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Clipped surrogate objective (policy loss)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = F.mse_loss(values, batch['returns'])
            
            # Entropy loss (for exploration)
            entropy_loss = -entropy
            
            # Total loss
            loss = (policy_loss + 
                   self.value_loss_coeff * value_loss + 
                   self.entropy_coeff * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            
            # Track losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_loss += loss.item()
        
        # Average losses over epochs
        avg_policy_loss = total_policy_loss / self.n_epochs
        avg_value_loss = total_value_loss / self.n_epochs
        avg_entropy_loss = total_entropy_loss / self.n_epochs
        avg_total_loss = total_loss / self.n_epochs
        
        # Store for tracking
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy_loss)
        
        # Clear buffer
        self.buffer.clear()
        
        self.update_training_info()
        
        return avg_total_loss
    
    def save_model(self, filepath: str):
        """Save the agent's model."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 PPO model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the agent's model."""
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        print(f"📂 PPO model loaded from {filepath}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get current training information."""
        return {
            'policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0,
            'value_loss': np.mean(self.value_losses) if self.value_losses else 0,
            'entropy_loss': np.mean(self.entropy_losses) if self.entropy_losses else 0,
            'buffer_size': self.buffer.ptr,
            'training_steps': self.training_step,
            'episodes_completed': self.episode_count
        }


def train_ppo_agent(config: Dict[str, Any], save_dir: str = "models") -> PPOAgent:
    """
    Train a PPO agent on the AgriTech environment.
    
    Args:
        config: Training configuration
        save_dir: Directory to save models
        
    Returns:
        Trained PPO agent
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
    dummy_agent = PPOAgent(0, 0, {})
    state_size = len(dummy_agent.preprocess_state(obs))
    action_size = env.action_space.n
    
    # Create agent
    agent = PPOAgent(state_size, action_size, config)
    
    # Training infrastructure
    logger = TrainingLogger("PPO", config.get('log_dir', 'logs'))
    checkpoint = ModelCheckpoint(save_dir, "PPO")
    
    # Training loop
    max_episodes = config.get('max_episodes', 1000)
    eval_interval = config.get('eval_interval', 50)
    save_interval = config.get('save_interval', 100)
    update_interval = config.get('update_interval', 10)  # Update every N episodes
    
    print(f"🚀 Starting PPO training for {max_episodes} episodes")
    print(f"📊 State size: {state_size}, Action size: {action_size}")
    
    episode_rewards = []
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        total_reward = 0
        
        # Run episode
        for step in range(env.MAX_STEPS):
            # Select action
            action, log_prob, value, state_array = agent.act(obs, training=True)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state_array, action, reward, log_prob, value, done)
            
            total_reward += reward
            obs = next_obs
            info = next_info
            
            if done:
                # Finish trajectory
                agent.finish_trajectory(last_value=0)
                break
        
        episode_rewards.append(total_reward)
        
        # Learn every update_interval episodes
        loss = 0.0
        if (episode + 1) % update_interval == 0:
            loss = agent.learn()
        
        # Calculate metrics
        success_rate = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
        
        # Log episode
        logger.log_episode(episode, total_reward, step + 1, success_rate, loss)
        
        # Print progress
        if episode % 10 == 0:
            training_info = agent.get_training_info()
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Episode {episode:4d}: Reward={total_reward:7.2f}, "
                  f"Avg10={avg_reward:7.2f}, Steps={step+1:3d}, "
                  f"Success={success_rate:.0%}, Loss={loss:.4f}")
        
        # Evaluation
        if episode % eval_interval == 0 and episode > 0:
            avg_reward, success_rate, avg_steps = evaluate_agent(env, agent, num_episodes=5)
            logger.log_evaluation(episode, avg_reward, success_rate, avg_steps)
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            model_path = checkpoint.save_model({
                'actor_state': agent.actor.state_dict(),
                'critic_state': agent.critic.state_dict()
            }, episode, {'avg_reward': np.mean(episode_rewards[-10:])})
            
            # Also save using agent's method
            agent.save_model(os.path.join(save_dir, f"ppo_episode_{episode}.pth"))
        
        agent.update_episode_info()
    
    # Final evaluation and save
    final_avg_reward, final_success_rate, final_avg_steps = evaluate_agent(env, agent, num_episodes=20)
    logger.log_evaluation(max_episodes, final_avg_reward, final_success_rate, final_avg_steps)
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_final.pth")
    agent.save_model(final_model_path)
    
    # Generate training plots
    plot_path = os.path.join(logger.log_dir, "ppo_training_plots.png")
    fig = logger.plot_training_progress(plot_path)
    
    # Save metrics
    metrics_path = os.path.join(logger.log_dir, "ppo_metrics.json")
    logger.save_metrics(metrics_path)
    
    print(f"🎉 PPO training completed!")
    print(f"📊 Final performance: Reward={final_avg_reward:.2f}, "
          f"Success Rate={final_success_rate:.1%}, Steps={final_avg_steps:.1f}")
    
    env.close()
    return agent


if __name__ == "__main__":
    # Test PPO agent
    print("🧪 Testing PPO Agent")
    print("=" * 25)
    
    # Test configuration
    test_config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'lam': 0.95,
        'clip_ratio': 0.2,
        'value_loss_coeff': 0.5,
        'entropy_coeff': 0.01,
        'hidden_size': 128,
        'num_layers': 2,
        'batch_size': 32,
        'n_epochs': 4,
        'buffer_size': 128
    }
    
    # Create test agent
    from base_agent import get_state_action_sizes
    state_size, action_size = get_state_action_sizes()
    
    agent = PPOAgent(state_size, action_size, test_config)
    print(f"✅ PPO agent created successfully")
    
    # Test action selection and trajectory collection
    from custom_env import PrecisionFarmingEnv
    env = PrecisionFarmingEnv()
    obs, _ = env.reset()
    
    # Simulate trajectory
    total_reward = 0
    for step in range(10):  # Short trajectory for testing
        action, log_prob, value, state_array = agent.act(obs, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.store_transition(state_array, action, reward, log_prob, value, terminated or truncated)
        total_reward += reward
        obs = next_obs
        
        if terminated or truncated:
            break
    
    # Finish trajectory and test learning
    agent.finish_trajectory(last_value=0)
    loss = agent.learn()
    print(f"✅ Trajectory learning test: Loss = {loss:.4f}, Total Reward = {total_reward:.2f}")
    
    # Test model saving and loading
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        agent.save_model(f.name)
        agent.load_model(f.name)
        print(f"✅ Model save/load test passed")
        try:
            os.unlink(f.name)
        except:
            pass  # Ignore cleanup error
    
    env.close()
    print("✅ PPO agent ready for training!")
