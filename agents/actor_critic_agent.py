"""
Actor-Critic Agent for AgriTech Precision Farming

Classic Actor-Critic algorithm with advantage estimation.
Actor learns the policy while Critic learns the value function.
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
    from .base_agent import BaseAgent, ActorCriticNetwork
except ImportError:
    from base_agent import BaseAgent, ActorCriticNetwork


class ActorCriticAgent(BaseAgent):
    """Actor-Critic agent with advantage estimation."""
    
    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any]):
        super().__init__(state_size, action_size, config, "ActorCritic")
        
        # Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.gamma = config.get('gamma', 0.99)
        self.value_loss_coeff = config.get('value_loss_coeff', 0.5)
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Shared network with separate actor and critic heads
        self.network = ActorCriticNetwork(
            state_size,
            action_size,
            config.get('hidden_size', 128),
            config.get('num_layers', 2)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        # Tracking
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
        print(f"🤖 Actor-Critic Agent initialized with {sum(p.numel() for p in self.network.parameters())} parameters")
    
    def act(self, state: Dict[str, Any], training: bool = True) -> int:
        """
        Select action using the current policy and estimate state value.
        
        Args:
            state: Current environment state
            training: Whether agent is in training mode
            
        Returns:
            Selected action
        """
        # Preprocess state
        state_array = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        # Get action probabilities and state value
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs, value = self.network(state_tensor)
            
            if training:
                # Sample from distribution during training
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Store for learning
                self.episode_states.append(state_array)
                self.episode_actions.append(action.item())
                self.episode_values.append(value.item())
                self.episode_log_probs.append(log_prob)
                
                return action.item()
            else:
                # Greedy action selection during evaluation
                return action_probs.argmax().item()
    
    def remember_reward(self, reward: float):
        """Store reward for current step."""
        self.episode_rewards.append(reward)
    
    def learn(self, experience: Tuple = None) -> float:
        """
        Learn from a complete episode using Actor-Critic algorithm.
        
        Returns:
            Total loss value
        """
        if len(self.episode_rewards) == 0:
            return 0.0
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        states = torch.FloatTensor(np.array(self.episode_states)).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        values = torch.FloatTensor(self.episode_values).to(self.device)
        log_probs = torch.stack(self.episode_log_probs).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages (TD error)
        advantages = returns - values.detach()
        
        # Get current action probabilities and values
        current_action_probs, current_values = self.network(states)
        current_values = current_values.squeeze()
        
        # Calculate current log probabilities
        dist = Categorical(current_action_probs)
        current_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Actor loss (policy gradient with advantage)
        actor_loss = -(current_log_probs * advantages.detach()).mean()
        
        # Critic loss (value function estimation)
        critic_loss = F.mse_loss(current_values, returns)
        
        # Entropy loss (for exploration)
        entropy_loss = -entropy
        
        # Total loss
        total_loss = (actor_loss + 
                     self.value_loss_coeff * critic_loss + 
                     self.entropy_coeff * entropy_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Store losses for tracking
        self.policy_losses.append(actor_loss.item())
        self.value_losses.append(critic_loss.item())
        self.entropy_losses.append(entropy_loss.item())
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        self.episode_log_probs.clear()
        
        self.update_training_info()
        
        return total_loss.item()
    
    def get_state_value(self, state: Dict[str, Any]) -> float:
        """Get the estimated value of a state."""
        state_array = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.network(state_tensor)
            return value.item()
    
    def save_model(self, filepath: str):
        """Save the agent's model."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Actor-Critic model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the agent's model."""
        if not os.path.exists(filepath):
            print(f"❌ Model file not found: {filepath}")
            return
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        print(f"📂 Actor-Critic model loaded from {filepath}")
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get current training information."""
        return {
            'actor_loss': np.mean(self.policy_losses) if self.policy_losses else 0,
            'critic_loss': np.mean(self.value_losses) if self.value_losses else 0,
            'entropy_loss': np.mean(self.entropy_losses) if self.entropy_losses else 0,
            'episode_length': len(self.episode_rewards),
            'training_steps': self.training_step,
            'episodes_completed': self.episode_count
        }


def train_actor_critic_agent(config: Dict[str, Any], save_dir: str = "models") -> ActorCriticAgent:
    """
    Train an Actor-Critic agent on the AgriTech environment.
    
    Args:
        config: Training configuration
        save_dir: Directory to save models
        
    Returns:
        Trained Actor-Critic agent
    """
    from trainer import TrainingLogger, ModelCheckpoint, evaluate_agent
    from custom_env import PrecisionFarmingEnv
    
    # Create environment
    env = PrecisionFarmingEnv()
    
    # Get state and action sizes
    obs, _ = env.reset()
    dummy_agent = ActorCriticAgent(0, 0, {})
    state_size = len(dummy_agent.preprocess_state(obs))
    action_size = env.action_space.n
    
    # Create agent
    agent = ActorCriticAgent(state_size, action_size, config)
    
    # Training infrastructure
    logger = TrainingLogger("ActorCritic", config.get('log_dir', 'logs'))
    checkpoint = ModelCheckpoint(save_dir, "ActorCritic")
    
    # Training loop
    max_episodes = config.get('max_episodes', 1000)
    eval_interval = config.get('eval_interval', 50)
    save_interval = config.get('save_interval', 100)
    
    print(f"🚀 Starting Actor-Critic training for {max_episodes} episodes")
    print(f"📊 State size: {state_size}, Action size: {action_size}")
    
    for episode in range(max_episodes):
        obs, info = env.reset()
        total_reward = 0
        
        # Run episode
        for step in range(env.MAX_STEPS):
            # Select action
            action = agent.act(obs, training=True)
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store reward
            agent.remember_reward(reward)
            total_reward += reward
            
            obs = next_obs
            info = next_info
            
            if done:
                break
        
        # Learn from episode
        loss = agent.learn()
        
        # Calculate metrics
        success_rate = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
        
        # Log episode
        logger.log_episode(episode, total_reward, step + 1, success_rate, loss)
        
        # Print progress
        if episode % 10 == 0:
            training_info = agent.get_training_info()
            print(f"Episode {episode:4d}: Reward={total_reward:7.2f}, "
                  f"Steps={step+1:3d}, Success={success_rate:.0%}, "
                  f"Loss={loss:.4f}, Actor={training_info['actor_loss']:.4f}, "
                  f"Critic={training_info['critic_loss']:.4f}")
        
        # Evaluation
        if episode % eval_interval == 0 and episode > 0:
            avg_reward, success_rate, avg_steps = evaluate_agent(env, agent, num_episodes=5)
            logger.log_evaluation(episode, avg_reward, success_rate, avg_steps)
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            model_path = checkpoint.save_model({
                'network_state': agent.network.state_dict()
            }, episode, {'avg_reward': total_reward})
            
            # Also save using agent's method
            agent.save_model(os.path.join(save_dir, f"actor_critic_episode_{episode}.pth"))
        
        agent.update_episode_info()
    
    # Final evaluation and save
    final_avg_reward, final_success_rate, final_avg_steps = evaluate_agent(env, agent, num_episodes=20)
    logger.log_evaluation(max_episodes, final_avg_reward, final_success_rate, final_avg_steps)
    
    # Save final model
    final_model_path = os.path.join(save_dir, "actor_critic_final.pth")
    agent.save_model(final_model_path)
    
    # Generate training plots
    plot_path = os.path.join(logger.log_dir, "actor_critic_training_plots.png")
    fig = logger.plot_training_progress(plot_path)
    
    # Save metrics
    metrics_path = os.path.join(logger.log_dir, "actor_critic_metrics.json")
    logger.save_metrics(metrics_path)
    
    print(f"🎉 Actor-Critic training completed!")
    print(f"📊 Final performance: Reward={final_avg_reward:.2f}, "
          f"Success Rate={final_success_rate:.1%}, Steps={final_avg_steps:.1f}")
    
    env.close()
    return agent


if __name__ == "__main__":
    # Test Actor-Critic agent
    print("🧪 Testing Actor-Critic Agent")
    print("=" * 35)
    
    # Test configuration
    test_config = {
        'learning_rate': 0.001,
        'gamma': 0.99,
        'value_loss_coeff': 0.5,
        'entropy_coeff': 0.01,
        'hidden_size': 128,
        'num_layers': 2
    }
    
    # Create test agent
    from base_agent import get_state_action_sizes
    state_size, action_size = get_state_action_sizes()
    
    agent = ActorCriticAgent(state_size, action_size, test_config)
    print(f"✅ Actor-Critic agent created successfully")
    
    # Test action selection and episode learning
    from custom_env import PrecisionFarmingEnv
    env = PrecisionFarmingEnv()
    obs, _ = env.reset()
    
    # Simulate episode
    total_reward = 0
    for step in range(10):  # Short episode for testing
        action = agent.act(obs, training=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.remember_reward(reward)
        total_reward += reward
        obs = next_obs
        
        if terminated or truncated:
            break
    
    # Test learning
    loss = agent.learn()
    print(f"✅ Episode learning test: Loss = {loss:.4f}, Total Reward = {total_reward:.2f}")
    
    # Test state value estimation
    obs, _ = env.reset()
    value = agent.get_state_value(obs)
    print(f"✅ State value estimation: Value = {value:.4f}")
    
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
    print("✅ Actor-Critic agent ready for training!")
