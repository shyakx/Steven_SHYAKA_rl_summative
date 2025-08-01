"""
Training Infrastructure for AgriTech RL Agents

Unified training framework supporting multiple RL algorithms
with hyperparameter management, logging, and model checkpointing.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pickle
from collections import defaultdict

# Add environment to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environment'))

from custom_env import PrecisionFarmingEnv


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    
    # Environment settings
    env_name: str = "PrecisionFarming"
    max_episodes: int = 1000
    max_steps_per_episode: int = 200
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Neural network settings
    hidden_size: int = 128
    num_layers: int = 2
    
    # Logging settings
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    eval_episodes: int = 5
    
    # Model saving
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # Device settings
    device: str = "cpu"  # Will auto-detect GPU if available


class TrainingLogger:
    """Handles logging and visualization for RL training."""
    
    def __init__(self, agent_name: str, log_dir: str = "logs"):
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, f"{agent_name}_training.log")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success_rates = []
        self.loss_history = []
        self.epsilon_history = []
        
        # Evaluation metrics
        self.eval_rewards = []
        self.eval_success_rates = []
        self.eval_efficiency = []  # Steps to completion
        
        # Performance tracking
        self.start_time = time.time()
        self.training_time = 0
        
    def log_episode(self, episode: int, reward: float, length: int, 
                   success_rate: float, loss: float = None, epsilon: float = None):
        """Log metrics for a training episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_success_rates.append(success_rate)
        
        if loss is not None:
            self.loss_history.append(loss)
        if epsilon is not None:
            self.epsilon_history.append(epsilon)
        
        # Write to log file
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (f"[{timestamp}] Episode {episode:4d}: "
                    f"Reward={reward:7.2f}, Length={length:3d}, "
                    f"Success={success_rate:.1%}")
        
        if loss is not None:
            log_entry += f", Loss={loss:.4f}"
        if epsilon is not None:
            log_entry += f", Epsilon={epsilon:.3f}"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def log_evaluation(self, episode: int, avg_reward: float, 
                      success_rate: float, avg_steps: float):
        """Log evaluation metrics."""
        self.eval_rewards.append(avg_reward)
        self.eval_success_rates.append(success_rate)
        self.eval_efficiency.append(avg_steps)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (f"[{timestamp}] EVAL Episode {episode:4d}: "
                    f"Avg Reward={avg_reward:7.2f}, "
                    f"Success Rate={success_rate:.1%}, "
                    f"Avg Steps={avg_steps:.1f}")
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        
        print(f"🔍 Evaluation: {log_entry.split('] ')[1]}")
    
    def plot_training_progress(self, save_path: str = None):
        """Generate training progress plots."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{self.agent_name} Training Progress", fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) > 50:
            # Moving average
            window = min(50, len(self.episode_rewards) // 10)
            moving_avg = np.convolve(self.episode_rewards, 
                                   np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.episode_rewards)), 
                           moving_avg, 'r-', label=f'Moving Avg ({window})')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths, alpha=0.7, color='green')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success rates
        axes[0, 2].plot(self.episode_success_rates, alpha=0.7, color='blue')
        axes[0, 2].set_title('Success Rates')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Success Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Loss history
        if self.loss_history:
            axes[1, 0].plot(self.loss_history, alpha=0.7, color='red')
            axes[1, 0].set_title('Training Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center',
                           transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Training Loss (N/A)')
        
        # Epsilon decay
        if self.epsilon_history:
            axes[1, 1].plot(self.epsilon_history, alpha=0.7, color='orange')
            axes[1, 1].set_title('Epsilon Decay')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Epsilon Data', ha='center', va='center',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Epsilon Decay (N/A)')
        
        # Evaluation metrics
        if self.eval_rewards:
            eval_episodes = range(0, len(self.eval_rewards) * 50, 50)  # Assuming eval every 50 episodes
            axes[1, 2].plot(eval_episodes, self.eval_rewards, 'ro-', alpha=0.7, label='Eval Reward')
            ax2 = axes[1, 2].twinx()
            ax2.plot(eval_episodes, self.eval_success_rates, 'bo-', alpha=0.7, label='Success Rate')
            axes[1, 2].set_title('Evaluation Performance')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Avg Reward', color='red')
            ax2.set_ylabel('Success Rate', color='blue')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Evaluation Data', ha='center', va='center',
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Evaluation (N/A)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training plots saved to {save_path}")
        
        return fig
    
    def save_metrics(self, save_path: str):
        """Save all metrics to file."""
        metrics = {
            'agent_name': self.agent_name,
            'training_time': time.time() - self.start_time,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_success_rates': self.episode_success_rates,
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': self.eval_success_rates,
            'eval_efficiency': self.eval_efficiency
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"💾 Metrics saved to {save_path}")


class ModelCheckpoint:
    """Handles model saving and loading."""
    
    def __init__(self, model_dir: str, agent_name: str):
        self.model_dir = model_dir
        self.agent_name = agent_name
        os.makedirs(model_dir, exist_ok=True)
    
    def save_model(self, model_state: Dict, episode: int, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'episode': episode,
            'model_state': model_state,
            'timestamp': time.time(),
            'metrics': metrics or {}
        }
        
        checkpoint_path = os.path.join(
            self.model_dir, f"{self.agent_name}_episode_{episode}.pkl"
        )
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save as latest
        latest_path = os.path.join(
            self.model_dir, f"{self.agent_name}_latest.pkl"
        )
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"💾 Model saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_model(self, checkpoint_path: str = None):
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.model_dir, f"{self.agent_name}_latest.pkl"
            )
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ No checkpoint found at {checkpoint_path}")
            return None
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"📂 Model loaded from {checkpoint_path}")
        return checkpoint


def evaluate_agent(env: PrecisionFarmingEnv, agent, num_episodes: int = 5, 
                  render: bool = False) -> Tuple[float, float, float]:
    """
    Evaluate an agent's performance.
    
    Returns:
        avg_reward: Average reward per episode
        success_rate: Percentage of episodes completed successfully
        avg_steps: Average steps to completion
    """
    total_reward = 0
    total_steps = 0
    successes = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while episode_steps < env.MAX_STEPS:
            # Get action from agent
            if hasattr(agent, 'act'):
                action = agent.act(obs, training=False)
            else:
                # Fallback for function-based agents
                action = agent(obs, env)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if render:
                env.render()
                time.sleep(0.1)
            
            if terminated or truncated:
                if info.get('current_diseased_count', 0) == 0:
                    successes += 1
                break
        
        total_reward += episode_reward
        total_steps += episode_steps
    
    avg_reward = total_reward / num_episodes
    success_rate = successes / num_episodes
    avg_steps = total_steps / num_episodes
    
    return avg_reward, success_rate, avg_steps


def compare_agents(results_dict: Dict[str, Dict], save_path: str = None):
    """Generate comparison plots for multiple agents."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('RL Agents Comparison - AgriTech Precision Farming', fontsize=16)
    
    # Collect data for comparison
    agent_names = list(results_dict.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # Learning curves
    for i, (agent_name, metrics) in enumerate(results_dict.items()):
        if 'episode_rewards' in metrics:
            rewards = metrics['episode_rewards']
            # Smooth the rewards for better visualization
            if len(rewards) > 50:
                window = min(50, len(rewards) // 10)
                smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(rewards)), smooth_rewards, 
                               color=colors[i % len(colors)], label=agent_name, linewidth=2)
            else:
                axes[0, 0].plot(rewards, color=colors[i % len(colors)], 
                               label=agent_name, linewidth=2)
    
    axes[0, 0].set_title('Learning Curves (Smoothed Rewards)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Success rates over time
    for i, (agent_name, metrics) in enumerate(results_dict.items()):
        if 'episode_success_rates' in metrics:
            success_rates = metrics['episode_success_rates']
            axes[0, 1].plot(success_rates, color=colors[i % len(colors)], 
                           label=agent_name, linewidth=2, alpha=0.7)
    
    axes[0, 1].set_title('Success Rates Over Time')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final performance comparison (box plots)
    final_rewards = []
    final_success_rates = []
    
    for agent_name, metrics in results_dict.items():
        if 'episode_rewards' in metrics and len(metrics['episode_rewards']) > 100:
            # Take last 100 episodes for final performance
            final_rewards.append(metrics['episode_rewards'][-100:])
        if 'episode_success_rates' in metrics and len(metrics['episode_success_rates']) > 100:
            final_success_rates.append(metrics['episode_success_rates'][-100:])
    
    if final_rewards:
        axes[1, 0].boxplot(final_rewards, labels=agent_names)
        axes[1, 0].set_title('Final Performance (Last 100 Episodes)')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Evaluation performance
    eval_data = []
    eval_labels = []
    for agent_name, metrics in results_dict.items():
        if 'eval_rewards' in metrics and metrics['eval_rewards']:
            eval_data.append(np.mean(metrics['eval_rewards']))
            eval_labels.append(agent_name)
    
    if eval_data:
        bars = axes[1, 1].bar(eval_labels, eval_data, color=colors[:len(eval_data)])
        axes[1, 1].set_title('Average Evaluation Performance')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, eval_data):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test the training infrastructure
    print("🧪 Testing Training Infrastructure")
    print("=" * 40)
    
    # Test configuration
    config = TrainingConfig(max_episodes=10)
    print(f"✅ Configuration created: {config.env_name}")
    
    # Test logger
    logger = TrainingLogger("TestAgent", "logs")
    
    # Simulate training data
    for episode in range(10):
        reward = np.random.normal(20, 5)
        length = np.random.randint(50, 150)
        success_rate = min(1.0, episode * 0.1 + np.random.uniform(0, 0.2))
        loss = np.random.exponential(0.1)
        epsilon = max(0.01, 1.0 - episode * 0.1)
        
        logger.log_episode(episode, reward, length, success_rate, loss, epsilon)
    
    print("✅ Logger tested successfully")
    
    # Test model checkpoint
    checkpoint = ModelCheckpoint("models", "TestAgent")
    dummy_state = {"weights": [1, 2, 3], "optimizer": "adam"}
    checkpoint.save_model(dummy_state, 10, {"reward": 25.5})
    
    loaded = checkpoint.load_model()
    if loaded:
        print("✅ Model checkpoint tested successfully")
    
    # Generate test plots
    fig = logger.plot_training_progress("logs/test_training_plot.png")
    plt.close(fig)
    
    print("✅ Training infrastructure ready!")
    print("\n🚀 Ready to implement RL agents!")


def plot_agent_comparison(agent_names: List[str], evaluation_results: Dict[str, Dict[str, float]], save_path: str = None):
    """
    Create comparison plots for multiple agents.
    
    Args:
        agent_names: List of agent names
        evaluation_results: Dictionary containing evaluation metrics for each agent
        save_path: Path to save the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter out agents with errors
    valid_results = {name: results for name, results in evaluation_results.items() 
                    if 'error' not in results and name in agent_names}
    
    if not valid_results:
        print("No valid results to plot")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('RL Agents Performance Comparison', fontsize=16, fontweight='bold')
    
    agents = list(valid_results.keys())
    metrics = ['avg_reward', 'success_rate', 'avg_steps', 'efficiency_score']
    titles = ['Average Reward', 'Success Rate (%)', 'Average Steps', 'Efficiency (Reward/Step)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        values = [valid_results[agent].get(metric, 0) for agent in agents]
        
        # Convert success rate to percentage
        if metric == 'success_rate':
            values = [v * 100 for v in values]
        
        bars = ax.bar(agents, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(agents)])
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to {save_path}")
    
    return fig
