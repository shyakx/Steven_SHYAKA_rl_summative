"""
Final Optimized Training - AgriTech RL Algorithms
Enhanced hyperparameters with Actor-Critic fix for maximum performance
"""

import os
import sys
import time
import torch
import json
import numpy as np
from datetime import datetime

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'environment'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))

from environment.custom_env_clean import PrecisionFarmingEnv
from agents.dqn_agent import DQNAgent
from agents.reinforce_agent import REINFORCEAgent  
from agents.ppo_agent import PPOAgent
from agents.actor_critic_agent import ActorCriticAgent
from agents.base_agent import BaseAgent

def create_final_optimized_configs():
    """Create the best possible configurations for each algorithm."""
    
    # Enhanced DQN configuration
    dqn_config = {
        'learning_rate': 0.0003,        # Slightly reduced for stability
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'hidden_size': 512,             # Increased for better capacity
        'num_layers': 3,                # Deeper network
        'batch_size': 128,              # Larger batches
        'buffer_size': 100000,          # Larger replay buffer
        'target_update_freq': 50,       # More frequent updates
        'memory_size': 100000
    }
    
    # Enhanced REINFORCE configuration  
    reinforce_config = {
        'learning_rate': 0.0001,        # Lower for stability
        'gamma': 0.995,                 # Higher discount for long-term planning
        'hidden_size': 512,             # Larger network
        'num_layers': 3,
        'baseline': True,
        'entropy_coeff': 0.01           # Small entropy bonus
    }
    
    # Enhanced PPO configuration
    ppo_config = {
        'learning_rate': 0.00005,       # Very low for stable updates
        'gamma': 0.99,
        'clip_epsilon': 0.15,           # Tighter clipping
        'gae_lambda': 0.98,             # Higher GAE lambda
        'value_loss_coeff': 0.25,       # Reduced value loss weight
        'entropy_coeff': 0.02,          # Higher entropy for exploration
        'hidden_size': 1024,            # Very large network
        'num_layers': 3,
        'batch_size': 128,
        'n_epochs': 8,                  # More update epochs
        'max_grad_norm': 0.5
    }
    
    # Enhanced Actor-Critic configuration (with broadcasting fix)
    actor_critic_config = {
        'actor_learning_rate': 0.0001,  # Lower actor LR
        'critic_learning_rate': 0.0005, # Higher critic LR
        'gamma': 0.99,
        'hidden_size': 512,             # Larger networks
        'num_layers': 3,
        'value_loss_coeff': 0.5,
        'entropy_coeff': 0.01,
        'max_grad_norm': 0.5
    }
    
    return {
        'DQN': dqn_config,
        'REINFORCE': reinforce_config,
        'PPO': ppo_config,
        'ActorCritic': actor_critic_config
    }

def evaluate_agent(env, agent, num_episodes=10):
    """Evaluate agent performance."""
    total_rewards = []
    success_count = 0
    total_steps = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 300:  # Max steps per episode
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                if info.get('current_diseased_count', 1) == 0:
                    success_count += 1
                break
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
    
    avg_reward = np.mean(total_rewards)
    success_rate = (success_count / num_episodes) * 100
    avg_steps = np.mean(total_steps)
    
    return avg_reward, success_rate, avg_steps

def train_algorithm(algorithm_name, config, max_episodes=300, save_dir="models"):
    """Train a single algorithm with enhanced configuration."""
    print(f"🏆 Training {algorithm_name} with Final Optimized Config")
    print("=" * 60)
    
    # Create environment
    env = PrecisionFarmingEnv()
    
    # Get state and action sizes
    obs, _ = env.reset()
    from agents.base_agent import get_state_action_sizes
    state_size, action_size = get_state_action_sizes()
    
    # Create agent
    if algorithm_name == 'DQN':
        agent = DQNAgent(state_size, action_size, config)
    elif algorithm_name == 'REINFORCE':
        agent = REINFORCEAgent(state_size, action_size, config)
    elif algorithm_name == 'PPO':
        agent = PPOAgent(state_size, action_size, config)
    elif algorithm_name == 'ActorCritic':
        agent = ActorCriticAgent(state_size, action_size, config)
    
    # Print agent info
    param_count = 0
    if hasattr(agent, 'q_network'):
        param_count = sum(p.numel() for p in agent.q_network.parameters())
    elif hasattr(agent, 'policy_network'):
        param_count = sum(p.numel() for p in agent.policy_network.parameters())
        if hasattr(agent, 'baseline_network'):
            param_count += sum(p.numel() for p in agent.baseline_network.parameters())
    elif hasattr(agent, 'actor') and hasattr(agent, 'critic'):
        param_count = sum(p.numel() for p in agent.actor.parameters())
        param_count += sum(p.numel() for p in agent.critic.parameters())
    elif hasattr(agent, 'network'):
        param_count = sum(p.numel() for p in agent.network.parameters())
    
    print(f"🖥️ {algorithm_name} using device: {agent.device}")
    print(f"🧠 {algorithm_name} Agent: {param_count:,} parameters")
    
    # Training loop
    start_time = time.time()
    rewards_history = []
    success_history = []
    
    for episode in range(max_episodes):
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        while steps < 250:  # Max steps per episode
            if algorithm_name == 'PPO':
                action, log_prob, value, state_array = agent.act(obs, training=True)
            else:
                action = agent.act(obs, training=True)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Store experience
            if algorithm_name == 'DQN':
                agent.remember(obs, action, reward, next_obs, terminated or truncated)
            elif algorithm_name == 'PPO':
                agent.store_transition(state_array, action, reward, log_prob, value, terminated or truncated)
            else:  # REINFORCE, ActorCritic
                agent.remember_reward(reward)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Learning step
        if algorithm_name == 'PPO':
            agent.finish_trajectory(last_value=0)
        
        loss = agent.learn()
        
        # Track metrics
        rewards_history.append(total_reward)
        success = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
        success_history.append(success)
        
        # Progress reporting
        if episode % 50 == 0:
            avg_reward_50 = np.mean(rewards_history[-50:])
            success_rate_50 = np.mean(success_history[-50:]) * 100
            print(f"Episode {episode:3d}: Reward={total_reward:.2f}, Avg50={avg_reward_50:.2f}, Success={success_rate_50:.1f}%")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_avg_reward, final_success_rate, final_avg_steps = evaluate_agent(env, agent, num_episodes=20)
    best_reward = max(rewards_history)
    avg_reward = np.mean(rewards_history)
    
    # Save model
    model_filename = f"{algorithm_name.lower()}_final_optimized.pth"
    model_path = os.path.join(save_dir, model_filename)
    agent.save_model(model_path)
    
    print(f"💾 {algorithm_name} model saved to {model_path}")
    print(f"✅ {algorithm_name} Final Optimized Training Complete:")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Best Reward: {best_reward:.2f}")  
    print(f"   Success Rate: {final_success_rate:.1f}%")
    print(f"   Training Time: {training_time:.1f}s")
    print(f"   Parameters: {param_count:,}")
    print(f"   Model saved: {model_path}")
    
    env.close()
    
    return {
        'algorithm': algorithm_name,
        'avg_reward': avg_reward,
        'best_reward': best_reward,
        'final_success_rate': final_success_rate,
        'training_time': training_time,
        'parameters': param_count,
        'model_path': model_path
    }

def main():
    """Main training pipeline with final optimized configurations."""
    print("🏆 AgriTech RL - Final Optimized Training")
    print("=" * 60)
    print("Enhanced hyperparameters with Actor-Critic fix")
    print("🚀 Starting Final Training Session")
    print("Episodes: 300, Max Steps: 250")
    print("=" * 60)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Get configurations
    configs = create_final_optimized_configs()
    
    # Train all algorithms
    results = []
    algorithms = ['DQN', 'REINFORCE', 'PPO', 'ActorCritic']
    
    for i, algorithm in enumerate(algorithms, 1):
        print(f"🤖 {i}/4 - Training {algorithm} with Final Config")
        print("=" * 60)
        
        try:
            result = train_algorithm(algorithm, configs[algorithm], max_episodes=300)
            results.append(result)
            
        except Exception as e:
            print(f"❌ Error training {algorithm}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Results summary
    print(f"\n🏆 Final Optimized Training Results Summary")
    print("=" * 60)
    
    # Sort by success rate
    results.sort(key=lambda x: x['final_success_rate'], reverse=True)
    
    for result in results:
        print(f"{result['algorithm']:12}: {result['avg_reward']:6.2f} reward, "
              f"{result['final_success_rate']:4.1f}% success, {result['training_time']:6.1f}s")
    
    # Save results
    results_path = "analysis/final_optimized_results.json"
    os.makedirs("analysis", exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'training_episodes': 300,
            'max_steps_per_episode': 250,
            'results': results
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_path}")
    print("🎉 All final optimized training completed successfully!")

if __name__ == "__main__":
    main()
