"""
Hyperparameter Tuned Training Script
Train all algorithms with improved configurations for better performance
"""

import os
import sys
import time
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_improved_training():
    """Train all algorithms with improved hyperparameters."""
    
    print("🔧 AgriTech RL - Hyperparameter Tuned Training")
    print("=" * 52)
    print("Training all algorithms with optimized configurations")
    print()
    
    try:
        from environment.custom_env import PrecisionFarmingEnv
        from agents.dqn_agent import DQNAgent
        from agents.reinforce_agent import REINFORCEAgent
        from agents.ppo_agent import PPOAgent
        from agents.actor_critic_agent import ActorCriticAgent
        
        # Environment setup
        env = PrecisionFarmingEnv()
        state_size = 230
        action_size = 6
        
        # Improved configurations
        configs = {
            'DQN': {
                'learning_rate': 0.0005,
                'gamma': 0.995,
                'epsilon_start': 1.0,
                'epsilon_end': 0.05,
                'epsilon_decay': 0.9995,
                'batch_size': 64,
                'target_update_frequency': 50,
                'memory_size': 20000,
                'hidden_size': 256,
                'num_layers': 3
            },
            'REINFORCE': {
                'learning_rate': 0.0003,
                'baseline_learning_rate': 0.0005,
                'gamma': 0.99,
                'hidden_size': 256,
                'num_layers': 3
            },
            'PPO': {
                'learning_rate': 0.0001,
                'gamma': 0.995,
                'epsilon': 0.15,
                'value_loss_coeff': 1.0,
                'entropy_coeff': 0.02,
                'hidden_size': 512,
                'num_layers': 3,
                'ppo_epochs': 6,
                'batch_size': 128
            },
            'ActorCritic': {
                'actor_lr': 0.0003,
                'critic_lr': 0.001,
                'gamma': 0.995,
                'hidden_size': 256,
                'num_layers': 3,
                'entropy_coeff': 0.02,
                'value_loss_coeff': 1.0
            }
        }
        
        # Training parameters
        episodes = 200
        max_steps = 250
        
        results = {}
        
        print("🚀 Starting Improved Training Session")
        print(f"Episodes: {episodes}, Max Steps: {max_steps}")
        print()
        
        # Train each algorithm
        algorithms = ['DQN', 'REINFORCE', 'PPO', 'ActorCritic']
        
        for i, alg_name in enumerate(algorithms, 1):
            print(f"\n{'='*60}")
            print(f"🤖 {i}/4 - Training {alg_name} with Improved Config")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Create agent with improved config
            if alg_name == 'DQN':
                agent = DQNAgent(state_size, action_size, configs[alg_name])
            elif alg_name == 'REINFORCE':
                agent = REINFORCEAgent(state_size, action_size, configs[alg_name])
            elif alg_name == 'PPO':
                agent = PPOAgent(state_size, action_size, configs[alg_name])
            elif alg_name == 'ActorCritic':
                agent = ActorCriticAgent(state_size, action_size, configs[alg_name])
            
            # Count parameters
            if hasattr(agent, 'q_network'):
                total_params = sum(p.numel() for p in agent.q_network.parameters())
            elif hasattr(agent, 'policy_network'):
                total_params = sum(p.numel() for p in agent.policy_network.parameters())
                if hasattr(agent, 'baseline_network'):
                    total_params += sum(p.numel() for p in agent.baseline_network.parameters())
            elif hasattr(agent, 'actor'):
                total_params = sum(p.numel() for p in agent.actor.parameters())
                total_params += sum(p.numel() for p in agent.critic.parameters())
            elif hasattr(agent, 'network'):
                total_params = sum(p.numel() for p in agent.network.parameters())
            
            print(f"🧠 {alg_name} Agent: {total_params:,} parameters")
            
            # Training loop
            episode_rewards = []
            success_count = 0
            best_reward = float('-inf')
            
            for episode in range(episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                # Algorithm-specific training
                if alg_name == 'DQN':
                    for step in range(max_steps):
                        action = agent.act(obs, training=True)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        agent.remember(obs, action, reward, next_obs, terminated or truncated)
                        
                        if len(agent.memory) >= agent.batch_size:
                            agent.learn()
                        
                        obs = next_obs
                        episode_reward += reward
                        
                        if terminated or truncated:
                            if info.get('completion_rate', 0) >= 1.0:
                                success_count += 1
                            break
                
                elif alg_name == 'REINFORCE':
                    for step in range(max_steps):
                        action = agent.act(obs, training=True)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        agent.remember_reward(reward)
                        
                        obs = next_obs
                        episode_reward += reward
                        
                        if terminated or truncated:
                            if info.get('completion_rate', 0) >= 1.0:
                                success_count += 1
                            break
                    
                    agent.learn()
                
                elif alg_name == 'PPO':
                    for step in range(max_steps):
                        action, log_prob, value, state_array = agent.act(obs, training=True)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                        
                        agent.store_transition(state_array, action, reward, log_prob, value, done)
                        
                        obs = next_obs
                        episode_reward += reward
                        
                        if done:
                            if info.get('completion_rate', 0) >= 1.0:
                                success_count += 1
                            final_value = agent.critic(torch.FloatTensor(agent.preprocess_state(obs)).unsqueeze(0).to(agent.device)).item()
                            agent.finish_trajectory(final_value)
                            break
                    
                    if len(agent.buffer.states) >= agent.batch_size:
                        agent.learn()
                
                elif alg_name == 'ActorCritic':
                    for step in range(max_steps):
                        action = agent.act(obs, training=True)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        agent.remember_reward(reward)
                        
                        if len(agent.episode_rewards) > 0:
                            agent.learn()
                        
                        obs = next_obs
                        episode_reward += reward
                        
                        if terminated or truncated:
                            if info.get('completion_rate', 0) >= 1.0:
                                success_count += 1
                            break
                
                episode_rewards.append(episode_reward)
                best_reward = max(best_reward, episode_reward)
                
                # Progress reporting
                if episode % 50 == 0:
                    avg_reward = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
                    success_rate = (success_count / (episode + 1)) * 100
                    print(f"Episode {episode:3d}: Reward={episode_reward:6.2f}, "
                          f"Avg50={avg_reward:6.2f}, Success={success_rate:4.1f}%")
            
            training_time = time.time() - start_time
            final_avg_reward = sum(episode_rewards) / len(episode_rewards)
            final_success_rate = (success_count / episodes) * 100
            
            # Save improved model
            model_path = f"models/{alg_name.lower()}_improved.pth"
            os.makedirs("models", exist_ok=True)
            agent.save_model(model_path)
            
            # Store results
            results[alg_name] = {
                'avg_reward': final_avg_reward,
                'best_reward': best_reward,
                'success_rate': final_success_rate,
                'training_time': training_time,
                'parameters': total_params
            }
            
            print(f"\n✅ {alg_name} Improved Training Complete:")
            print(f"   Average Reward: {final_avg_reward:.2f}")
            print(f"   Best Reward: {best_reward:.2f}")
            print(f"   Success Rate: {final_success_rate:.1f}%")
            print(f"   Training Time: {training_time:.1f}s")
            print(f"   Parameters: {total_params:,}")
            print(f"   Model saved: {model_path}")
        
        env.close()
        
        # Summary comparison
        print(f"\n🏆 Improved Training Results Summary")
        print("=" * 50)
        for alg, result in results.items():
            print(f"{alg:12}: {result['avg_reward']:6.2f} reward, "
                  f"{result['success_rate']:4.1f}% success, "
                  f"{result['training_time']:5.1f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in improved training: {str(e)}")
        return None

if __name__ == "__main__":
    results = run_improved_training()
    if results:
        print("\n🎉 All improved training completed successfully!")
    else:
        print("\n❌ Improved training failed.")
