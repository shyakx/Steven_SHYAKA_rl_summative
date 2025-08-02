"""
Individual REINFORCE Training Script
Run REINFORCE algorithm alone to see its performance in the AgriTech environment
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_reinforce_training():
    """Train REINFORCE agent and demonstrate its performance."""
    
    print("🎯 REINFORCE Algorithm - Individual Training")
    print("=" * 48)
    print("Training Policy Gradient (REINFORCE) on AgriTech Environment")
    print()
    
    try:
        from environment.custom_env import PrecisionFarmingEnv
        from agents.reinforce_agent import REINFORCEAgent
        from training.trainer import TrainingLogger, evaluate_agent
        
        # Create environment
        env = PrecisionFarmingEnv()
        state_size = 230
        action_size = 6
        
        # REINFORCE Configuration
        config = {
            'learning_rate': 0.001,
            'baseline_learning_rate': 0.001,
            'gamma': 0.99,
            'hidden_size': 128,
            'num_layers': 2
        }
        
        print("🔧 REINFORCE Configuration:")
        print(f"   Policy Learning Rate: {config['learning_rate']}")
        print(f"   Baseline Learning Rate: {config['baseline_learning_rate']}")
        print(f"   Discount Factor: {config['gamma']}")
        print(f"   Hidden Units: {config['hidden_size']}")
        print()
        
        # Create REINFORCE agent
        agent = REINFORCEAgent(state_size, action_size, config)
        policy_params = sum(p.numel() for p in agent.policy_network.parameters())
        baseline_params = sum(p.numel() for p in agent.baseline_network.parameters())
        total_params = policy_params + baseline_params
        print(f"🧠 REINFORCE Agent initialized:")
        print(f"   Policy Network: {policy_params:,} parameters")
        print(f"   Baseline Network: {baseline_params:,} parameters")
        print(f"   Total: {total_params:,} parameters")
        
        # Training setup
        episodes = 150
        max_steps = 200
        
        print(f"\n🚀 Starting REINFORCE Training ({episodes} episodes)")
        print("=" * 55)
        
        start_time = time.time()
        episode_rewards = []
        episode_steps = []
        success_count = 0
        
        for episode in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            # Collect episode trajectory
            for step in range(max_steps):
                action = agent.act(obs, training=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store reward for this step
                agent.remember_reward(reward)
                
                obs = next_obs
                episode_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if info.get('completion_rate', 0) >= 1.0:
                        success_count += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            
            # Learn from episode
            agent.learn()
            
            # Progress reporting
            if episode % 25 == 0:
                avg_reward = sum(episode_rewards[-25:]) / min(25, len(episode_rewards))
                success_rate = (success_count / (episode + 1)) * 100
                print(f"Episode {episode:3d}: Reward={episode_reward:6.2f}, "
                      f"Avg25={avg_reward:6.2f}, Success={success_rate:4.1f}%, "
                      f"Steps={steps:3d}")
        
        training_time = time.time() - start_time
        
        print("\n📊 REINFORCE Training Completed!")
        print("-" * 35)
        print(f"Training Time: {training_time:.1f} seconds")
        print(f"Final Success Rate: {(success_count/episodes)*100:.1f}%")
        print(f"Average Reward: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"Best Episode Reward: {max(episode_rewards):.2f}")
        
        # Save model
        model_path = "models/reinforce_individual.pth"
        os.makedirs("models", exist_ok=True)
        agent.save_model(model_path)
        print(f"💾 Model saved to {model_path}")
        
        # Visual Demo
        print(f"\n🎮 REINFORCE Visual Demo (3 episodes with rendering)")
        print("-" * 50)
        
        # Create environment with rendering enabled
        demo_env = PrecisionFarmingEnv(render_mode='human')
        
        for demo_ep in range(3):
            obs, _ = demo_env.reset()
            total_reward = 0
            print(f"\n🎯 Demo Episode {demo_ep + 1}/3 - Watch the REINFORCE agent play!")
            
            for step in range(max_steps):
                action = agent.act(obs, training=False)
                obs, reward, terminated, truncated, info = demo_env.step(action)
                total_reward += reward
                
                # Add small delay to make it watchable
                time.sleep(0.1)
                
                if terminated or truncated:
                    break
            
            print(f"Demo {demo_ep+1}: Reward={total_reward:6.2f}, "
                  f"Completion={info.get('completion_rate', 0)*100:4.1f}%, Steps={step+1}")
        
        demo_env.close()
        
        # Evaluation
        print(f"\n🎯 REINFORCE Evaluation (10 test episodes)")
        print("-" * 40)
        
        eval_rewards = []
        eval_success = 0
        
        for test_ep in range(10):
            obs, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.act(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    if info.get('completion_rate', 0) >= 1.0:
                        eval_success += 1
                    break
            
            eval_rewards.append(total_reward)
            print(f"Test {test_ep+1:2d}: Reward={total_reward:6.2f}, "
                  f"Completion={info.get('completion_rate', 0)*100:4.1f}%")
        
        print(f"\n✅ REINFORCE Final Performance:")
        print(f"   Average Test Reward: {sum(eval_rewards)/len(eval_rewards):.2f}")
        print(f"   Test Success Rate: {(eval_success/10)*100:.1f}%")
        print(f"   Total Parameters: {total_params:,}")
        
        env.close()
        return agent, {
            'avg_reward': sum(eval_rewards)/len(eval_rewards),
            'success_rate': eval_success/10,
            'training_time': training_time,
            'episodes': episodes
        }
        
    except Exception as e:
        print(f"❌ Error in REINFORCE training: {str(e)}")
        return None, None

if __name__ == "__main__":
    agent, results = run_reinforce_training()
    
    if results:
        print(f"\n🏆 REINFORCE Algorithm Summary:")
        print(f"   Performance: {results['avg_reward']:.2f} average reward")
        print(f"   Success Rate: {results['success_rate']*100:.1f}%")
        print(f"   Training Duration: {results['training_time']:.1f}s")
        print(f"   Episodes Completed: {results['episodes']}")
        print("\nREINFORCE training completed successfully! 🎉")
    else:
        print("\n❌ REINFORCE training failed. Check error messages above.")
