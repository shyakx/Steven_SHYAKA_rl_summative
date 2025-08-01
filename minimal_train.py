"""
Minimal Training Test - Quick DQN Demo
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def quick_train():
    print("🚀 Starting Quick DQN Training")
    print("=" * 35)
    
    try:
        # Import components
        from environment.custom_env import PrecisionFarmingEnv
        from agents.dqn_agent import DQNAgent
        from training.trainer import TrainingLogger
        
        # Create environment
        env = PrecisionFarmingEnv()
        obs, _ = env.reset()
        
        # Get dimensions
        from agents.base_agent import BaseAgent
        dummy = BaseAgent(0, 0, {}, "Test")
        state_size = len(dummy.preprocess_state(obs))
        action_size = env.action_space.n
        
        print(f"📊 Environment: {state_size} state dims, {action_size} actions")
        
        # Create agent
        config = {
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 0.99,
            'memory_size': 1000,
            'batch_size': 32,
            'target_update_freq': 50,
            'gamma': 0.99,
            'hidden_size': 64,
            'num_layers': 2
        }
        
        agent = DQNAgent(state_size, action_size, config)
        logger = TrainingLogger("QuickDQN", "logs")
        
        print(f"🤖 Agent created with {sum(p.numel() for p in agent.q_network.parameters())} parameters")
        
        # Quick training loop
        max_episodes = 50  # Very short for quick demo
        
        for episode in range(max_episodes):
            obs, info = env.reset()
            total_reward = 0
            
            for step in range(env.MAX_STEPS):
                action = agent.act(obs, training=True)
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated
                
                agent.remember(obs, action, reward, next_obs, done)
                total_reward += reward
                
                if done:
                    break
                
                obs = next_obs
                info = next_info
            
            # Learn from experience
            loss = agent.learn()
            
            # Calculate success
            success_rate = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
            
            # Log and print progress
            logger.log_episode(episode, total_reward, step + 1, success_rate, loss)
            
            if episode % 10 == 0:
                print(f"Episode {episode:2d}: Reward={total_reward:6.2f}, "
                      f"Steps={step+1:3d}, Success={success_rate:.0%}, "
                      f"Loss={loss:.4f}, ε={agent.epsilon:.3f}")
        
        print(f"\n🎉 Quick training completed!")
        print(f"📊 Final episode: Reward={total_reward:.2f}, Success={success_rate:.0%}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        agent.save_model("models/quick_dqn.pth")
        print(f"💾 Model saved to models/quick_dqn.pth")
        
        env.close()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_train()
