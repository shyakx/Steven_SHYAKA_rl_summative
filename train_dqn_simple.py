"""
Simple Training Launcher - Start with DQN Agent
"""

import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🚀 AgriTech RL Training Launcher")
    print("=" * 40)
    
    # Simple DQN training configuration
    config = {
        'max_episodes': 200,
        'learning_rate': 0.001,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 5000,
        'batch_size': 32,
        'target_update_freq': 50,
        'eval_interval': 25,
        'save_interval': 50,
        'hidden_size': 128,
        'num_layers': 2,
        'gamma': 0.99,
        'log_dir': 'logs'
    }
    
    try:
        from agents.dqn_agent import train_dqn_agent
        
        print("🤖 Starting DQN training...")
        print(f"📊 Configuration: {config['max_episodes']} episodes")
        
        # Create directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Train DQN agent
        agent = train_dqn_agent(config, save_dir="models")
        
        print("🎉 DQN training completed!")
        print("📄 Check logs/ for training progress")
        print("💾 Check models/ for saved models")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
