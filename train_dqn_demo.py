"""
Quick DQN Training Demo for AgriTech Precision Farming

Train a DQN agent for a short session to validate the complete pipeline.
"""

import os
import sys

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from agents.dqn_agent import train_dqn_agent


def main():
    """Run a quick DQN training session."""
    
    print("🌾 AgriTech Precision Farming - DQN Training Demo")
    print("=" * 55)
    print("Training an intelligent drone to treat diseased crops!")
    print()
    
    # Training configuration
    config = {
        # Environment
        'max_episodes': 200,  # Short training for demo
        'max_steps_per_episode': 200,
        
        # DQN hyperparameters
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        
        # Network architecture
        'hidden_size': 128,
        'num_layers': 2,
        
        # Training settings
        'memory_size': 10000,
        'target_update_frequency': 100,
        
        # Logging
        'log_interval': 10,
        'eval_interval': 25,
        'save_interval': 50,
        'log_dir': 'logs',
        'model_dir': 'models'
    }
    
    print(f"🔧 Training Configuration:")
    print(f"   • Episodes: {config['max_episodes']}")
    print(f"   • Learning Rate: {config['learning_rate']}")
    print(f"   • Network: {config['num_layers']} layers, {config['hidden_size']} hidden units")
    print(f"   • Memory Size: {config['memory_size']:,}")
    print()
    
    try:
        # Train the agent
        trained_agent = train_dqn_agent(config, save_dir="models")
        
        print("\n🎉 Training completed successfully!")
        print("\n📁 Generated files:")
        print("   • models/dqn_final.pth - Trained DQN model")
        print("   • logs/DQN_training.log - Training log")
        print("   • logs/dqn_training_plots.png - Learning curves")
        print("   • logs/dqn_metrics.json - Performance metrics")
        
        # Get final training info
        training_info = trained_agent.get_training_info()
        print(f"\n🤖 Final Agent Stats:")
        print(f"   • Exploration (ε): {training_info['epsilon']:.3f}")
        print(f"   • Memory Size: {training_info['memory_size']:,}")
        print(f"   • Training Steps: {training_info['training_steps']:,}")
        print(f"   • Target Updates: {training_info['target_updates']}")
        
        return trained_agent
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    agent = main()
    
    if agent:
        print(f"\n✅ DQN agent ready! You can now:")
        print(f"   1. Continue training with more episodes")
        print(f"   2. Evaluate the agent's performance")
        print(f"   3. Compare with other RL algorithms")
        print(f"   4. Deploy for real-world agricultural applications")
    
    print(f"\n🚀 Next steps: Implement REINFORCE, PPO, and Actor-Critic!")
