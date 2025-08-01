"""
Model Validation Script - Verify saved models work correctly
"""

import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_saved_models():
    """Check if models are saved and can be loaded correctly."""
    
    print("🔍 Model Validation")
    print("=" * 25)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("❌ Models directory not found!")
        return False
    
    # Check for saved model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("ℹ️  No .pth model files found yet")
        print("   Run training scripts to generate models:")
        print("   - python quick_demo.py")
        print("   - python train_all_agents.py")
        return True
    
    print(f"Found {len(model_files)} model files:")
    for file in model_files:
        file_path = os.path.join(models_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  ✅ {file} ({file_size:.1f} KB)")
    
    # Test loading models
    print("\n🧪 Testing Model Loading:")
    
    try:
        from agents.dqn_agent import DQNAgent
        from agents.reinforce_agent import REINFORCEAgent
        from agents.ppo_agent import PPOAgent
        from agents.actor_critic_agent import ActorCriticAgent
        from environment.custom_env import PrecisionFarmingEnv
        
        # Create test environment
        env = PrecisionFarmingEnv()
        state_size = 230
        action_size = 6
        
        # Test each model type
        agent_classes = {
            'dqn': DQNAgent,
            'reinforce': REINFORCEAgent,
            'ppo': PPOAgent,
            'actorcritic': ActorCriticAgent
        }
        
        for agent_type, agent_class in agent_classes.items():
            # Look for model files containing this agent type
            matching_files = [f for f in model_files if agent_type in f.lower()]
            
            if matching_files:
                model_path = os.path.join(models_dir, matching_files[0])
                
                try:
                    # Create agent with correct constructor arguments
                    config = {'learning_rate': 0.001}
                    
                    if agent_type == 'actorcritic':
                        agent = agent_class(state_size, action_size, config, "Test")
                    else:
                        agent = agent_class(state_size, action_size, config)
                    
                    # Try to load the model
                    agent.load_model(model_path)
                    print(f"  ✅ {agent_type.upper()}: Model loaded successfully")
                    
                    # Test inference
                    obs, _ = env.reset()
                    action = agent.act(obs, training=False)
                    print(f"     Test action: {action} (valid: {0 <= action < action_size})")
                    
                except Exception as e:
                    print(f"  ❌ {agent_type.upper()}: Loading failed - {str(e)}")
            else:
                print(f"  ⏸️  {agent_type.upper()}: No model file found")
        
        env.close()
        print("\n✅ Model validation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        return False

def check_training_outputs():
    """Check all training output directories."""
    
    print("\n📁 Training Output Validation")
    print("=" * 35)
    
    directories = ["models", "logs", "analysis"]
    
    for dir_name in directories:
        if os.path.exists(dir_name):
            files = os.listdir(dir_name)
            print(f"✅ {dir_name}/: {len(files)} files")
            
            # Show some example files
            if files:
                for file in files[:3]:  # Show first 3 files
                    file_path = os.path.join(dir_name, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path) / 1024
                        print(f"   - {file} ({size:.1f} KB)")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more files")
        else:
            print(f"⏸️  {dir_name}/: Directory not created yet")
    
    print()

def main():
    """Run complete model validation."""
    
    print("🤖 AgriTech RL - Model & Output Validation")
    print("=" * 45)
    
    # Check training outputs
    check_training_outputs()
    
    # Validate models
    validate_saved_models()
    
    print("\n💡 Tips:")
    print("   - Run 'python quick_demo.py' for fast training")
    print("   - Run 'python train_all_agents.py' for comprehensive training")
    print("   - Check 'analysis/' for performance reports")
    print("   - Models in 'models/' can be loaded for inference")

if __name__ == "__main__":
    main()
