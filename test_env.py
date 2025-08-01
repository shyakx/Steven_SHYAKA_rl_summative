"""Test script for the AgriTech environment"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))

from environment.custom_env import PrecisionFarmingEnv

def test_environment():
    print("🌾 Testing AgriTech Precision Farming Environment")
    print("=" * 50)
    
    # Create environment
    env = PrecisionFarmingEnv()
    print("✅ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Environment reset - {info['diseased_remaining']} diseased crops to treat")
    
    # Test a few actions
    print("\n🎮 Testing actions:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = env.ACTIONS[action]
        print(f"  Step {i+1}: {action_name} -> Reward: {reward:.2f}, Battery: {info['battery_level']}%")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Environment test completed successfully!")
    print(f"Final completion rate: {info['completion_rate']:.1f}%")
    
    env.close()

if __name__ == "__main__":
    test_environment()
