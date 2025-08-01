"""
Single Agent Tester - Test one agent at a time
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dqn():
    """Test DQN agent specifically."""
    print("🤖 Testing DQN Agent")
    print("=" * 25)
    
    from environment.custom_env import PrecisionFarmingEnv
    from agents.dqn_agent import DQNAgent
    from agents.base_agent import BaseAgent
    
    # Setup
    env = PrecisionFarmingEnv()
    obs, _ = env.reset()
    dummy = BaseAgent(0, 0, {}, "Test")
    state_size = len(dummy.preprocess_state(obs))
    action_size = env.action_space.n
    
    config = {
        'learning_rate': 0.001,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.98,
        'memory_size': 1000,
        'batch_size': 32,
        'target_update_freq': 25,
        'gamma': 0.99,
        'hidden_size': 128,
        'num_layers': 2
    }
    
    agent = DQNAgent(state_size, action_size, config)
    print(f"📊 {state_size} states, {action_size} actions, {sum(p.numel() for p in agent.q_network.parameters()):,} params")
    
    # Quick training
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(80):  # Moderate training length
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
        
        loss = agent.learn()
        episode_rewards.append(total_reward)
        success = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
        
        if episode % 10 == 0 or episode == 79:
            avg_10 = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
            print(f"Ep {episode:2d}: Reward={total_reward:6.2f}, Avg10={avg_10:6.2f}, "
                  f"Steps={step+1:3d}, Success={success:.0%}, Loss={loss:.4f}, ε={agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\n🧪 Final Evaluation...")
    eval_rewards = []
    eval_successes = []
    eval_steps = []
    
    for _ in range(10):
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(env.MAX_STEPS):
            action = agent.act(obs, training=False)  # No exploration
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        eval_rewards.append(total_reward)
        eval_successes.append(1.0 if info.get('current_diseased_count', 1) == 0 else 0.0)
        eval_steps.append(step + 1)
    
    avg_reward = sum(eval_rewards) / len(eval_rewards)
    avg_success = sum(eval_successes) / len(eval_successes)
    avg_steps = sum(eval_steps) / len(eval_steps)
    
    print(f"\n📊 DQN Results:")
    print(f"   Training Time: {training_time:.1f}s")
    print(f"   Final Avg Reward: {avg_reward:.2f}")
    print(f"   Success Rate: {avg_success:.1%}")
    print(f"   Avg Steps: {avg_steps:.1f}")
    print(f"   Efficiency: {avg_reward/avg_steps:.3f} reward/step")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    agent.save_model("models/test_dqn.pth")
    
    env.close()
    return {
        'agent': 'DQN',
        'reward': avg_reward,
        'success': avg_success,
        'steps': avg_steps,
        'time': training_time,
        'efficiency': avg_reward/avg_steps
    }

def test_reinforce():
    """Test REINFORCE agent specifically."""
    print("\n🤖 Testing REINFORCE Agent")
    print("=" * 30)
    
    from environment.custom_env import PrecisionFarmingEnv
    from agents.reinforce_agent import REINFORCEAgent
    from agents.base_agent import BaseAgent
    
    # Setup
    env = PrecisionFarmingEnv()
    obs, _ = env.reset()
    dummy = BaseAgent(0, 0, {}, "Test")
    state_size = len(dummy.preprocess_state(obs))
    action_size = env.action_space.n
    
    config = {
        'learning_rate': 0.002,
        'baseline_learning_rate': 0.002,
        'gamma': 0.99,
        'hidden_size': 128,
        'num_layers': 2
    }
    
    agent = REINFORCEAgent(state_size, action_size, config)
    policy_params = sum(p.numel() for p in agent.policy_network.parameters())
    baseline_params = sum(p.numel() for p in agent.baseline_network.parameters())
    print(f"📊 {state_size} states, {action_size} actions, {policy_params + baseline_params:,} params")
    
    # Training
    episode_rewards = []
    start_time = time.time()
    
    for episode in range(80):
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(env.MAX_STEPS):
            action = agent.act(obs, training=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            agent.remember_reward(reward)
            total_reward += reward
            
            if terminated or truncated:
                break
            
            obs = next_obs
            info = next_info
        
        loss = agent.learn()
        episode_rewards.append(total_reward)
        success = 1.0 if info.get('current_diseased_count', 1) == 0 else 0.0
        
        if episode % 10 == 0 or episode == 79:
            avg_10 = sum(episode_rewards[-10:]) / min(len(episode_rewards), 10)
            print(f"Ep {episode:2d}: Reward={total_reward:6.2f}, Avg10={avg_10:6.2f}, "
                  f"Steps={step+1:3d}, Success={success:.0%}, Loss={loss:.4f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\n🧪 Final Evaluation...")
    eval_rewards = []
    eval_successes = []
    eval_steps = []
    
    for _ in range(10):
        obs, info = env.reset()
        total_reward = 0
        
        for step in range(env.MAX_STEPS):
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        eval_rewards.append(total_reward)
        eval_successes.append(1.0 if info.get('current_diseased_count', 1) == 0 else 0.0)
        eval_steps.append(step + 1)
    
    avg_reward = sum(eval_rewards) / len(eval_rewards)
    avg_success = sum(eval_successes) / len(eval_successes)
    avg_steps = sum(eval_steps) / len(eval_steps)
    
    print(f"\n📊 REINFORCE Results:")
    print(f"   Training Time: {training_time:.1f}s")
    print(f"   Final Avg Reward: {avg_reward:.2f}")
    print(f"   Success Rate: {avg_success:.1%}")
    print(f"   Avg Steps: {avg_steps:.1f}")
    print(f"   Efficiency: {avg_reward/avg_steps:.3f} reward/step")
    
    # Save model
    agent.save_model("models/test_reinforce.pth")
    
    env.close()
    return {
        'agent': 'REINFORCE',
        'reward': avg_reward,
        'success': avg_success,
        'steps': avg_steps,
        'time': training_time,
        'efficiency': avg_reward/avg_steps
    }

def compare_results(results):
    """Compare results from multiple agents."""
    print(f"\n🏆 COMPARISON RESULTS")
    print("=" * 40)
    print("| Agent     | Reward | Success | Steps | Efficiency |")
    print("|-----------|--------|---------|-------|------------|")
    
    for result in results:
        print(f"| {result['agent']:9} | {result['reward']:6.2f} | "
              f"{result['success']:6.1%} | {result['steps']:5.1f} | "
              f"{result['efficiency']:9.3f} |")
    
    # Find winners
    best_reward = max(results, key=lambda x: x['reward'])
    best_success = max(results, key=lambda x: x['success'])
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    
    print(f"\n🥇 Winners:")
    print(f"   🎯 Best Reward: {best_reward['agent']} ({best_reward['reward']:.2f})")
    print(f"   ✅ Best Success: {best_success['agent']} ({best_success['success']:.1%})")
    print(f"   ⚡ Most Efficient: {best_efficiency['agent']} ({best_efficiency['efficiency']:.3f})")

if __name__ == "__main__":
    print("🌾 AgriTech Agent Testing")
    print("=" * 30)
    
    results = []
    
    # Test DQN
    try:
        result_dqn = test_dqn()
        results.append(result_dqn)
    except Exception as e:
        print(f"❌ DQN failed: {e}")
    
    # Test REINFORCE
    try:
        result_reinforce = test_reinforce()
        results.append(result_reinforce)
    except Exception as e:
        print(f"❌ REINFORCE failed: {e}")
    
    # Compare results if we have any
    if results:
        compare_results(results)
        
        print(f"\n💾 Models saved to models/ directory")
        print(f"🚀 Run individual tests with:")
        print(f"   python -c \"from single_agent_test import test_dqn; test_dqn()\"")
        print(f"   python -c \"from single_agent_test import test_reinforce; test_reinforce()\"")
    else:
        print("❌ No agents completed successfully!")
