"""
Quick Agent Validation - Test all 4 agents can run
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def validate_agents():
    print("🧪 Quick Agent Validation")
    print("=" * 30)
    
    try:
        # Import everything
        from environment.custom_env import PrecisionFarmingEnv
        from agents.dqn_agent import DQNAgent
        from agents.reinforce_agent import REINFORCEAgent
        from agents.ppo_agent import PPOAgent
        from agents.actor_critic_agent import ActorCriticAgent
        from agents.base_agent import BaseAgent
        
        # Setup
        env = PrecisionFarmingEnv()
        obs, _ = env.reset()
        dummy = BaseAgent(0, 0, {}, "Test")
        state_size = len(dummy.preprocess_state(obs))
        action_size = env.action_space.n
        
        config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'hidden_size': 32,
            'num_layers': 1,
            'batch_size': 16,
            'memory_size': 100,
            'buffer_size': 100
        }
        
        agents = {
            'DQN': DQNAgent(state_size, action_size, config),
            'REINFORCE': REINFORCEAgent(state_size, action_size, config),
            'PPO': PPOAgent(state_size, action_size, config),
            'ActorCritic': ActorCriticAgent(state_size, action_size, config)
        }
        
        print(f"📊 Environment: {state_size} states, {action_size} actions")
        print()
        
        # Test each agent
        for agent_name, agent in agents.items():
            print(f"🤖 Testing {agent_name}...")
            
            obs, _ = env.reset()
            total_reward = 0
            
            # Run a few steps
            for step in range(5):
                if agent_name == 'PPO':
                    action, log_prob, value, state_array = agent.act(obs, training=True)
                else:
                    action = agent.act(obs, training=True)
                
                next_obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                
                # Store experience based on agent type
                if agent_name == 'DQN':
                    agent.remember(obs, action, reward, next_obs, terminated or truncated)
                elif agent_name == 'PPO':
                    agent.store_transition(state_array, action, reward, log_prob, value, terminated or truncated)
                else:  # REINFORCE, ActorCritic
                    agent.remember_reward(reward)
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Test learning
            if agent_name == 'PPO':
                agent.finish_trajectory(last_value=0)
            
            loss = agent.learn()
            
            print(f"  ✅ {agent_name}: {step+1} steps, reward={total_reward:.2f}, loss={loss:.4f}")
        
        env.close()
        print()
        print("🎉 All agents validated successfully!")
        print("Ready to start full training!")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_agents()
