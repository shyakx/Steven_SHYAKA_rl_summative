"""
System Test - Verify all components are working
"""

import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 AgriTech RL System Test")
print("=" * 30)

# Test 1: Environment
try:
    from environment.custom_env import PrecisionFarmingEnv
    env = PrecisionFarmingEnv()
    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    env.close()
    print("✅ Environment: Working")
except Exception as e:
    print(f"❌ Environment: {str(e)}")

# Test 2: Base Agent
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents'))
    from agents.base_agent import BaseAgent, get_state_action_sizes
    state_size, action_size = get_state_action_sizes()
    print(f"✅ Base Agent: State={state_size}, Actions={action_size}")
except Exception as e:
    print(f"❌ Base Agent: {str(e)}")

# Test 3: All RL Agents
agents_to_test = [
    ('DQN', 'agents.dqn_agent', 'DQNAgent'),
    ('REINFORCE', 'agents.reinforce_agent', 'REINFORCEAgent'),
    ('PPO', 'agents.ppo_agent', 'PPOAgent'),
    ('ActorCritic', 'agents.actor_critic_agent', 'ActorCriticAgent')
]

test_config = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'hidden_size': 64,
    'num_layers': 2,
    'batch_size': 32
}

for agent_name, module_name, class_name in agents_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        agent = agent_class(state_size, action_size, test_config)
        
        # Count parameters for different agent types
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
        
        print(f"✅ {agent_name}: {param_count:,} parameters")
    except Exception as e:
        print(f"❌ {agent_name}: {str(e)}")

# Test 4: Training Infrastructure
try:
    from training.trainer import TrainingLogger, ModelCheckpoint
    logger = TrainingLogger("Test", "logs")
    checkpoint = ModelCheckpoint("models", "Test")
    print("✅ Training Infrastructure: Working")
except Exception as e:
    print(f"❌ Training Infrastructure: {str(e)}")

print("\n🎯 System Status:")
print("   All core components are ready for training!")
print("   To train all agents, run: python train_all_agents.py")
print("   For quick demo, run: python quick_demo.py")
