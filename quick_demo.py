"""
Quick RL Demo - Train all 4 agents with reduced episodes for testing
"""

import os
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_all_agents import create_training_configs, train_all_agents, evaluate_all_agents, generate_comparison_report


def quick_demo():
    """Run a quick demo with reduced episodes."""
    print("🚀 Quick RL Demo - All 4 Agents")
    print("=" * 40)
    
    # Create reduced configs for quick demo
    configs = create_training_configs()
    
    # Reduce episodes for quick testing
    for agent_name in configs:
        configs[agent_name]['max_episodes'] = 100  # Much shorter for demo
        configs[agent_name]['eval_interval'] = 20
        configs[agent_name]['save_interval'] = 50
    
    print("📝 Training Configuration:")
    for agent_name, config in configs.items():
        print(f"  {agent_name}: {config['max_episodes']} episodes")
    
    try:
        # Train all agents
        agents, training_metrics = train_all_agents(configs)
        
        if agents:
            # Quick evaluation
            print(f"\n📊 Quick Evaluation (10 episodes each)")
            evaluation_results = evaluate_all_agents(agents, num_episodes=10)
            
            # Generate report
            report_path = generate_comparison_report(agents, training_metrics, evaluation_results)
            
            print(f"\n🎉 Quick Demo Completed!")
            print(f"📄 Report: {report_path}")
            
            # Print summary
            print(f"\n📋 Quick Results Summary:")
            for agent_name in ['DQN', 'REINFORCE', 'PPO', 'ActorCritic']:
                if agent_name in evaluation_results and 'error' not in evaluation_results[agent_name]:
                    results = evaluation_results[agent_name]
                    print(f"  {agent_name:12}: Reward={results['avg_reward']:6.2f}, "
                          f"Success={results['success_rate']:5.1%}, "
                          f"Steps={results['avg_steps']:5.1f}")
                else:
                    print(f"  {agent_name:12}: ❌ Failed to train/evaluate")
        
        else:
            print("❌ No agents were successfully trained!")
            
    except Exception as e:
        print(f"❌ Error in quick demo: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    quick_demo()
