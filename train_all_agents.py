"""
Comprehensive RL Training Script for AgriTech Precision Farming

This script trains all 4 RL agents (DQN, REINFORCE, PPO, Actor-Critic)
on the AgriTech environment and generates comparative analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from typing import Dict, List, Any

# Import all agents
from agents.dqn_agent import DQNAgent, train_dqn_agent
from agents.reinforce_agent import REINFORCEAgent, train_reinforce_agent
from agents.ppo_agent import PPOAgent, train_ppo_agent
from agents.actor_critic_agent import ActorCriticAgent, train_actor_critic_agent

# Import training infrastructure
from training.trainer import TrainingLogger, compare_agents, plot_agent_comparison
from environment.custom_env import PrecisionFarmingEnv


def create_training_configs() -> Dict[str, Dict[str, Any]]:
    """Create training configurations for all agents."""
    
    base_config = {
        'max_episodes': 500,  # Reduced for faster comparison
        'eval_interval': 25,
        'save_interval': 50,
        'log_dir': 'logs',
        'gamma': 0.99
    }
    
    configs = {
        'DQN': {
            **base_config,
            'learning_rate': 0.001,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'hidden_size': 128,
            'num_layers': 2
        },
        
        'REINFORCE': {
            **base_config,
            'learning_rate': 0.001,
            'baseline_learning_rate': 0.001,
            'hidden_size': 128,
            'num_layers': 2
        },
        
        'PPO': {
            **base_config,
            'learning_rate': 3e-4,
            'lam': 0.95,
            'clip_ratio': 0.2,
            'value_loss_coeff': 0.5,
            'entropy_coeff': 0.01,
            'hidden_size': 128,
            'num_layers': 2,
            'batch_size': 64,
            'n_epochs': 10,
            'buffer_size': 2048,
            'update_interval': 10
        },
        
        'ActorCritic': {
            **base_config,
            'learning_rate': 0.001,
            'value_loss_coeff': 0.5,
            'entropy_coeff': 0.01,
            'hidden_size': 128,
            'num_layers': 2
        }
    }
    
    return configs


def train_all_agents(configs: Dict[str, Dict[str, Any]], save_dir: str = "models") -> Dict[str, Any]:
    """
    Train all RL agents and return their performance metrics.
    
    Args:
        configs: Training configurations for each agent
        save_dir: Directory to save models
        
    Returns:
        Dictionary containing all trained agents and their metrics
    """
    print("🚀 Starting Comprehensive RL Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("analysis", exist_ok=True)
    
    agents = {}
    training_metrics = {}
    
    # Training functions mapping
    training_functions = {
        'DQN': train_dqn_agent,
        'REINFORCE': train_reinforce_agent,
        'PPO': train_ppo_agent,
        'ActorCritic': train_actor_critic_agent
    }
    
    # Train each agent
    for agent_name, config in configs.items():
        print(f"\n🤖 Training {agent_name} Agent")
        print("-" * 30)
        
        start_time = datetime.now()
        
        try:
            # Train the agent
            agent = training_functions[agent_name](config, save_dir)
            agents[agent_name] = agent
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Collect metrics
            training_metrics[agent_name] = {
                'training_time': training_time,
                'episodes_completed': agent.episode_count,
                'training_steps': agent.training_step,
                'final_model_path': os.path.join(save_dir, f"{agent_name.lower()}_final.pth")
            }
            
            print(f"✅ {agent_name} training completed in {training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ Error training {agent_name}: {str(e)}")
            training_metrics[agent_name] = {'error': str(e)}
    
    return agents, training_metrics


def evaluate_all_agents(agents: Dict[str, Any], num_episodes: int = 50) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all trained agents on the environment.
    
    Args:
        agents: Dictionary of trained agents
        num_episodes: Number of episodes for evaluation
        
    Returns:
        Dictionary containing evaluation metrics for each agent
    """
    print(f"\n📊 Evaluating All Agents ({num_episodes} episodes each)")
    print("-" * 50)
    
    from training.trainer import evaluate_agent
    
    env = PrecisionFarmingEnv()
    evaluation_results = {}
    
    for agent_name, agent in agents.items():
        print(f"Evaluating {agent_name}...")
        
        try:
            avg_reward, success_rate, avg_steps = evaluate_agent(env, agent, num_episodes)
            
            evaluation_results[agent_name] = {
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'efficiency_score': avg_reward / avg_steps if avg_steps > 0 else 0
            }
            
            print(f"  Reward: {avg_reward:.2f}, Success: {success_rate:.1%}, "
                  f"Steps: {avg_steps:.1f}, Efficiency: {evaluation_results[agent_name]['efficiency_score']:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {agent_name}: {str(e)}")
            evaluation_results[agent_name] = {'error': str(e)}
    
    env.close()
    return evaluation_results


def generate_comparison_report(agents: Dict[str, Any], 
                             training_metrics: Dict[str, Dict],
                             evaluation_results: Dict[str, Dict[str, float]],
                             save_path: str = "analysis") -> str:
    """
    Generate a comprehensive comparison report.
    
    Args:
        agents: Dictionary of trained agents
        training_metrics: Training metrics for each agent
        evaluation_results: Evaluation results for each agent
        save_path: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    print("\n📝 Generating Comparison Report")
    print("-" * 35)
    
    os.makedirs(save_path, exist_ok=True)
    
    # Create comparison plots
    try:
        plot_path = os.path.join(save_path, "agent_comparison.png")
        fig = plot_agent_comparison(list(agents.keys()), evaluation_results, plot_path)
        print(f"📊 Comparison plots saved to {plot_path}")
    except Exception as e:
        print(f"❌ Error creating plots: {str(e)}")
        plot_path = None
    
    # Generate text report
    report_path = os.path.join(save_path, "rl_comparison_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# AgriTech Precision Farming - RL Agents Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Environment Overview\n")
        f.write("- **Environment**: AgriTech Precision Farming Drone\n")
        f.write("- **Task**: Navigate 15x15 farm grid, treat diseased crops, manage resources\n")
        f.write("- **State Space**: 230-dimensional (grid + position + resources)\n")
        f.write("- **Action Space**: 6 discrete actions (move + treat)\n")
        f.write("- **Success Metric**: Complete treatment of all diseased crops\n\n")
        
        f.write("## Training Summary\n")
        f.write("| Agent | Episodes | Training Time (s) | Parameters | Status |\n")
        f.write("|-------|----------|------------------|------------|--------|\n")
        
        for agent_name in ['DQN', 'REINFORCE', 'PPO', 'ActorCritic']:
            if agent_name in agents:
                params = sum(p.numel() for p in agents[agent_name].network.parameters() if hasattr(agents[agent_name], 'network'))
                if not params and hasattr(agents[agent_name], 'q_network'):
                    params = sum(p.numel() for p in agents[agent_name].q_network.parameters())
                elif not params and hasattr(agents[agent_name], 'policy_network'):
                    params = sum(p.numel() for p in agents[agent_name].policy_network.parameters())
                    if hasattr(agents[agent_name], 'baseline_network'):
                        params += sum(p.numel() for p in agents[agent_name].baseline_network.parameters())
                elif not params and hasattr(agents[agent_name], 'actor'):
                    params = sum(p.numel() for p in agents[agent_name].actor.parameters())
                    params += sum(p.numel() for p in agents[agent_name].critic.parameters())
                
                metrics = training_metrics.get(agent_name, {})
                episodes = metrics.get('episodes_completed', 'N/A')
                time_taken = metrics.get('training_time', 'N/A')
                if isinstance(time_taken, float):
                    time_taken = f"{time_taken:.1f}"
                status = "✅ Completed" if 'error' not in metrics else f"❌ {metrics['error']}"
                
                f.write(f"| {agent_name} | {episodes} | {time_taken} | {params:,} | {status} |\n")
            else:
                f.write(f"| {agent_name} | N/A | N/A | N/A | ❌ Not trained |\n")
        
        f.write("\n## Performance Results\n")
        f.write("| Agent | Avg Reward | Success Rate | Avg Steps | Efficiency |\n")
        f.write("|-------|------------|--------------|-----------|------------|\n")
        
        for agent_name in ['DQN', 'REINFORCE', 'PPO', 'ActorCritic']:
            if agent_name in evaluation_results and 'error' not in evaluation_results[agent_name]:
                results = evaluation_results[agent_name]
                f.write(f"| {agent_name} | {results['avg_reward']:.2f} | {results['success_rate']:.1%} | "
                       f"{results['avg_steps']:.1f} | {results['efficiency_score']:.4f} |\n")
            else:
                f.write(f"| {agent_name} | N/A | N/A | N/A | N/A |\n")
        
        f.write("\n## Algorithm Analysis\n\n")
        
        algorithm_descriptions = {
            'DQN': "**Deep Q-Network**: Value-based method using experience replay and target network for stability.",
            'REINFORCE': "**REINFORCE**: Policy gradient method with baseline for variance reduction.",
            'PPO': "**Proximal Policy Optimization**: Advanced policy gradient with clipped objective and GAE.",
            'ActorCritic': "**Actor-Critic**: Classic approach with separate policy and value function networks."
        }
        
        for agent_name, description in algorithm_descriptions.items():
            f.write(f"### {agent_name}\n")
            f.write(f"{description}\n\n")
            
            if agent_name in evaluation_results and 'error' not in evaluation_results[agent_name]:
                results = evaluation_results[agent_name]
                f.write(f"- **Performance**: {results['avg_reward']:.2f} avg reward, {results['success_rate']:.1%} success rate\n")
                f.write(f"- **Efficiency**: {results['efficiency_score']:.4f} reward per step\n")
            
            if agent_name in training_metrics and 'error' not in training_metrics[agent_name]:
                metrics = training_metrics[agent_name]
                f.write(f"- **Training**: {metrics.get('episodes_completed', 'N/A')} episodes completed\n")
            
            f.write("\n")
        
        f.write("## Key Findings\n\n")
        
        if evaluation_results:
            # Find best performing agent
            valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}
            if valid_results:
                best_reward = max(valid_results.items(), key=lambda x: x[1]['avg_reward'])
                best_success = max(valid_results.items(), key=lambda x: x[1]['success_rate'])
                best_efficiency = max(valid_results.items(), key=lambda x: x[1]['efficiency_score'])
                
                f.write(f"- **Highest Reward**: {best_reward[0]} ({best_reward[1]['avg_reward']:.2f})\n")
                f.write(f"- **Best Success Rate**: {best_success[0]} ({best_success[1]['success_rate']:.1%})\n")
                f.write(f"- **Most Efficient**: {best_efficiency[0]} ({best_efficiency[1]['efficiency_score']:.4f})\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the experimental results:\n\n")
        f.write("1. **For Production Deployment**: Choose the agent with the highest success rate and stability\n")
        f.write("2. **For Further Research**: Focus on the most efficient algorithm for resource optimization\n")
        f.write("3. **For Real-world Applications**: Consider ensemble methods combining multiple algorithms\n\n")
        
        if plot_path:
            f.write(f"## Visualizations\n\n")
            f.write(f"Detailed performance comparison charts are available in: `{os.path.basename(plot_path)}`\n\n")
        
        f.write("---\n")
        f.write("*Report generated by AgriTech RL Training Pipeline*\n")
    
    print(f"📄 Report saved to {report_path}")
    
    # Save metrics as JSON for further analysis
    metrics_path = os.path.join(save_path, "all_metrics.json")
    all_metrics = {
        'training_metrics': training_metrics,
        'evaluation_results': evaluation_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"📊 Metrics saved to {metrics_path}")
    
    return report_path


def main():
    """Main training and evaluation pipeline."""
    print("🌾 AgriTech Precision Farming - RL Agents Training Pipeline")
    print("=" * 60)
    
    # Create training configurations
    configs = create_training_configs()
    
    # Train all agents
    agents, training_metrics = train_all_agents(configs)
    
    # Evaluate agents
    if agents:
        evaluation_results = evaluate_all_agents(agents, num_episodes=50)
        
        # Generate comprehensive report
        report_path = generate_comparison_report(agents, training_metrics, evaluation_results)
        
        print(f"\n🎉 Training Pipeline Completed!")
        print(f"📄 Full report available at: {report_path}")
        print(f"📊 Models saved in: models/")
        print(f"📈 Logs available in: logs/")
        print(f"📋 Analysis in: analysis/")
        
    else:
        print("\n❌ No agents were successfully trained!")


if __name__ == "__main__":
    main()
