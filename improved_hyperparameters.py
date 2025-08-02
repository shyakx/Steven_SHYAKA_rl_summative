"""
Improved Hyperparameter Configurations for Better Performance
Based on analysis of current training results
"""

# Improved DQN Configuration (already performing well, minor tweaks)
DQN_IMPROVED_CONFIG = {
    'learning_rate': 0.0005,  # Slightly lower for stability
    'gamma': 0.995,           # Higher discount for long-term planning
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,      # Higher minimum exploration
    'epsilon_decay': 0.9995,  # Slower decay
    'batch_size': 64,         # Larger batch
    'target_update_frequency': 50,  # More frequent updates
    'memory_size': 20000,     # Larger replay buffer
    'hidden_size': 256,       # Larger network
    'num_layers': 3
}

# Improved REINFORCE Configuration (needs major fixes)
REINFORCE_IMPROVED_CONFIG = {
    'learning_rate': 0.0003,     # Lower learning rate
    'baseline_learning_rate': 0.0005,  # Higher baseline LR
    'gamma': 0.99,
    'hidden_size': 256,          # Larger network
    'num_layers': 3,
    'entropy_coeff': 0.01,       # Add entropy for exploration
    'grad_clip': 1.0             # Gradient clipping
}

# Improved PPO Configuration (good speed, needs better performance)
PPO_IMPROVED_CONFIG = {
    'learning_rate': 0.0001,     # Lower learning rate
    'gamma': 0.995,              # Higher discount
    'epsilon': 0.15,             # Tighter clipping
    'value_loss_coeff': 1.0,     # Higher value loss weight
    'entropy_coeff': 0.02,       # More exploration
    'hidden_size': 512,          # Much larger network
    'num_layers': 3,
    'ppo_epochs': 6,             # More update epochs
    'batch_size': 128,           # Larger batch
    'buffer_size': 4096,         # Larger buffer
    'max_grad_norm': 0.5
}

# Improved Actor-Critic Configuration (needs faster convergence)
ACTOR_CRITIC_IMPROVED_CONFIG = {
    'actor_lr': 0.0003,          # Lower actor LR
    'critic_lr': 0.001,          # Higher critic LR
    'gamma': 0.995,              # Higher discount
    'hidden_size': 256,          # Larger network
    'num_layers': 3,
    'entropy_coeff': 0.02,       # More exploration
    'value_loss_coeff': 1.0,     # Higher value loss
    'grad_clip': 1.0             # Gradient clipping
}

# Training improvements
TRAINING_IMPROVEMENTS = {
    'episodes': 200,             # More episodes
    'max_steps': 250,            # More steps per episode
    'early_stopping': True,      # Stop if performance degrades
    'learning_rate_decay': True, # Decay LR over time
    'reward_shaping': True       # Better reward design
}

print("🔧 Improved Hyperparameter Configurations")
print("=" * 50)
print("Key improvements:")
print("• Lower learning rates for stability")
print("• Larger networks for better capacity") 
print("• Higher discount factors for long-term planning")
print("• Better exploration strategies")
print("• Gradient clipping to prevent exploding gradients")
print("• Larger buffers and batch sizes")
