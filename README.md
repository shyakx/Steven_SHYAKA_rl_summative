# 🌾 AgriTech Precision Farming - RL Research Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A comprehensive reinforcement learning research platform for agricultural AI applications**

## 📖 Project Overview

This project implements a **complete RL research environment** for training autonomous farming drones to efficiently treat diseased crops while managing limited resources. The platform features **4 different RL algorithms** with comprehensive comparison and analysis tools.

### 🎯 Mission Statement
Using AI to build impactful solutions in agricultural sectors through precision farming and resource optimization.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Test Installation
```bash
python system_test.py
```

### Run Demo
```bash
python demo.py
```

### Train All Models
```bash
python train_all_agents.py
```

---

## 🌾 Environment: Precision Farming Drone

### Overview
An autonomous drone navigates a **15×15 farm grid** to locate and treat diseased crops while efficiently managing battery and treatment resources.

### 🎮 Action Space (6 Discrete Actions)
| Action | Battery Cost | Description |
|--------|--------------|-------------|
| **Move Up/Down/Left/Right** | 2 units | Navigate in cardinal directions |
| **Treat Crop** | 5 units | Apply treatment to diseased crop |
| **Charge Battery** | 0 units | Recharge at charging station |

### 📊 Observation Space
- **Grid State**: 15×15 array showing all cell types (225 features)
- **Agent Position**: (x, y) coordinates (2 features)  
- **Resources**: [battery_level, treatment_capacity, steps_remaining] (3 features)
- **Total Dimensionality**: 230 features

### 🌱 Farm Cell Types
| Cell Type | Color | Description |
|-----------|-------|-------------|
| 🟤 Empty Soil | Brown | Traversable empty space |
| 🟢 Healthy Crops | Green | Crops in good condition |
| 🔴 Diseased Crops | Red | **Primary targets** for treatment |
| 🟫 Obstacles | Gray | Rocks/trees to avoid |
| 🔵 Charging Stations | Blue | Battery recharge points |
| 🟡 Treated Crops | Light Green | Successfully treated crops |

### 🏆 Mission Objectives
- **Primary Goal**: Treat all diseased crops before battery depletion
- **Resource Management**: Efficiently use battery (100 initial) and treatment capacity (20 initial)
- **Time Constraint**: Complete mission within 200 steps
- **Success Metric**: 100% crop treatment rate

---

## 🤖 RL Algorithms Implemented

| Algorithm | Type | Parameters | Key Features |
|-----------|------|------------|--------------|
| **DQN** | Value-based | 19,334 | Experience replay, target network, ε-greedy |
| **REINFORCE** | Policy gradient | 38,343 | Monte Carlo updates, baseline for variance reduction |
| **PPO** | Advanced policy | 38,343 | Clipped objective, GAE, actor-critic structure |
| **Actor-Critic** | Classic AC | 19,399 | Separate policy and value networks |

### Algorithm Details

#### 🔷 DQN (Deep Q-Network)
- **Neural Network**: 2-layer MLP with 128 hidden units
- **Experience Replay**: 10,000 memory buffer
- **Target Network**: Updated every 100 steps
- **Exploration**: ε-greedy with decay (1.0 → 0.01)

#### 🔶 REINFORCE
- **Policy Network**: 2-layer MLP with 128 hidden units
- **Baseline Network**: Separate value function for variance reduction
- **Updates**: Monte Carlo policy gradient
- **Optimization**: Adam optimizer with learning rate 0.001

#### 🔴 PPO (Proximal Policy Optimization)
- **Actor-Critic Architecture**: Shared backbone with separate heads
- **Clipped Objective**: Prevents large policy updates
- **GAE**: Generalized Advantage Estimation (λ=0.95)
- **Mini-batch Updates**: 64 batch size, 10 epochs per update

#### 🔵 Actor-Critic
- **Dual Networks**: Separate actor (policy) and critic (value)
- **Advantage**: TD(0) advantage estimation
- **Entropy Regularization**: Encourages exploration
- **Synchronous Updates**: Policy and value updated simultaneously

---

## 📁 Project Structure

```
d:\trimester3_ml_summative\
├── 🌾 environment/
│   ├── custom_env.py          # Main RL environment
│   ├── rendering.py           # Pygame visualization
│   └── __pycache__/
├── 🤖 agents/
│   ├── base_agent.py          # Abstract base class
│   ├── dqn_agent.py           # DQN implementation
│   ├── reinforce_agent.py     # REINFORCE implementation
│   ├── ppo_agent.py           # PPO implementation
│   ├── actor_critic_agent.py  # Actor-Critic implementation
│   └── __pycache__/
├── 🏋️ training/
│   └── trainer.py             # Training infrastructure
├── 📊 models/                 # Saved model files (.pth)
├── 📈 logs/                   # Training logs and metrics
├── 📋 analysis/               # Comparison reports and plots
├── 🎮 demo.py                 # Interactive visualization
├── 🧪 system_test.py          # System validation
├── ⚡ quick_demo.py           # Fast multi-agent training
├── 🏆 train_all_agents.py     # Complete training pipeline
└── 📚 SCRIPTS_REFERENCE.md    # Complete scripts guide
```

---

## 🎯 Usage Guide

### 🔧 System Verification
```bash
# Test all components
python system_test.py

# Test environment only
python test_env.py

# Validate all agents
python validate_agents.py
```

### 🎮 Interactive Demo
```bash
# Visual demonstration with pygame
python demo.py
```

### 🚀 Training Options

#### Quick Training (Recommended for Testing)
```bash
# Train all 4 agents with 100 episodes each (~10 minutes)
python quick_demo.py
```

#### Individual Agent Training
```bash
# Train DQN only
python train_dqn_simple.py

# Minimal training test
python minimal_train.py
```

#### Complete Research Training
```bash
# Train all agents with 500 episodes each (~60 minutes)
python train_all_agents.py
```

#### Sequential Comparison
```bash
# Train agents one by one with detailed analysis
python sequential_comparison.py
```

### 📊 Results Analysis

After training, check these locations for results:

- **`analysis/rl_comparison_report.md`**: Comprehensive comparison report
- **`analysis/agent_comparison.png`**: Performance visualization charts
- **`analysis/all_metrics.json`**: Raw metrics data
- **`models/`**: Trained model files (.pth format)
- **`logs/`**: Individual agent training logs

---

## 📈 Expected Results

### Performance Metrics
The system tracks and compares:
- **Average Reward**: Cumulative reward per episode
- **Success Rate**: Percentage of missions completed successfully
- **Efficiency**: Reward per step taken
- **Training Time**: Time to convergence
- **Stability**: Variance in performance

### Typical Performance Ranges
| Algorithm | Avg Reward | Success Rate | Training Episodes |
|-----------|------------|--------------|-------------------|
| **DQN** | 50-80 | 60-80% | 300-500 |
| **REINFORCE** | 40-70 | 50-70% | 400-500 |
| **PPO** | 60-90 | 70-90% | 200-400 |
| **Actor-Critic** | 45-75 | 55-75% | 350-500 |

*Results may vary based on random initialization and environment complexity*

---

## 🛠️ Advanced Usage

### Custom Training Configuration
```python
# Modify training parameters in train_all_agents.py
configs = {
    'DQN': {
        'max_episodes': 1000,        # Increase for longer training
        'learning_rate': 0.0005,     # Adjust learning rate
        'epsilon_decay': 0.999,      # Slower exploration decay
        'memory_size': 20000,        # Larger replay buffer
    }
}
```

### Model Loading and Evaluation
```python
from agents.dqn_agent import DQNAgent
from environment.custom_env import PrecisionFarmingEnv

# Load trained model
agent = DQNAgent(state_size=230, action_size=6, config={})
agent.load_model('models/dqn_final.pth')

# Evaluate performance
env = PrecisionFarmingEnv()
obs, _ = env.reset()
total_reward = 0

for step in range(200):
    action = agent.act(obs, training=False)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Total Reward: {total_reward}")
```

### Custom Environment Modifications
```python
# Modify environment parameters in environment/custom_env.py
class PrecisionFarmingEnv:
    def __init__(self):
        self.grid_size = 20          # Larger farm
        self.initial_battery = 150   # More battery
        self.max_steps = 300         # Longer episodes
        self.num_diseased = 15       # More crops to treat
```

---

## 🔬 Research Extensions

### Potential Improvements
1. **Multi-Agent Systems**: Multiple drones working collaboratively
2. **Dynamic Environment**: Weather effects, crop growth over time
3. **Hierarchical RL**: High-level planning with low-level control
4. **Transfer Learning**: Pre-trained models for different farm layouts
5. **Real-world Integration**: Simulation-to-real transfer

### Academic Applications
- **Algorithm Comparison**: Systematic RL algorithm evaluation
- **Agricultural AI**: Real-world farming optimization research  
- **Resource Management**: Constrained optimization in RL
- **Multi-objective RL**: Balance between speed and efficiency
- **Curriculum Learning**: Progressive difficulty increase

---

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Verify Python path
python -c "import torch, numpy, matplotlib, pygame; print('All imports successful')"
```

#### CUDA/GPU Issues
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force CPU usage if needed (modify agent files)
device = torch.device('cpu')
```

#### Training Failures
```bash
# Check system test first
python system_test.py

# Start with minimal training
python minimal_train.py

# Verify individual agents
python validate_agents.py
```

#### Performance Issues
- **Reduce episode count** for faster testing
- **Use CPU training** if GPU memory insufficient
- **Monitor memory usage** during PPO training (highest memory requirement)

---

## 📋 Requirements

### Software Dependencies
```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
pygame>=2.1.0
gymnasium>=0.26.0
```

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for PPO)
- **Storage**: 1GB free space for models and logs
- **OS**: Windows 10/11, macOS, Linux

### Optional
- **CUDA**: For GPU acceleration (recommended for faster training)
- **Tensorboard**: For advanced training visualization

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone <repository-url>
cd trimester3_ml_summative
pip install -r requirements.txt
python system_test.py
```

---

## 📞 Support

For questions or issues:
1. Run `python system_test.py` to verify installation
2. Check the `SCRIPTS_REFERENCE.md` for detailed script documentation
3. Review training logs in `logs/` directory for debugging

---

## 🎉 Acknowledgments

- Built for agricultural AI research and education
- Implements state-of-the-art RL algorithms
- Designed for academic and research applications

**Ready to revolutionize agriculture with AI!** 🌾🤖

---

*Last updated: August 1, 2025*

## 🏆 Reward System

**Positive Rewards:**
- **+25**: Treat diseased crop (main objective)
- **+2**: Charge battery at station (resource management)
- **+0.5**: Valid movement (exploration)
- **+100**: Complete all diseased crops (mission success)
- **+0-50**: Efficiency bonus based on remaining resources

**Negative Rewards:**
- **-5**: Hit obstacle or waste treatment on healthy crop
- **-2**: Invalid actions (treat without capacity, charge away from station)
- **-50**: Battery depletion (mission failure)
- **-0.1**: Time penalty per step (encourages efficiency)

## 🚀 Quick Start

### 1. Setup
```powershell
# Install dependencies
pip install -r requirements.txt
```

### 2. Test Environment
```powershell
# Test basic functionality
python demo.py --mode test

# Expected output:
# ✅ Environment reset successful
# 🎮 Testing all actions...
# ✅ Environment test completed successfully!
```

### 3. Interactive Demo
```powershell
# Run visual demonstration with intelligent agent
python demo.py --mode demo

# Watch the drone navigate and treat crops!
# Controls: ESC to exit, close window to quit
```

### 4. Direct Environment Test
```powershell
# Test core environment
python environment/custom_env.py
```

## 🎬 Visual Features

The pygame-based renderer provides:
- **Real-time farm visualization** with color-coded cells
- **Animated drone** with pulsing effect and propeller rotation
- **Battery indicator** above drone (color changes with level)
- **Mission control panel** showing:
  - Step counter and time remaining
  - Battery and treatment levels
  - Progress tracking with completion percentage
  - Total score and agent position
  - Visual legend for all cell types
- **Mission status display** at top of screen
- **Smooth animations** at 30 FPS

## 🧠 Environment Design Rationale

### **Mission Alignment**
This scenario directly addresses real-world agricultural challenges:
- **Precision Agriculture**: Targeted treatment reduces chemical waste
- **Resource Optimization**: Battery and treatment management mirrors real constraints
- **Autonomous Navigation**: Obstacle avoidance and path planning
- **Efficiency Focus**: Time and resource penalties encourage optimal strategies

### **RL Challenge Characteristics**
- **Sparse Rewards**: Main rewards come from treating diseased crops
- **Resource Management**: Multi-objective optimization (treatment vs. battery)
- **Spatial Reasoning**: Navigation and planning in 2D grid
- **Exploration vs. Exploitation**: Balance between searching and treating
- **Temporal Constraints**: Time limit adds urgency

### **Scalability**
The environment can be extended with:
- Variable grid sizes
- Dynamic crop disease spread
- Weather conditions
- Multiple drone coordination
- Real agricultural data integration

## 📁 Project Structure

```
trimester3_ml_summative/
├── environment/
│   ├── custom_env.py       # Core farming environment
│   └── rendering.py        # Pygame visualization
├── demo.py                 # Interactive demonstration
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 🔬 Technical Specifications

- **Framework**: Gymnasium (OpenAI Gym successor)
- **Rendering**: Pygame with custom graphics
- **State Representation**: Mixed discrete/continuous
- **Episode Length**: Variable (ends on success/failure/timeout)
- **Determinism**: Stochastic farm layout generation
- **Performance**: ~30 FPS visualization, fast environment steps

## 🎓 Educational Value

This environment demonstrates:
- **Custom RL environment design** following Gymnasium standards
- **Real-world problem modeling** with agricultural applications
- **Resource management** in multi-objective settings
- **Visual feedback systems** for RL development
- **Mission-driven AI** aligned with agricultural sector needs

## 🌱 Future Applications

This foundation enables:
- **RL Algorithm Training**: DQN, PPO, Actor-Critic, etc.
- **Multi-Agent Systems**: Coordinated drone fleets
- **Transfer Learning**: Adapt to different farm layouts
- **Real-World Deployment**: Integration with actual drone hardware
- **Agricultural Research**: Test precision farming strategies

---

**Mission**: "Using AI to build impactful solutions in agricultural sectors through precision farming and resource optimization." 🌾🤖

*A focused, single-scenario RL environment for agricultural AI research and education.*
