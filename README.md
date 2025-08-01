# 🌾 AgriTech Precision Farming - RL School Project

## 📖 Project Overview

This is a **reinforcement learning school project** that implements an autonomous farming drone using 4 different RL algorithms. The drone learns to navigate a farm field and treat diseased crops while managing limited battery and resources.

### 🎯 Project Goal
Train an AI agent to efficiently treat diseased crops in a 15×15 farm grid while managing:
- Battery consumption (limited to 100 units)
- Treatment capacity (20 treatments available)
- Time constraints (200 steps maximum)

---

## 🌾 Environment: Precision Farming Drone

### Environment Details
- **Grid Size**: 15×15 farm field (225 cells)
- **State Space**: 230 features (grid + position + resources)
- **Action Space**: 6 discrete actions
- **Objective**: Treat all diseased crops before battery runs out

### 🎮 Available Actions
| Action | Battery Cost | Description |
|--------|--------------|-------------|
| Move Up/Down/Left/Right | 2 units | Navigate in cardinal directions |
| Treat Crop | 5 units | Apply treatment to diseased crop |
| Charge Battery | 0 units | Recharge at charging station |

### 🌱 Farm Elements
- 🟤 **Empty Soil**: Safe to traverse
- 🟢 **Healthy Crops**: No action needed
- 🔴 **Diseased Crops**: Primary targets for treatment
- 🟫 **Obstacles**: Blocks movement
- 🔵 **Charging Stations**: Restore battery
- 🟡 **Treated Crops**: Successfully completed treatment

---

## 🤖 RL Algorithms Implemented

### 1. DQN (Deep Q-Network)
- **Type**: Value-based learning
- **Parameters**: 19,334
- **Features**: Experience replay, target network

### 2. REINFORCE
- **Type**: Policy gradient
- **Parameters**: 38,343
- **Features**: Monte Carlo updates, baseline network

### 3. PPO (Proximal Policy Optimization)
- **Type**: Advanced policy gradient
- **Parameters**: 38,343
- **Features**: Clipped objective, advantage estimation

### 4. Actor-Critic
- **Type**: Hybrid approach
- **Parameters**: 19,399
- **Features**: Separate policy and value networks

---

## 🚀 How to Run the Project

### 1. Setup and Installation
```bash
# Install required packages
pip install torch numpy matplotlib pygame

# Verify everything works
python system_test.py
```

**Expected Output:**
```
🧪 AgriTech RL System Test
==============================
✅ Environment: Working
✅ Base Agent: State=230, Actions=6
✅ DQN: 19,334 parameters
✅ REINFORCE: 38,343 parameters
✅ PPO: 38,343 parameters
✅ ActorCritic: 19,399 parameters
✅ Training Infrastructure: Working
```

### 2. Test the Environment
```bash
python test_env.py
```

**Expected Output:**
```
🌾 Testing AgriTech Precision Farming Environment
==================================================
✅ Environment created successfully
✅ Environment reset - 19 diseased crops to treat
🎮 Testing actions:
  Step 1: MOVE_LEFT -> Reward: -0.01, Battery: 99%
  Step 2: TREAT_CROP -> Reward: -0.50, Battery: 99%
  Step 3: MOVE_RIGHT -> Reward: -0.01, Battery: 98%
  Step 4: MOVE_RIGHT -> Reward: -0.10, Battery: 98%
  Step 5: MOVE_LEFT -> Reward: -0.01, Battery: 97%
✅ Environment test completed successfully!
```

### 3. Visual Demo
```bash
python demo.py
```
This opens a pygame window showing the drone navigating the farm in real-time.

### 4. Quick Training (All 4 Agents)
```bash
python quick_demo.py
```

---

## 📊 Training Results

### System Test Results
All components working correctly:
- ✅ Environment: 230-dimensional state space, 6 actions
- ✅ DQN Agent: 19,334 parameters
- ✅ REINFORCE Agent: 38,343 parameters  
- ✅ PPO Agent: 38,343 parameters
- ✅ Actor-Critic Agent: 19,399 parameters

### Actual Training Output (100 episodes each)

#### DQN Training Results
```
🚀 Starting DQN training for 100 episodes
📊 State size: 230, Action size: 6
Episode    0: Reward= -36.63, Steps=200, Success=0%, Loss=0.0357
Episode   10: Reward= -17.26, Steps=200, Success=0%, Loss=1.2613
Episode   20: Reward= -24.02, Steps=200, Success=0%, Loss=1.2791
Episode   50: Reward= -20.50, Steps=200, Success=0%, Loss=1.3104
Episode  100: Avg Reward= -25.07, Success Rate=0.0%, Steps=200.0
✅ DQN training completed in 457.2s
```

#### REINFORCE Training Results
```
🚀 Starting REINFORCE training for 100 episodes
📊 State size: 230, Action size: 6
Episode    0: Reward= -38.79, Steps=174, Success=0%, Loss=0.9214
Episode   20: Reward= -22.33, Steps=200, Success=0%, Loss=1.0516
Episode   50: Reward= -41.49, Steps=200, Success=0%, Loss=0.9645
Episode  100: Avg Reward= -31.13, Success Rate=0.0%, Steps=200.0
✅ REINFORCE training completed in 226.5s
```

#### PPO Training Results
```
🚀 Starting PPO training for 100 episodes
📊 State size: 230, Action size: 6
Episode    0: Reward= -13.97, Steps=188, Success=0%
Episode   20: Reward= -42.51, Steps=180, Success=0%
Episode   50: Reward= -17.53, Steps=200, Success=0%
Episode  100: Avg Reward= -20.39, Success Rate=0.0%, Steps=184.2
✅ PPO training completed in 113.8s
```

### Performance Comparison
| Algorithm | Final Avg Reward | Training Time | Efficiency |
|-----------|------------------|---------------|------------|
| **PPO** | -20.39 | 113.8s | **Best** |
| **DQN** | -25.07 | 457.2s | Good |
| **REINFORCE** | -31.13 | 226.5s | Baseline |

**Key Findings:**
- **PPO performed best** with highest average reward (-20.39)
- **PPO was fastest to train** (113.8 seconds)
- **DQN was most stable** but took longest to train
- **All algorithms struggled** with the complex environment (0% success rate indicates room for improvement)

---

## 📁 Project Structure

```
📁 Core Components:
├── environment/custom_env.py     # Main RL environment
├── agents/                       # All 4 RL agent implementations
├── training/trainer.py           # Training infrastructure
├── demo.py                       # Visual demonstration
├── system_test.py               # System validation
└── quick_demo.py                # Train all agents

📁 Outputs:
├── models/                       # Saved trained models (.pth files)
├── logs/                        # Training metrics and plots
└── analysis/                    # Performance comparison reports
```

---

## 💡 Key Learning Outcomes

### Technical Skills Demonstrated
1. **Reinforcement Learning**: Implemented 4 different RL paradigms
2. **Neural Networks**: Used PyTorch for deep learning models
3. **Environment Design**: Created custom OpenAI Gym-style environment
4. **Python Programming**: Object-oriented design with proper documentation
5. **Data Visualization**: Training progress plots and performance analysis

### RL Concepts Applied
- **Value-based learning** (DQN with experience replay)
- **Policy gradient methods** (REINFORCE with baseline)
- **Actor-critic architectures** (PPO and classic AC)
- **Exploration vs exploitation** (epsilon-greedy, entropy regularization)
- **Reward shaping** for agricultural optimization

---

## 🎯 Future Improvements

1. **Environment Enhancements**:
   - Add weather effects
   - Multiple drone coordination
   - Dynamic crop growth

2. **Algorithm Improvements**:
   - Hyperparameter tuning
   - Curriculum learning
   - Multi-objective optimization

3. **Real-world Applications**:
   - Transfer to real farm data
   - Integration with IoT sensors
   - Economic optimization models

---

## 📋 Dependencies

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
pygame>=2.1.0
```

Install with: `pip install torch numpy matplotlib pygame`

---

## 🎉 Project Completion

This project successfully demonstrates:
- ✅ **Custom RL environment** for agricultural applications
- ✅ **4 different RL algorithms** implemented from scratch
- ✅ **Comparative analysis** of algorithm performance
- ✅ **Professional code structure** with documentation
- ✅ **Real training results** with performance metrics

**Total Development Time**: ~40 hours  
**Lines of Code**: ~2,000  
**Technologies Used**: Python, PyTorch, Pygame, Matplotlib

---

*School Project by Steven SHYAKA*  
*Course: Machine Learning/Reinforcement Learning*  
*Completed: August 1, 2025*

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
