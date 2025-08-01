# 🌾 AgriTech Precision Farming - Complete Scripts Reference

## 📖 **Project Overview**
This document provides a comprehensive guide to all available scripts in the AgriTech Precision Farming RL project. The project implements 4 different reinforcement learning algorithms (DQN, REINFORCE, PPO, Actor-Critic) to train an autonomous farming drone.

---

## 🎯 **Main Entry Points**

### **🔧 Environment Testing & Validation Scripts**

#### `simple_test.py`
```bash
python simple_test.py
```
**Purpose**: Basic environment functionality test
- Tests if PrecisionFarmingEnv can be imported and instantiated
- Validates environment reset and basic step functionality
- Quick sanity check before running larger experiments

#### `test_env.py`
```bash
python test_env.py
```
**Purpose**: Comprehensive environment testing
- Tests environment creation, reset, and action execution
- Displays sample action results with rewards and battery levels
- Validates all 6 actions work correctly

#### `system_test.py`
```bash
python system_test.py
```
**Purpose**: Full system validation
- Tests all components: environment, agents, training infrastructure
- Validates imports for all 4 RL agents
- Comprehensive pre-training system check
- **Run this first** to ensure everything is working

#### `environment/custom_env.py`
```bash
python environment/custom_env.py
```
**Purpose**: Direct environment testing
- Standalone environment execution
- Basic functionality demonstration
- Minimal dependencies test

---

### **🎮 Demo & Visualization Scripts**

#### `demo.py`
```bash
python demo.py
```
**Purpose**: Interactive visual demonstration
- Real-time pygame visualization of the farming environment
- Animated drone with rotating propellers
- Shows intelligent agent behavior navigating and treating crops
- Interactive demo with live resource tracking
- **Best for understanding how the environment works visually**

#### `environment/test_exec.py`
```bash
python environment/test_exec.py
```
**Purpose**: Environment execution test
- Basic environment functionality validation
- Quick execution test for the custom environment

---

### **🤖 Agent Validation Scripts**

#### `validate_agents.py`
```bash
python validate_agents.py
```
**Purpose**: Quick validation of all 4 RL agents
- Tests that all agents (DQN, REINFORCE, PPO, Actor-Critic) can be instantiated
- Validates agent creation with proper configurations
- Quick compatibility check without training
- **Essential pre-training validation**

#### `single_agent_test.py`
```bash
python single_agent_test.py
```
**Purpose**: Individual agent testing
- Test each agent type separately with different configurations
- Detailed agent-specific validation
- Useful for debugging specific agent issues

---

### **🚀 Training Scripts - Individual Agents**

#### `minimal_train.py`
```bash
python minimal_train.py
```
**Purpose**: Minimal DQN training session
- Quick DQN training with basic configuration
- Short training session for testing pipeline
- Validates complete training loop works
- **Good first training test**

#### `train_dqn_demo.py`
```bash
python train_dqn_demo.py
```
**Purpose**: DQN-only training demonstration
- Focused DQN training with 200 episodes
- Demonstrates complete DQN training pipeline
- Saves trained model and generates basic metrics

#### `train_dqn_simple.py`
```bash
python train_dqn_simple.py
```
**Purpose**: Simple DQN training launcher
- Basic DQN configuration with standard parameters
- Straightforward training session
- Good for learning DQN implementation

#### Individual Agent Scripts
```bash
python agents/dqn_agent.py           # Test DQN agent implementation
python agents/reinforce_agent.py     # Test REINFORCE agent implementation
python agents/ppo_agent.py           # Test PPO agent implementation
python agents/actor_critic_agent.py  # Test Actor-Critic agent implementation
```
**Purpose**: Direct agent testing
- Run individual agent implementations
- Test agent-specific functionality
- Validate neural network architectures

---

### **🏆 Multi-Agent Training Scripts**

#### `train_all_agents.py` ⭐ **MAIN SCRIPT**
```bash
python train_all_agents.py
```
**Purpose**: Complete training pipeline
- Trains all 4 RL agents (DQN, REINFORCE, PPO, Actor-Critic)
- 500 episodes per agent (comprehensive training)
- Generates performance comparison charts
- Creates detailed markdown report with analysis
- Saves all models, logs, and metrics
- **This is your primary research script**

**Outputs**:
- `models/`: Trained model files (.pth format)
- `logs/`: Training logs and metrics
- `analysis/`: Comparison report and visualizations
- `analysis/rl_comparison_report.md`: Detailed analysis report
- `analysis/agent_comparison.png`: Performance comparison charts

#### `complete_demo.py`
```bash
python complete_demo.py
```
**Purpose**: Sequential training of all 4 agents
- Train all agents one by one with detailed logging
- Comprehensive training with full configurations
- Alternative to `train_all_agents.py` with different approach

#### `quick_demo.py`
```bash
python quick_demo.py
```
**Purpose**: Fast multi-agent training
- Trains all 4 agents with reduced episodes (100 each)
- Quick comparison for testing purposes
- **Ideal for rapid prototyping and testing**
- Generates basic comparison metrics

#### `sequential_comparison.py`
```bash
python sequential_comparison.py
```
**Purpose**: Sequential agent comparison
- Train and compare agents one by one
- Detailed metrics collection for each agent
- Step-by-step comparative analysis

---

### **🔥 Batch Launcher**

#### `launch_training.bat` (Windows)
```cmd
.\launch_training.bat
```
**Purpose**: Interactive training menu
- Menu-driven interface for choosing training options
- Options include:
  1. Quick validation (`validate_agents.py`)
  2. Minimal DQN training (`minimal_train.py`)
  3. Complete 4-agent demo (`complete_demo.py`)
  4. Full training pipeline (`train_all_agents.py`)

---

## 📊 **Output Structure**

### **Generated Directories**
- **`models/`**: Saved model files (.pth and .pkl formats)
- **`logs/`**: Training logs, metrics JSON files, and training plots
- **`analysis/`**: Comparison reports, charts, and analysis files

### **Key Output Files**
- **`analysis/rl_comparison_report.md`**: Comprehensive analysis report
- **`analysis/agent_comparison.png`**: Performance comparison visualizations
- **`analysis/all_metrics.json`**: Raw metrics data for further analysis
- **`logs/{agent_name}_training.log`**: Individual agent training logs
- **`models/{agent_name}_final.pth`**: Final trained model files

---

## 🚀 **Recommended Execution Order**

### **Phase 1: Verification (Required First)**
```bash
# 1. Verify system components
python system_test.py

# 2. Test environment functionality
python simple_test.py

# 3. Validate all agents can be created
python validate_agents.py
```

### **Phase 2: Understanding the Environment**
```bash
# 4. See the environment in action
python demo.py

# 5. Test basic training works
python minimal_train.py
```

### **Phase 3: Quick Experiments**
```bash
# 6. Fast training of all agents (testing)
python quick_demo.py

# 7. Single agent focused training
python train_dqn_simple.py
```

### **Phase 4: Full Research**
```bash
# 8. Complete research-grade training
python train_all_agents.py
```

---

## 🤖 **Agent Implementations**

### **DQN (Deep Q-Network)**
- **Parameters**: 46,854
- **Type**: Value-based method
- **Features**: Experience replay, target network, epsilon-greedy exploration

### **REINFORCE**
- **Parameters**: 46,854 (policy) + baseline network
- **Type**: Policy gradient method
- **Features**: Monte Carlo policy gradient with baseline for variance reduction

### **PPO (Proximal Policy Optimization)**
- **Parameters**: 93,063
- **Type**: Advanced policy gradient
- **Features**: Clipped objective, GAE (Generalized Advantage Estimation), actor-critic structure

### **Actor-Critic**
- **Parameters**: 46,983
- **Type**: Classic actor-critic
- **Features**: Separate policy (actor) and value (critic) networks

---

## 🌾 **Environment Details**

### **PrecisionFarmingEnv Specifications**
- **Grid Size**: 15×15 farm field
- **State Space**: 230-dimensional (grid + position + resources)
- **Action Space**: 6 discrete actions
- **Mission**: Treat all diseased crops while managing battery and treatment resources

### **Actions Available**
1. **Move Up** (2 battery units)
2. **Move Down** (2 battery units) 
3. **Move Left** (2 battery units)
4. **Move Right** (2 battery units)
5. **Treat Crop** (5 battery units)
6. **Charge Battery** (0 battery units)

### **Cell Types**
- 🟤 Empty soil
- 🟢 Healthy crops
- 🔴 Diseased crops (targets)
- 🟫 Obstacles 
- 🔵 Charging stations
- 🟡 Treated crops

---

## 💡 **Usage Tips**

### **For First-Time Users**
1. Start with `python system_test.py`
2. Run `python demo.py` to see the environment
3. Try `python quick_demo.py` for fast results

### **For Research**
1. Use `python train_all_agents.py` for comprehensive comparison
2. Check `analysis/rl_comparison_report.md` for detailed results
3. Models are saved in `models/` for further use

### **For Debugging**
1. Use individual agent scripts for specific testing
2. Run `python validate_agents.py` for quick agent checks
3. Check `logs/` directory for training details

---

## ⚠️ **Important Notes**

- **Windows Users**: Use PowerShell or Command Prompt
- **Dependencies**: Ensure all requirements from `requirements.txt` are installed
- **GPU**: CUDA support optional but recommended for faster training
- **Memory**: PPO agent requires more memory due to larger parameter count
- **Time**: Full training (`train_all_agents.py`) takes significant time (~30-60 minutes)

---

## 🎯 **Project Status**

✅ **All components implemented and tested**  
✅ **Ready for immediate use**  
✅ **Production-quality code**  
✅ **Comprehensive documentation**  

**The system is 100% ready for RL research and experimentation!** 🌾🤖

---

*Generated on: August 1, 2025*  
*Project: AgriTech Precision Farming RL Research Platform*
