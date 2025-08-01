# 🚀 Quick Start Guide - AgriTech RL Platform

## 1️⃣ **Verify Installation** (Required First)
```bash
python system_test.py
```
**Expected output**: ✅ All components working

---

## 2️⃣ **See It In Action** (Recommended)
```bash
python demo.py
```
**Shows**: Interactive pygame visualization of the farming drone

---

## 3️⃣ **Quick Training** (10 minutes)
```bash
python quick_demo.py
```
**Trains**: All 4 RL agents with 100 episodes each
**Outputs**: Models saved to `models/`, logs to `logs/`

---

## 4️⃣ **Validate Models** (Check Training Worked)
```bash
python validate_models.py
```
**Verifies**: Saved models can be loaded and used for inference

---

## 5️⃣ **Full Research Training** (60 minutes)
```bash
python train_all_agents.py
```
**Trains**: All 4 agents with 500 episodes each
**Generates**: Complete comparison report in `analysis/`

---

## 📊 **Check Results**

### Training Outputs
- **`models/`**: Trained model files (.pth format)
- **`logs/`**: Training metrics and plots
- **`analysis/`**: Performance comparison reports

### Key Files
- **`analysis/rl_comparison_report.md`**: Detailed analysis
- **`analysis/agent_comparison.png`**: Performance charts
- **`models/{agent}_final.pth`**: Best trained models

---

## 🤖 **Available Agents**

| Agent | Type | Parameters | Strengths |
|-------|------|------------|-----------|
| **DQN** | Value-based | 46,854 | Stable, experience replay |
| **REINFORCE** | Policy gradient | 38,343 | Simple, direct optimization |
| **PPO** | Advanced policy | 38,343 | Sample efficient, stable |
| **Actor-Critic** | Hybrid | 19,399 | Fast convergence |

---

## 🎯 **Environment Details**
- **Grid**: 15×15 farm field
- **Actions**: 6 discrete (move + treat + charge)
- **Goal**: Treat diseased crops efficiently
- **Challenge**: Resource management (battery + treatment)

---

## ⚡ **Need Help?**

### Quick Fixes
```bash
# If imports fail
pip install -r requirements.txt

# If training fails
python system_test.py

# If models not found
python quick_demo.py
```

### Documentation
- **`README.md`**: Complete project documentation
- **`SCRIPTS_REFERENCE.md`**: All available scripts
- **`PROJECT_SUMMARY.md`**: Project status and cleanup

---

**🌾 Ready to revolutionize agriculture with AI!** 🤖
