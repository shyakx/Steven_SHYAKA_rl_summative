#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing direct module import...")
    import environment.custom_env as env_module
    print(f"Module imported successfully: {env_module}")
    
    print("Checking module contents...")
    print(f"Module dir: {dir(env_module)}")
    
    print("Testing class import...")
    from environment.custom_env import PrecisionFarmingEnv
    print(f"Class imported successfully: {PrecisionFarmingEnv}")
    
    print("Creating environment instance...")
    env = PrecisionFarmingEnv()
    print(f"Environment created: {env}")
    
    print("Testing reset...")
    obs, info = env.reset()
    print("Reset successful!")
    print(f"Observation keys: {obs.keys()}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
