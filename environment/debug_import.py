#!/usr/bin/env python3

print("Starting import test...")

try:
    print("Importing numpy...")
    import numpy as np
    print("✓ numpy imported")
    
    print("Importing gymnasium...")
    import gymnasium as gym
    print("✓ gymnasium imported")
    
    print("Importing spaces...")
    from gymnasium import spaces
    print("✓ spaces imported")
    
    print("Importing typing...")
    from typing import Tuple, Dict, Any, Optional
    print("✓ typing imported")
    
    print("Importing random...")
    import random
    print("✓ random imported")
    
    print("Now executing custom_env.py content...")
    exec(open('custom_env.py').read())
    
    print("Checking if PrecisionFarmingEnv is in locals:")
    print(f"PrecisionFarmingEnv available: {'PrecisionFarmingEnv' in locals()}")
    
    if 'PrecisionFarmingEnv' in locals():
        print("Creating environment...")
        env = locals()['PrecisionFarmingEnv']()
        print("✓ Environment created successfully!")
    else:
        print("❌ PrecisionFarmingEnv not found in locals")
        print("Available locals:")
        for key in locals().keys():
            if not key.startswith('__'):
                print(f"  - {key}")
                
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
