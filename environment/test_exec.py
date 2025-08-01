#!/usr/bin/env python3

print("Testing custom_env.py execution...")

# Execute the file and capture globals
global_namespace = {}
try:
    exec(open('custom_env.py').read(), global_namespace)
    print("File executed successfully!")
    
    print("Checking for PrecisionFarmingEnv in globals:")
    if 'PrecisionFarmingEnv' in global_namespace:
        print("✓ Found PrecisionFarmingEnv!")
        cls = global_namespace['PrecisionFarmingEnv']
        print(f"Class: {cls}")
        
        # Try to create instance
        env = cls()
        print("✓ Environment instance created!")
        
        # Test reset
        obs, info = env.reset()
        print("✓ Environment reset successful!")
        print(f"Observation keys: {list(obs.keys())}")
        
    else:
        print("❌ PrecisionFarmingEnv not found")
        print("Available items:")
        for key, value in global_namespace.items():
            if not key.startswith('__'):
                print(f"  {key}: {type(value)}")
                
except Exception as e:
    print(f"❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
