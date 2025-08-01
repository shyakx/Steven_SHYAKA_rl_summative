#!/usr/bin/env python3

print("🧪 Testing environment and rendering imports...")

try:
    print("Testing custom_env import...")
    from custom_env import PrecisionFarmingEnv
    print("✅ PrecisionFarmingEnv imported successfully")
    
    # Test creating environment
    env = PrecisionFarmingEnv()
    print("✅ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print("✅ Environment reset successful")
    
except Exception as e:
    print(f"❌ Error with custom_env: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)

try:
    print("Testing rendering import...")
    from rendering import FarmingRenderer
    print("✅ FarmingRenderer imported successfully")
    
    # Test creating renderer
    renderer = FarmingRenderer()
    print("✅ Renderer created successfully")
    
except Exception as e:
    print(f"❌ Error with rendering: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("Import test complete!")
