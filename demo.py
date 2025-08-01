"""
AgriTech Precision Farming Demo

Interactive demonstration of the farming drone environment
showcasing the single mission-aligned scenario.
"""

import sys
import os
import time
import numpy as np

# Add environment to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))

from custom_env import PrecisionFarmingEnv
from rendering import FarmingRenderer


def run_interactive_demo():
    """Run interactive demonstration of the farming environment."""
    
    print("🌾 AgriTech Precision Farming - Interactive Demo")
    print("=" * 50)
    print("Mission: Autonomous drone treats diseased crops while managing resources")
    print()
    
    # Create environment and renderer
    env = PrecisionFarmingEnv()
    renderer = FarmingRenderer(cell_size=32)
    
    # Reset environment
    obs, info = env.reset()
    
    print(f"🚁 Mission Briefing:")
    print(f"   • Farm Size: {env.GRID_SIZE}x{env.GRID_SIZE} grid")
    print(f"   • Diseased Crops to Treat: {info['initial_diseased_count']}")
    print(f"   • Initial Battery: {info['battery_level']}%")
    print(f"   • Treatment Capacity: {info['treatment_capacity']} units")
    print(f"   • Drone Position: {info['agent_pos']}")
    print()
    print("🎮 Controls: ESC to exit, Close window to quit")
    print("🤖 Watching intelligent agent navigate and treat crops...")
    print()
    
    # Simple intelligent agent
    def get_smart_action(obs, env):
        """Improved heuristic agent for demonstration."""
        grid = obs['grid']
        agent_pos = obs['agent_pos']
        resources = obs['resources']
        
        battery_level = resources[0]
        treatment_capacity = resources[1]
        
        row, col = agent_pos
        current_cell = grid[row, col]
        
        # Priority 1: If battery very low, find charging station
        if battery_level < 20:
            # Find nearest charging station
            charging_stations = np.where(grid == env.CHARGING_STATION)
            if len(charging_stations[0]) > 0:
                min_dist = float('inf')
                target_row, target_col = None, None
                
                for i in range(len(charging_stations[0])):
                    station_row, station_col = charging_stations[0][i], charging_stations[1][i]
                    dist = abs(station_row - row) + abs(station_col - col)
                    if dist < min_dist:
                        min_dist = dist
                        target_row, target_col = station_row, station_col
                
                # Move towards nearest charging station
                if target_row is not None:
                    if target_row < row and row > 0 and grid[row-1, col] != env.OBSTACLE:
                        return env.ACTION_MOVE_UP
                    elif target_row > row and row < env.GRID_SIZE - 1 and grid[row+1, col] != env.OBSTACLE:
                        return env.ACTION_MOVE_DOWN
                    elif target_col < col and col > 0 and grid[row, col-1] != env.OBSTACLE:
                        return env.ACTION_MOVE_LEFT
                    elif target_col > col and col < env.GRID_SIZE - 1 and grid[row, col+1] != env.OBSTACLE:
                        return env.ACTION_MOVE_RIGHT
            
            # If at charging station, charge
            if current_cell == env.CHARGING_STATION:
                return env.ACTION_CHARGE_BATTERY
        
        # Priority 2: Treat diseased crop if on one
        if current_cell == env.DISEASED_CROP and treatment_capacity > 0:
            return env.ACTION_TREAT_CROP
        
        # Priority 3: Charge if at station and not full battery
        if current_cell == env.CHARGING_STATION and battery_level < 90:
            return env.ACTION_CHARGE_BATTERY
        
        # Priority 4: Move towards nearest diseased crop
        diseased_positions = np.where(grid == env.DISEASED_CROP)
        if len(diseased_positions[0]) > 0:
            min_dist = float('inf')
            target_row, target_col = None, None
            
            # Find nearest diseased crop
            for i in range(len(diseased_positions[0])):
                diseased_row, diseased_col = diseased_positions[0][i], diseased_positions[1][i]
                dist = abs(diseased_row - row) + abs(diseased_col - col)
                if dist < min_dist:
                    min_dist = dist
                    target_row, target_col = diseased_row, diseased_col
            
            # Move towards target with basic pathfinding
            if target_row is not None:
                # Prefer vertical movement first, then horizontal
                if target_row < row and row > 0 and grid[row-1, col] != env.OBSTACLE:
                    return env.ACTION_MOVE_UP
                elif target_row > row and row < env.GRID_SIZE - 1 and grid[row+1, col] != env.OBSTACLE:
                    return env.ACTION_MOVE_DOWN
                elif target_col < col and col > 0 and grid[row, col-1] != env.OBSTACLE:
                    return env.ACTION_MOVE_LEFT
                elif target_col > col and col < env.GRID_SIZE - 1 and grid[row, col+1] != env.OBSTACLE:
                    return env.ACTION_MOVE_RIGHT
        
        # Fallback: Find any valid movement
        possible_moves = []
        for action, dr, dc in [(env.ACTION_MOVE_UP, -1, 0), (env.ACTION_MOVE_DOWN, 1, 0),
                              (env.ACTION_MOVE_LEFT, 0, -1), (env.ACTION_MOVE_RIGHT, 0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < env.GRID_SIZE and 0 <= new_col < env.GRID_SIZE and
                grid[new_row, new_col] != env.OBSTACLE):
                possible_moves.append(action)
        
        if possible_moves:
            return np.random.choice(possible_moves)
        
        return env.ACTION_MOVE_RIGHT  # Last resort
    
    # Main demo loop
    running = True
    step_delay = 0
    total_steps = 0
    
    while running and total_steps < env.MAX_STEPS:
        # Handle events and render
        running = renderer.render(env, info)
        
        # Take action every few frames for visibility
        step_delay += 1
        if step_delay >= 20:  # Slower for better observation
            # Get intelligent action
            action = get_smart_action(obs, env)
            action_name = env.ACTIONS[action]
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Print step info
            print(f"Step {info['steps_taken']:3d}: {action_name:15s} -> "
                  f"Reward: {reward:6.2f} | Battery: {info['battery_level']:3d}% | "
                  f"Diseased Remaining: {info['current_diseased_count']:2d} | "
                  f"Progress: {info['completion_rate']:5.1f}%")
            
            step_delay = 0
            total_steps += 1
            
            # Check episode end
            if terminated or truncated:
                print(f"\n{'='*60}")
                if info['current_diseased_count'] == 0:
                    print(f"🎉 MISSION ACCOMPLISHED!")
                    print(f"   All crops treated in {info['steps_taken']} steps!")
                elif info['battery_level'] <= 0:
                    print(f"🔋 MISSION FAILED - Battery Depleted")
                    print(f"   Ran out of power after {info['steps_taken']} steps")
                else:
                    print(f"⏰ MISSION TIMEOUT")
                    print(f"   Time limit reached after {info['steps_taken']} steps")
                
                print(f"\n📊 Final Mission Statistics:")
                print(f"   • Completion Rate: {info['completion_rate']:.1f}%")
                print(f"   • Diseased Crops Remaining: {info['current_diseased_count']}")
                print(f"   • Battery Remaining: {info['battery_level']}%")
                print(f"   • Treatment Remaining: {info['treatment_capacity']}/20")
                
                # Show final state for a few seconds
                print(f"\nShowing final state for 3 seconds...")
                for _ in range(60):  # 3 seconds at ~20 FPS (reduced for stability)
                    if not renderer.render(env, info):
                        break
                
                # Ask to restart
                print(f"\nPress Enter in terminal to restart mission...")
                input()
                
                # Reset for new mission
                obs, info = env.reset()
                print(f"\n🔄 NEW MISSION STARTED!")
                print(f"   New diseased crops to treat: {info['current_diseased_count']}")
                total_steps = 0
    
    renderer.close()
    print("\nDemo completed! Thank you for watching the AgriTech mission! 🌾")


def run_environment_test():
    """Run basic environment functionality test."""
    print("🧪 Testing AgriTech Environment Functionality")
    print("=" * 45)
    
    env = PrecisionFarmingEnv()
    
    # Test reset
    obs, info = env.reset()
    print(f"✅ Environment reset successful")
    print(f"   Grid shape: {obs['grid'].shape}")
    print(f"   Agent position: {obs['agent_pos']}")
    print(f"   Resources: {obs['resources']}")
    print(f"   Initial diseased crops: {info['initial_diseased_count']}")
    
    # Test actions
    print(f"\n🎮 Testing all actions:")
    for action_id, action_name in env.ACTIONS.items():
        obs, reward, terminated, truncated, info = env.step(action_id)
        print(f"   Action {action_id} ({action_name}): Reward = {reward:.2f}")
        
        if terminated or truncated:
            print(f"   Episode ended after action {action_name}")
            break
    
    print(f"\n✅ Environment test completed successfully!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AgriTech Precision Farming Demo")
    parser.add_argument("--mode", choices=["demo", "test"], default="demo",
                       help="Run interactive demo or functionality test")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "test":
            run_environment_test()
        else:
            run_interactive_demo()
    
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
