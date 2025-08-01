"""
Pygame Visualization for AgriTech Precision Farming Environment

Rich visual rendering of the farming drone scenario with:
- Real-time farm grid visualization
- Animated drone with battery indicator
- Resource management display
- Mission progress tracking
"""

import pygame
import numpy as np
from typing import Tuple, Optional, Dict, Any
import os
import sys

# Import our custom environment
from custom_env import PrecisionFarmingEnv


class FarmingRenderer:
    """Pygame-based renderer for the precision farming environment."""
    
    # Visual theme colors
    COLORS = {
        'EMPTY': (101, 67, 33),          # Rich brown soil
        'HEALTHY_CROP': (34, 139, 34),   # Forest green crops
        'DISEASED_CROP': (220, 20, 60),  # Crimson diseased crops
        'OBSTACLE': (105, 105, 105),     # Gray rocks/trees
        'CHARGING_STATION': (30, 144, 255), # Bright blue stations
        'TREATED_CROP': (50, 205, 50),  # Lime green treated crops
        'DRONE': (255, 215, 0),          # Gold drone
        'BACKGROUND': (245, 245, 220),   # Beige background
        'TEXT': (25, 25, 25),            # Dark text
        'GRID_LINE': (200, 200, 200),    # Light gray grid
        'BATTERY_HIGH': (0, 255, 0),     # Green battery
        'BATTERY_MED': (255, 255, 0),    # Yellow battery  
        'BATTERY_LOW': (255, 0, 0),      # Red battery
        'INFO_PANEL': (240, 248, 255),   # Alice blue info panel
        'PANEL_BORDER': (70, 130, 180)   # Steel blue border
    }
    
    def __init__(self, cell_size: int = 30, window_width: int = 800, window_height: int = 600):
        """Initialize the Pygame renderer."""
        pygame.init()
        
        self.cell_size = cell_size
        self.window_width = window_width
        self.window_height = window_height
        
        # Calculate grid display area
        self.grid_width = 15 * cell_size  # 15x15 grid
        self.grid_height = 15 * cell_size
        self.grid_offset_x = 50
        self.grid_offset_y = 50
        
        # Info panel area
        self.info_panel_x = self.grid_offset_x + self.grid_width + 20
        self.info_panel_width = window_width - self.info_panel_x - 20
        
        # Initialize display
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("🌾 AgriTech Precision Farming Drone")
        
        # Initialize fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_small = pygame.font.Font(None, 16)
        
        # Animation variables
        self.drone_animation_offset = 0
        self.animation_speed = 0.1
        
        # Performance tracking
        self.clock = pygame.time.Clock()
        
    def render(self, env: PrecisionFarmingEnv, info: Dict[str, Any]):
        """Render the current state of the environment."""
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen
        self.screen.fill(self.COLORS['BACKGROUND'])
        
        # Draw farm grid
        self._draw_farm_grid(env.grid)
        
        # Draw agent (drone)
        self._draw_agent(env.agent_pos)
        
        # Draw info panel
        self._draw_info_panel(env, info)
        
        # Draw grid lines
        self._draw_grid_lines()
        
        # Update animation
        self.drone_animation_offset += self.animation_speed
        if self.drone_animation_offset > 2 * np.pi:
            self.drone_animation_offset = 0
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
        
        return True
    
    def _draw_farm_grid(self, grid: np.ndarray):
        """Draw the farm grid with different cell types."""
        for i in range(15):
            for j in range(15):
                cell_value = grid[i, j]
                
                # Determine color based on cell type
                if cell_value == PrecisionFarmingEnv.EMPTY:
                    color = self.COLORS['EMPTY']
                elif cell_value == PrecisionFarmingEnv.HEALTHY_CROP:
                    color = self.COLORS['HEALTHY_CROP']
                elif cell_value == PrecisionFarmingEnv.DISEASED_CROP:
                    color = self.COLORS['DISEASED_CROP']
                elif cell_value == PrecisionFarmingEnv.OBSTACLE:
                    color = self.COLORS['OBSTACLE']
                elif cell_value == PrecisionFarmingEnv.CHARGING_STATION:
                    color = self.COLORS['CHARGING_STATION']
                elif cell_value == PrecisionFarmingEnv.TREATED_CROP:
                    color = self.COLORS['TREATED_CROP']
                else:
                    color = self.COLORS['EMPTY']  # Default
                
                # Draw cell
                x = self.grid_offset_x + j * self.cell_size
                y = self.grid_offset_y + i * self.cell_size
                
                pygame.draw.rect(self.screen, color, 
                               (x, y, self.cell_size, self.cell_size))
    
    def _draw_agent(self, agent_pos: Tuple[int, int]):
        """Draw the drone agent with animation."""
        i, j = agent_pos
        
        # Calculate center position
        center_x = self.grid_offset_x + j * self.cell_size + self.cell_size // 2
        center_y = self.grid_offset_y + i * self.cell_size + self.cell_size // 2
        
        # Add subtle floating animation
        animation_y = int(3 * np.sin(self.drone_animation_offset))
        center_y += animation_y
        
        # Draw drone as a circle with propeller effect
        drone_radius = self.cell_size // 3
        pygame.draw.circle(self.screen, self.COLORS['DRONE'], 
                          (center_x, center_y), drone_radius)
        
        # Draw propeller animation (rotating lines)
        prop_length = drone_radius + 3
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            # Rotate propellers based on animation offset
            rotated_angle = angle + self.drone_animation_offset * 4
            end_x = center_x + int(prop_length * np.cos(rotated_angle))
            end_y = center_y + int(prop_length * np.sin(rotated_angle))
            
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (center_x, center_y), (end_x, end_y), 2)
    
    def _draw_grid_lines(self):
        """Draw grid lines for better visibility."""
        # Vertical lines
        for i in range(16):  # 15 cells = 16 lines
            x = self.grid_offset_x + i * self.cell_size
            pygame.draw.line(self.screen, self.COLORS['GRID_LINE'],
                           (x, self.grid_offset_y), 
                           (x, self.grid_offset_y + self.grid_height), 1)
        
        # Horizontal lines
        for i in range(16):
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLORS['GRID_LINE'],
                           (self.grid_offset_x, y), 
                           (self.grid_offset_x + self.grid_width, y), 1)
    
    def _draw_info_panel(self, env: PrecisionFarmingEnv, info: Dict[str, Any]):
        """Draw information panel with current stats."""
        panel_rect = pygame.Rect(self.info_panel_x, self.grid_offset_y, 
                               self.info_panel_width, self.grid_height)
        
        # Draw panel background
        pygame.draw.rect(self.screen, self.COLORS['INFO_PANEL'], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS['PANEL_BORDER'], panel_rect, 2)
        
        # Starting position for text
        text_x = self.info_panel_x + 10
        text_y = self.grid_offset_y + 15
        line_height = 25
        
        # Title
        title_text = self.font_large.render("🌾 Mission Control", True, self.COLORS['TEXT'])
        self.screen.blit(title_text, (text_x, text_y))
        text_y += line_height + 10
        
        # Drone Status
        status_text = self.font_medium.render("🚁 Drone Status:", True, self.COLORS['TEXT'])
        self.screen.blit(status_text, (text_x, text_y))
        text_y += line_height
        
        # Position
        pos_text = f"Position: ({env.agent_pos[0]}, {env.agent_pos[1]})"
        pos_surface = self.font_small.render(pos_text, True, self.COLORS['TEXT'])
        self.screen.blit(pos_surface, (text_x + 10, text_y))
        text_y += line_height
        
        # Battery with color coding
        battery_level = env.battery_level
        if battery_level > 60:
            battery_color = self.COLORS['BATTERY_HIGH']
        elif battery_level > 30:
            battery_color = self.COLORS['BATTERY_MED']
        else:
            battery_color = self.COLORS['BATTERY_LOW']
        
        battery_text = f"Battery: {battery_level}%"
        battery_surface = self.font_small.render(battery_text, True, battery_color)
        self.screen.blit(battery_surface, (text_x + 10, text_y))
        text_y += line_height
        
        # Treatment capacity
        treatment_text = f"Treatment: {env.treatment_capacity}"
        treatment_surface = self.font_small.render(treatment_text, True, self.COLORS['TEXT'])
        self.screen.blit(treatment_surface, (text_x + 10, text_y))
        text_y += line_height + 10
        
        # Mission Progress
        progress_text = self.font_medium.render("📊 Mission Progress:", True, self.COLORS['TEXT'])
        self.screen.blit(progress_text, (text_x, text_y))
        text_y += line_height
        
        # Diseased crops count
        diseased_text = f"Diseased Crops: {env.current_diseased_count}"
        diseased_surface = self.font_small.render(diseased_text, True, self.COLORS['DISEASED_CROP'])
        self.screen.blit(diseased_surface, (text_x + 10, text_y))
        text_y += line_height
        
        # Completion rate
        if env.initial_diseased_count > 0:
            completion_rate = (env.initial_diseased_count - env.current_diseased_count) / env.initial_diseased_count * 100
        else:
            completion_rate = 100.0
        
        completion_text = f"Completion: {completion_rate:.1f}%"
        completion_surface = self.font_small.render(completion_text, True, self.COLORS['TEXT'])
        self.screen.blit(completion_surface, (text_x + 10, text_y))
        text_y += line_height
        
        # Steps taken
        steps_text = f"Steps: {env.steps_taken}/{env.MAX_STEPS}"
        steps_surface = self.font_small.render(steps_text, True, self.COLORS['TEXT'])
        self.screen.blit(steps_surface, (text_x + 10, text_y))
        text_y += line_height + 10
        
        # Legend
        legend_text = self.font_medium.render("🗺️ Legend:", True, self.COLORS['TEXT'])
        self.screen.blit(legend_text, (text_x, text_y))
        text_y += line_height
        
        # Legend items
        legend_items = [
            ("🌱 Healthy Crop", self.COLORS['HEALTHY_CROP']),
            ("🔴 Diseased Crop", self.COLORS['DISEASED_CROP']),
            ("✅ Treated Crop", self.COLORS['TREATED_CROP']),
            ("🪨 Obstacle", self.COLORS['OBSTACLE']),
            ("⚡ Charging Station", self.COLORS['CHARGING_STATION'])
        ]
        
        for legend_item, color in legend_items:
            # Draw small color square
            color_rect = pygame.Rect(text_x + 10, text_y + 2, 12, 12)
            pygame.draw.rect(self.screen, color, color_rect)
            pygame.draw.rect(self.screen, self.COLORS['TEXT'], color_rect, 1)
            
            # Draw text
            legend_surface = self.font_small.render(legend_item, True, self.COLORS['TEXT'])
            self.screen.blit(legend_surface, (text_x + 25, text_y))
            text_y += 16
    
    def close(self):
        """Clean up pygame resources."""
        pygame.quit()


# Convenience function for easy rendering
def create_renderer(**kwargs):
    """Create a FarmingRenderer instance."""
    return FarmingRenderer(**kwargs)


if __name__ == "__main__":
    # Test the renderer
    print("🎨 Testing Pygame Renderer")
    print("=" * 30)
    
    # Test if pygame is available
    try:
        pygame.init()
        print("✅ Pygame initialized successfully")
        
        # Test renderer creation
        renderer = FarmingRenderer()
        print("✅ Renderer created successfully")
        
        # Test with environment
        env = PrecisionFarmingEnv()
        obs, info = env.reset()
        
        print("✅ Environment setup complete")
        print("📝 Close the window to end the test")
        
        # Simple render loop
        running = True
        step_count = 0
        
        while running and step_count < 100:  # Max 100 steps for demo
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Take a random action occasionally
            if step_count % 20 == 0:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    obs, info = env.reset()
                    step_count = 0
            
            # Render
            running = renderer.render(env, info)
            step_count += 1
        
        renderer.close()
        print("✅ Renderer test completed successfully!")
        
    except ImportError:
        print("❌ Pygame not available - install with: pip install pygame")
    except Exception as e:
        print(f"❌ Error during rendering test: {e}")
        import traceback
        traceback.print_exc()
