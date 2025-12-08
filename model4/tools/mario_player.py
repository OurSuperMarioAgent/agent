import gym_super_mario_bros
import pygame
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import sys
import os

# ===================== Config (Customizable) =====================
SCALE_FACTOR = 2.5  # Screen zoom factor (2.5x for clarity)
FPS = 60  # Game frame rate (60fps for smoothness)
CROP_TOP = 20  # Crop top pixels
CROP_BOTTOM = 220  # Crop bottom pixels
CROP_LEFT = 0  # Crop left pixels
CROP_RIGHT = 256  # Crop right pixels
SHOW_CROP_PREVIEW = True  # Show crop area preview
# ==============================================================

class EnhancedMarioPlayer:
    def __init__(self):
        # Initialize environment
        self.env = gym_super_mario_bros.make('SuperMarioBros-v2')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.obs = self.env.reset()
        self.original_h, self.original_w = self.obs.shape[0], self.obs.shape[1]
        
        # Initialize Pygame (no external resources)
        pygame.init()
        
        # Window setup
        self.screen_w = int(self.original_w * SCALE_FACTOR)
        self.screen_h = int(self.original_h * SCALE_FACTOR)
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("✨ Enhanced Mario (Manual Mode) | Press F1 for Controls")
        
        # Font setup (system default, no external files)
        self.font_small = pygame.font.SysFont(None, 16)
        self.font_medium = pygame.font.SysFont(None, 20)
        self.font_large = pygame.font.SysFont(None, 24)
        
        # Game state
        self.paused = False
        self.game_over = False
        self.clear_level = False
        self.last_score = 0
        self.frame_count = 0
        self.clock = pygame.time.Clock()
        
        # Key mapping (dual-mode support)
        self.key_mapping = {
            "right": [pygame.K_RIGHT, pygame.K_d],  # Right arrow / D
            "left": [pygame.K_LEFT, pygame.K_a],    # Left arrow / A
            "jump": [pygame.K_SPACE, pygame.K_w],   # Space / W (Jump)
            "speed": [pygame.K_x, pygame.K_s],      # X / S (Speed)
            "pause": pygame.K_p,                    # P (Pause)
            "reset": pygame.K_r,                    # R (Reset level)
            "help": pygame.K_F1,                    # F1 (Show controls)
            "quit": pygame.K_ESCAPE                 # ESC (Quit)
        }
        
        # Show help on start
        self.show_help = True

    def is_key_pressed(self, key_group):
        """Check if any key in the group is pressed"""
        keys = pygame.key.get_pressed()
        if isinstance(key_group, list):
            return any(keys[key] for key in key_group)
        return keys[key_group]

    def get_action(self):
        """Optimized action mapping (anti-shake + priority)"""
        action = 0  # Default: NOOP
        
        # Right movement logic (priority: jump+speed > speed > jump > move)
        if self.is_key_pressed(self.key_mapping["right"]):
            if self.is_key_pressed(self.key_mapping["jump"]) and self.is_key_pressed(self.key_mapping["speed"]):
                action = 4  # Right + Jump + Speed
            elif self.is_key_pressed(self.key_mapping["speed"]):
                action = 3  # Right + Speed
            elif self.is_key_pressed(self.key_mapping["jump"]):
                action = 2  # Right + Jump
            else:
                action = 1  # Right only
        # Left movement logic
        elif self.is_key_pressed(self.key_mapping["left"]):
            action = 6  # Left only
        # Jump in place
        elif self.is_key_pressed(self.key_mapping["jump"]):
            action = 5  # Jump in place
        
        return action

    def draw_info_panel(self):
        """Draw real-time info panel (non-obstructive)"""
        # Semi-transparent panel background
        panel_surface = pygame.Surface((300, 160), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))  # Black semi-transparent
        self.screen.blit(panel_surface, (10, 10))
        
        # Get real-time game info
        current_info = self.env.unwrapped._get_info()
        score = current_info["score"]
        x_pos = current_info["x_pos"]
        y_pos = current_info["y_pos"]
        lives = current_info["life"]
        fps = int(self.clock.get_fps())
        action_name = SIMPLE_MOVEMENT[self.get_action()]
        
        # Draw text
        info_texts = [
            f"🎮 FPS: {fps} / {FPS}",
            f"🏆 Score: {score}",
            f"📍 Position: X={x_pos:.0f} Y={y_pos:.0f}",
            f"❤️ Lives: {lives}",
            f"🎯 Current Action: {action_name}",
            f"✂️ Crop Area: Y[{CROP_TOP}-{CROP_BOTTOM}]",
            f"📌 State: {'Paused' if self.paused else 'Running'}"
        ]
        
        for i, text in enumerate(info_texts):
            text_surface = self.font_small.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (20, 20 + i * 20))

    def draw_crop_preview(self):
        """Draw crop area highlight (semi-transparent mask + guides)"""
        # Mask for areas outside crop range
        crop_mask = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        
        # Top mask (above crop line)
        top_rect = pygame.Rect(0, 0, self.screen_w, CROP_TOP * SCALE_FACTOR)
        crop_mask.fill((0, 0, 0, 120), top_rect)
        
        # Bottom mask (below crop line)
        bottom_rect = pygame.Rect(0, CROP_BOTTOM * SCALE_FACTOR, self.screen_w, self.screen_h - CROP_BOTTOM * SCALE_FACTOR)
        crop_mask.fill((0, 0, 0, 120), bottom_rect)
        
        self.screen.blit(crop_mask, (0, 0))
        
        # Crop guide lines (colored)
        pygame.draw.line(
            self.screen, (0, 255, 0),  # Green top line
            (0, CROP_TOP * SCALE_FACTOR),
            (self.screen_w, CROP_TOP * SCALE_FACTOR),
            3
        )
        pygame.draw.line(
            self.screen, (255, 0, 0),  # Red bottom line
            (0, CROP_BOTTOM * SCALE_FACTOR),
            (self.screen_w, CROP_BOTTOM * SCALE_FACTOR),
            3
        )
        
        # Crop area label
        crop_text = self.font_medium.render(f"Cropped Area (Game Content)", True, (255, 255, 0))
        self.screen.blit(crop_text, (self.screen_w // 2 - 120, CROP_TOP * SCALE_FACTOR + 5))

    def draw_help_screen(self):
        """Draw control help screen"""
        help_surface = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        help_surface.fill((0, 0, 0, 220))
        self.screen.blit(help_surface, (0, 0))
        
        help_texts = [
            "📋 Mario Controls",
            "",
            "【Movement】",
            "→ / D: Move Right",
            "← / A: Move Left",
            "Space / W: Jump",
            "X / S: Speed (hold with direction)",
            "",
            "【Shortcuts】",
            "P: Pause / Resume",
            "R: Reset Current Level",
            "F1: Show / Hide Controls",
            "ESC: Quit Game (Confirm)",
            "",
            "【Crop Guide】",
            f"Green Line (Y={CROP_TOP}): Crop top area",
            f"Red Line (Y={CROP_BOTTOM}): Crop bottom area",
            "",
            "💡 Tip: Press any key to close this menu"
        ]
        
        for i, text in enumerate(help_texts):
            color = (255, 255, 255) if text else (0,0,0)
            text_surface = self.font_medium.render(text, True, color)
            self.screen.blit(text_surface, (self.screen_w // 2 - 180, self.screen_h // 2 - 200 + i * 25))

    def draw_status_popup(self, title, content):
        """Draw status popup (clear/over/pause)"""
        popup_surface = pygame.Surface((400, 200), pygame.SRCALPHA)
        popup_surface.fill((0, 0, 0, 200))
        popup_rect = popup_surface.get_rect(center=(self.screen_w//2, self.screen_h//2))
        self.screen.blit(popup_surface, popup_rect)
        
        # Title
        title_surface = self.font_large.render(title, True, (255, 215, 0))  # Gold color
        self.screen.blit(title_surface, (popup_rect.x + 20, popup_rect.y + 30))
        
        # Content
        content_surface = self.font_medium.render(content, True, (255, 255, 255))
        self.screen.blit(content_surface, (popup_rect.x + 20, popup_rect.y + 80))
        
        # Hint
        hint_surface = self.font_small.render("Press any key to continue...", True, (180, 180, 180))
        self.screen.blit(hint_surface, (popup_rect.x + 20, popup_rect.y + 130))

    def handle_events(self):
        """Handle game events (keys/window)"""
        for event in pygame.event.get():
            # Close window
            if event.type == pygame.QUIT:
                self.quit_game()
            
            # Key press events
            if event.type == pygame.KEYDOWN:
                # Toggle help screen
                if event.key == self.key_mapping["help"]:
                    self.show_help = not self.show_help
                
                # Pause / Resume
                if event.key == self.key_mapping["pause"] and not self.show_help:
                    self.paused = not self.paused
                    pygame.display.set_caption(f"✨ Enhanced Mario (Manual Mode) | {'Paused' if self.paused else 'Running'} | Press F1 for Controls")
                
                # Reset level
                if event.key == self.key_mapping["reset"] and not self.show_help:
                    self.obs = self.env.reset()
                    self.game_over = False
                    self.clear_level = False
                
                # Quit confirmation
                if event.key == self.key_mapping["quit"]:
                    self.quit_confirmation()
                
                # Close help/popup
                if self.show_help or self.game_over or self.clear_level:
                    self.show_help = False
                    self.game_over = False
                    self.clear_level = False

    def quit_confirmation(self):
        """Quit confirmation popup"""
        confirm_surface = pygame.Surface((300, 150), pygame.SRCALPHA)
        confirm_surface.fill((0, 0, 0, 220))
        self.screen.blit(confirm_surface, (self.screen_w//2 - 150, self.screen_h//2 - 75))
        
        confirm_texts = [
            "❓ Confirm Quit?",
            "",
            "Y: Confirm Quit",
            "N: Cancel"
        ]
        
        for i, text in enumerate(confirm_texts):
            text_surface = self.font_medium.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (self.screen_w//2 - 120, self.screen_h//2 - 50 + i * 30))
        
        pygame.display.flip()
        
        # Wait for user input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        self.quit_game()
                        waiting = False
                    elif event.key == pygame.K_n:
                        waiting = False

    def quit_game(self):
        """Safely exit game"""
        self.env.close()
        pygame.quit()
        sys.exit()

    def run(self):
        """Main game loop"""
        print("🚀 Enhanced Mario (Manual Mode) Started!")
        print("💡 Press F1 for Controls")
        
        while True:
            # Handle events
            self.handle_events()
            
            if not self.paused and not self.show_help:
                # Get action and step
                action = self.get_action()
                self.obs, reward, done, info = self.env.step(action)
                
                # Check level clear / game over
                if info["flag_get"]:
                    self.clear_level = True
                if done and not info["flag_get"]:
                    self.game_over = True
                
                # Auto-reset level
                if done:
                    self.obs = self.env.reset()
                
                # Update last score
                if info["score"] > self.last_score:
                    self.last_score = info["score"]
            
            # Render game frame (smooth scale + anti-aliasing)
            frame = pygame.surfarray.make_surface(np.transpose(self.obs, (1, 0, 2)))
            frame = pygame.transform.smoothscale(frame, (self.screen_w, self.screen_h))
            self.screen.blit(frame, (0, 0))
            
            # Draw crop preview
            if SHOW_CROP_PREVIEW:
                self.draw_crop_preview()
            
            # Draw info panel
            self.draw_info_panel()
            
            # Draw help screen
            if self.show_help:
                self.draw_help_screen()
            
            # Draw status popups
            if self.clear_level:
                self.draw_status_popup("🎉 Level Cleared!", "You completed the level!")
            if self.game_over:
                self.draw_status_popup("💀 Game Over", "Try again and beat the level!")
            if self.paused and not self.show_help:
                self.draw_status_popup("⏸️ Game Paused", "Press P to resume")
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            self.frame_count += 1

if __name__ == "__main__":
    try:
        player = EnhancedMarioPlayer()
        player.run()
    except Exception as e:
        print(f"❌ Failed to start game: {e}")
        print("\n💡 Possible Fixes:")
        print("1. Install dependencies: pip install pygame gym-super-mario-bros nes-py numpy")
        print("2. Use Python 3.8-3.11 (compatible with gym-super-mario-bros)")
        pygame.quit()
        sys.exit()