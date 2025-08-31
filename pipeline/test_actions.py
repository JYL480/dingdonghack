"""
Test all available ADB actions to verify they work
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.nich_utils import execute_adb, screen_action

def test_all_actions(device="emulator-5554"):
    """Test all available screen actions"""
    
    print("Testing ADB Actions on device:", device)
    print("="*50)
    
    # Test basic navigation
    actions = [
        ("home", "Going to home screen"),
        ("back", "Pressing back button"),
        ("recent_apps", "Opening recent apps"),
    ]
    
    for action, description in actions:
        print(f"\nTesting: {description}")
        result = screen_action(device=device, action=action)
        print(f"Result: {result}")
        time.sleep(2)
    
    # Test swipe actions
    swipe_actions = [
        ("swipe", "up", "Swiping up"),
        ("swipe", "down", "Swiping down"),
        ("swipe", "left", "Swiping left"),
        ("swipe", "right", "Swiping right"),
    ]
    
    for action, direction, description in swipe_actions:
        print(f"\nTesting: {description}")
        result = screen_action(device=device, action=action, direction=direction)
        print(f"Result: {result}")
        time.sleep(2)
    
    # Test tap at center
    print("\nTesting: Tap at center of screen")
    result = screen_action(device=device, action="tap", x=540, y=1200)
    print(f"Result: {result}")
    
    print("\n" + "="*50)
    print("All tests completed!")
    
    # List all available actions
    print("\nAvailable actions in screen_action:")
    print("- tap (with x, y coordinates)")
    print("- double_tap")
    print("- long_press")
    print("- swipe (with direction: up/down/left/right)")
    print("- scroll (with direction)")
    print("- text (with input_str)")
    print("- clear_text")
    print("- back")
    print("- home")
    print("- recent_apps")
    print("- app_switch")
    print("- volume_up/volume_down/mute")
    print("- power")
    print("- screenshot")
    print("- lock/unlock/wake")
    print("- notification_panel")
    print("- quick_settings")

if __name__ == "__main__":
    test_all_actions()