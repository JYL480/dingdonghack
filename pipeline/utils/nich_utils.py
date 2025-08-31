#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nicholas Utils - Utility functions for Android navigation pipeline
Extracted from deployment.py and related files
Ensures perfect integration with nich_pipeline.py
"""

import base64
import datetime
import json
import os
import subprocess
import time
import re
import requests
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pydantic import SecretStr
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Set OpenAI API key for other imports
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", config.LLM_API_KEY if hasattr(config, 'LLM_API_KEY') else "")

# Now safe to import ui_test_agent
try:
    from ui_test_agent import ui_test_agent_node
except ImportError:
    # Fallback if ui_test_agent not available
    def ui_test_agent_node(state):
        return state

# Import click visualization functions
try:
    from tool.click_visualizer import draw_click_marker, save_action_visualization
except ImportError:
    # Fallback if click_visualizer not available
    def draw_click_marker(*args, **kwargs):
        return args[0] if args else None
    def save_action_visualization(*args, **kwargs):
        return {"clicked_image": None, "action_json": None}

# =====================================================
# LLM Configuration
# =====================================================

def get_model():
    """Get configured ChatOpenAI model"""
    # Set environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = "DeploymentExecution"
    
    return ChatOpenAI(
        openai_api_base=config.LLM_BASE_URL,
        openai_api_key=SecretStr(config.LLM_API_KEY),
        model_name=config.LLM_MODEL,
        request_timeout=config.LLM_REQUEST_TIMEOUT,
        max_retries=config.LLM_MAX_RETRIES,
        max_tokens=config.LLM_MAX_TOKEN,
    )

# =====================================================
# ADB Command Execution
# =====================================================

def find_adb_path():
    """Find ADB path dynamically"""
    import shutil
    import platform
    
    # First check if adb is in PATH
    adb_in_path = shutil.which("adb")
    if adb_in_path:
        return adb_in_path
    
    # Common ADB locations based on OS
    possible_paths = []
    
    if platform.system() == "Windows":
        # Windows common paths
        user_home = os.path.expanduser("~")
        possible_paths = [
            r"C:\platform-tools\adb.exe",
            r"C:\Android\platform-tools\adb.exe",
            r"C:\Android\Sdk\platform-tools\adb.exe",
            os.path.join(user_home, r"AppData\Local\Android\Sdk\platform-tools\adb.exe"),
            os.path.join(user_home, r"AppData\Local\Android\platform-tools\adb.exe"),
            r"C:\Program Files (x86)\Android\android-sdk\platform-tools\adb.exe",
            r"C:\Program Files\Android\android-sdk\platform-tools\adb.exe",
        ]
    elif platform.system() == "Darwin":  # macOS
        user_home = os.path.expanduser("~")
        possible_paths = [
            "/usr/local/bin/adb",
            os.path.join(user_home, "Library/Android/sdk/platform-tools/adb"),
            "/opt/homebrew/bin/adb",
            "/Applications/Android Studio.app/Contents/platform-tools/adb",
        ]
    else:  # Linux
        user_home = os.path.expanduser("~")
        possible_paths = [
            "/usr/bin/adb",
            "/usr/local/bin/adb",
            os.path.join(user_home, "Android/Sdk/platform-tools/adb"),
            "/opt/android-sdk/platform-tools/adb",
        ]
    
    # Check each possible path
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[INFO] Found ADB at: {path}")
            return path
    
    # If not found, show helpful message
    print("[WARNING] ADB not found. Please ensure Android SDK is installed.")
    print("[WARNING] Add platform-tools to PATH or set ADB_PATH environment variable")
    
    # Check for environment variable as fallback
    env_path = os.environ.get("ADB_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Return None to use 'adb' command as-is (might work if in PATH)
    return None

def execute_adb(adb_command):
    """Execute ADB commands with proper path quoting."""
    # Add full path to adb if not already present
    if adb_command.startswith("adb"):
        adb_path = find_adb_path()
        if adb_path:
            # IMPORTANT: Quote the path to handle spaces
            quoted_path = f'"{adb_path}"'
            adb_command = adb_command.replace("adb", quoted_path, 1)

    print(f"[ADB] Executing: {adb_command}") # Added for better debugging
    
    result = subprocess.run(
        adb_command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15 # Added timeout to prevent hangs
    )
    if result.returncode == 0:
        return result.stdout.strip()
    
    # Provide more helpful error messages
    print(f"[ADB ERROR] Command failed: {adb_command}")
    print(f"[ADB ERROR] Return Code: {result.returncode}")
    print(f"[ADB ERROR] Stderr: {result.stderr.strip()}")
    return "ERROR"

def list_all_devices() -> list:
    """List all connected ADB devices"""
    adb_command = "adb devices"
    device_list = []
    result = execute_adb(adb_command)
    if result != "ERROR":
        devices = result.split("\n")[1:]
        for d in devices:
            if d.strip() and '\t' in d:
                device_list.append(d.split('\t')[0])
    return device_list

def get_device_size(device: str = "emulator-5554") -> dict:
    """Get device screen size"""
    adb_command = f"adb -s {device} shell wm size"
    result = execute_adb(adb_command)
    if result != "ERROR":
        try:
            size_str = result.split(": ")[1]
            width, height = map(int, size_str.split("x"))
            return {"width": width, "height": height}
        except:
            pass
    return {"width": 1080, "height": 1920}  # Default size

# =====================================================
# Screenshot Functions (matching tool/screen_content.py)
# =====================================================

@tool
def take_screenshot(device: str = "emulator-5554", app_name: str = None, step: int = 0) -> str:
    """
    Take a screenshot of the device - matches tool/screen_content.py:take_screenshot
    """
    save_dir = "./log/screenshots"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if app_name is None:
        app_name = "unknown_app"

    # Create subdirectory for app
    app_dir = os.path.join(save_dir, app_name)
    if not os.path.exists(app_dir):
        os.makedirs(app_dir)

    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if step is not None:
        filename = f"{app_name}_step{step}_{timestamp}.png"
    else:
        filename = f"{app_name}_{timestamp}.png"

    screenshot_file = os.path.join(app_dir, filename)
    remote_file = f"/sdcard/{filename}"

    # ADB commands
    cap_command = f"adb -s {device} shell screencap -p {remote_file}"
    pull_command = f"adb -s {device} pull {remote_file} {screenshot_file}"
    delete_command = f"adb -s {device} shell rm {remote_file}"

    time.sleep(1)  # Small delay for stability
    
    try:
        if execute_adb(cap_command) != "ERROR":
            if execute_adb(pull_command) != "ERROR":
                execute_adb(delete_command)
                # Silent - no print here
                return screenshot_file
    except Exception as e:
        return f"Screenshot failed: {str(e)}"

    return "Screenshot failed. Please check device connection."

# =====================================================
# Screen Element Parsing (matching tool/screen_content.py:screen_element)
# =====================================================

@tool
def screen_element(image_path: str) -> Dict[str, Any]:
    """
    Parse screen elements using OmniParser model ONLY
    Calls the OmniParser API to extract UI elements from screenshots
    Raises exception if OmniParser fails - no fallback
    """
    
    api_url = f"{config.Omni_URI}/parse"
    
    # Check if screenshot exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Screenshot file does not exist: {image_path}")
    
    # Create processed images directory
    processed_dir = os.path.join(os.path.dirname(image_path), "processed_images")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    try:
        # Call OmniParser API
        print(f"[INFO] Calling OmniParser API at {api_url}")
        with open(image_path, "rb") as file:
            files = [("file", (os.path.basename(image_path), file, "image/png"))]
            response = requests.post(api_url, files=files, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"OmniParser API call failed with status {response.status_code}: {response.text}")
        
        # Parse response
        data = response.json()
        if data.get("status") != "success":
            raise Exception(f"OmniParser returned error status: {data}")
        
        # Get elements and annotated image
        elements = data.get("elements", [])
        annotated_image_base64 = data.get("annotated_image", "")
        
        # Save annotated image if provided
        labeled_image_path = os.path.join(processed_dir, f"labeled_{os.path.basename(image_path)}")
        if annotated_image_base64:
            if annotated_image_base64.startswith('data:'):
                base64_data = annotated_image_base64.split(',')[1] if ',' in annotated_image_base64 else annotated_image_base64
            else:
                base64_data = annotated_image_base64
            
            labeled_image_data = base64.b64decode(base64_data)
            with open(labeled_image_path, "wb") as f:
                f.write(labeled_image_data)
        else:
            # Copy original if no labeled version
            shutil.copy2(image_path, labeled_image_path)
        
        # Save elements JSON
        elements_json_path = os.path.join(
            processed_dir, 
            f"{os.path.splitext(os.path.basename(image_path))[0]}_elements.json"
        )
        with open(elements_json_path, "w", encoding="utf-8") as f:
            json.dump(elements, f, ensure_ascii=False, indent=2)
        
        print(f"[INFO] OmniParser successfully extracted {len(elements)} elements")
        
        return {
            "parsed_content_json_path": elements_json_path,
            "labeled_image_path": labeled_image_path,
            "elements": elements,
            "parser_used": "OmniParser"
        }
        
    except requests.exceptions.Timeout:
        raise Exception("OmniParser API request timed out after 30 seconds")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to OmniParser API at {api_url}")
    except Exception as e:
        raise Exception(f"OmniParser API error: {str(e)}")

def capture_and_parse_screen(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture and parse screen - matches deployment.py:153
    """
    try:
        # Use task folder if available
        app_name = state.get("task_folder", "deployment")
        
        # 1. Take screenshot using the tool
        screenshot_result = take_screenshot.invoke({
            "device": state["device"],
            "app_name": app_name,
            "step": state.get("current_step", 0)
        })
        
        # Handle tool result format
        if isinstance(screenshot_result, dict):
            screenshot_path = screenshot_result.get("screenshot_path", "")
        else:
            screenshot_path = screenshot_result
        
        if not screenshot_path or not os.path.exists(screenshot_path):
            print("[ERROR] Screenshot failed")
            return state
        
        # 2. Parse screen elements using the tool
        screen_result = screen_element.invoke({"image_path": screenshot_path})
        
        if "error" in screen_result:
            print(f"[ERROR] Screen element parsing failed: {screen_result['error']}")
            return state
        
        # 3. Update current page information
        if "current_page" not in state:
            state["current_page"] = {}
            
        state["current_page"]["screenshot"] = screenshot_path
        state["current_page"]["elements_json"] = screen_result.get("parsed_content_json_path", "")
        
        # 4. Load element data
        if screen_result.get("parsed_content_json_path") and os.path.exists(screen_result["parsed_content_json_path"]):
            with open(screen_result["parsed_content_json_path"], "r", encoding="utf-8") as f:
                state["current_page"]["elements_data"] = json.load(f)
        else:
            state["current_page"]["elements_data"] = []
        
        return state
        
    except Exception as e:
        print(f"[ERROR] Error capturing and parsing screen: {str(e)}")
        return state

# =====================================================
# Smart Screen Action (matching tool/screen_content.py:smart_screen_action)
# =====================================================

@tool
def smart_screen_action(
    device: str,
    action: str,
    element_id: int = None,
    json_path: str = None,
    input_str: str = None,
    x: int = None,
    y: int = None,
    direction: str = None,
    duration: int = 1000,
    screenshot_path: str = None,
    step: int = None
) -> str:
    """
    Smart screen action with element detection and click visualization
    """
    try:
        # Try to import the actual tool
        from tool.screen_content import smart_screen_action as actual_smart_action
        return actual_smart_action.invoke({
            "device": device,
            "action": action,
            "element_id": element_id,
            "json_path": json_path,
            "input_str": input_str,
            "x": x,
            "y": y,
            "direction": direction,
            "duration": duration
        })
    except ImportError:
        # Fallback implementation
        # If element_id provided, get coordinates from JSON
        if element_id is not None and json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    elements = json.load(f)
                if 0 <= element_id < len(elements):
                    element = elements[element_id]
                    device_size = get_device_size(device)
                    x, y = convert_element_to_click_coordinates(element, device_size)
                    print(f"   -> Using element {element_id} at ({x}, {y})")
            except Exception as e:
                print(f"[WARNING] Could not get element coordinates: {e}")
        
        # Save click visualization if we have coordinates and a screenshot
        if (x is not None and y is not None and screenshot_path and 
            action in ['tap', 'double_tap', 'long_press', 'text']):
            try:
                # Find the labeled image path
                labeled_path = screenshot_path
                processed_dir = os.path.join(os.path.dirname(screenshot_path), "processed_images")
                if os.path.exists(processed_dir):
                    base_name = os.path.basename(screenshot_path)
                    possible_labeled = os.path.join(processed_dir, f"labeled_{base_name}")
                    if os.path.exists(possible_labeled):
                        labeled_path = possible_labeled
                
                # Create click visualization
                action_info = {
                    "action": action,
                    "element_id": element_id,
                    "step": step if step is not None else 0,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "coordinates": {"x": x, "y": y}
                }
                
                if input_str:
                    action_info["input_text"] = input_str
                
                # Save visualization with click marker
                viz_result = save_action_visualization(
                    labeled_image_path=labeled_path,
                    click_position=(x, y),
                    action_info=action_info,
                    save_dir=os.path.dirname(screenshot_path)
                )
                
                if viz_result.get("clicked_image"):
                    print(f"   -> Saved click visualization: {viz_result['clicked_image']}")
                    
            except Exception as e:
                print(f"[DEBUG] Could not save click visualization: {e}")
        
        # Execute the action
        return screen_action.invoke({
            "device": device,
            "action": action,
            "x": x,
            "y": y,
            "input_str": input_str,
            "direction": direction,
            "dist": "medium" if direction else None
        })

# =====================================================
# Screen Action Execution (matching tool/screen_content.py:screen_action)
# =====================================================

@tool
def screen_action(
    device: str,
    action: str,
    x: int = None,
    y: int = None,
    input_str: str = None,
    direction: str = None,
    dist: str = "medium"
) -> str:
    """
    Execute screen actions - matches tool/screen_content.py:screen_action
    """
    try:
        # Try to import the actual tool
        from tool.screen_content import screen_action as actual_screen_action
        return actual_screen_action.invoke({
            "device": device,
            "action": action,
            "x": x,
            "y": y,
            "input_str": input_str,
            "direction": direction,
            "dist": dist
        })
    except ImportError:
        # Fallback implementation
        return execute_screen_action(device, action, x, y, input_str, direction, 1000)

def execute_screen_action(
    device: str,
    action: str,
    x: int = None,
    y: int = None,
    input_str: str = None,
    direction: str = None,
    duration: int = 1000
) -> str:
    """
    Execute screen actions - Enhanced with all Android atomic actions.
    This version uses correct integer keycodes for ADB commands.
    
    Supported actions:
    - tap, double_tap, long_press: Touch actions
    - text, clear_text: Text input actions  
    - swipe, scroll: Movement actions
    - back, home, recent_apps, app_switch: Navigation
    - volume_up, volume_down: Volume controls
    - power, screenshot: System actions
    - notification_panel, quick_settings: Status bar actions
    """
    # Initialize cmd to an empty string to avoid potential errors
    cmd = ""
    try:
        # Touch Actions
        if action == "tap" and x is not None and y is not None:
            cmd = f"adb -s {device} shell input tap {x} {y}"
        elif action == "double_tap" and x is not None and y is not None:
            # Execute two taps quickly
            cmd1 = f"adb -s {device} shell input tap {x} {y}"
            cmd2 = f"adb -s {device} shell input tap {x} {y}"
            execute_adb(cmd1)
            time.sleep(0.05)  # 50ms between taps
            cmd = cmd2
        elif action == "long_press" and x is not None and y is not None:
            cmd = f"adb -s {device} shell input swipe {x} {y} {x} {y} {duration}"
            
        # Text Actions
        elif action == "text" and input_str:
            # Tap the element first if coordinates are provided to ensure focus
            if x is not None and y is not None:
                execute_adb(f"adb -s {device} shell input tap {x} {y}")
                time.sleep(0.5)
            # Escape special characters for shell command
            escaped_text = input_str.replace("'", "'\\''").replace('"', '\\"')
            cmd = f"adb -s {device} shell input text '{escaped_text}'"
        elif action == "clear_text":
            # Select all (KEYCODE_MOVE_END + SHIFT) then delete (KEYCODE_DEL)
            # This is more reliable than sending many DEL events
            execute_adb(f"adb -s {device} shell input keyevent --longpress 123") # KEYCODE_MOVE_END
            execute_adb(f"adb -s {device} shell input keyevent 67") # KEYCODE_DEL
            return json.dumps({"status": "success", "message": "Text cleared"})
            
        # Movement Actions
        elif action == "swipe":
            if x is not None and y is not None:
                # Swipe from specified coordinates
                if direction == "up":
                    end_x, end_y = x, y - 500
                elif direction == "down":
                    end_x, end_y = x, y + 500
                elif direction == "left":
                    end_x, end_y = x - 300, y
                elif direction == "right":
                    end_x, end_y = x + 300, y
                else:
                    end_x, end_y = x, y - 500  # Default to swipe up
                cmd = f"adb -s {device} shell input swipe {x} {y} {end_x} {end_y} {duration}"
            else:
                # Full screen swipe if no coordinates are given
                device_size = get_device_size(device)
                center_x = device_size["width"] // 2
                center_y = device_size["height"] // 2
                if direction == "up":
                    cmd = f"adb -s {device} shell input swipe {center_x} {center_y + 300} {center_x} {center_y - 300} {duration}"
                elif direction == "down":
                    cmd = f"adb -s {device} shell input swipe {center_x} {center_y - 300} {center_x} {center_y + 300} {duration}"
                elif direction == "left":
                    cmd = f"adb -s {device} shell input swipe {center_x + 200} {center_y} {center_x - 200} {center_y} {duration}"
                elif direction == "right":
                    cmd = f"adb -s {device} shell input swipe {center_x - 200} {center_y} {center_x + 200} {center_y} {duration}"
                else:
                    cmd = f"adb -s {device} shell input swipe {center_x} {center_y + 300} {center_x} {center_y - 300} {duration}"
        elif action == "scroll":
            # A slower swipe for smoother scrolling
            device_size = get_device_size(device)
            center_x = device_size["width"] // 2
            center_y = device_size["height"] // 2
            scroll_duration = 1500
            if direction == "up":
                cmd = f"adb -s {device} shell input swipe {center_x} {center_y + 200} {center_x} {center_y - 200} {scroll_duration}"
            elif direction == "down":
                cmd = f"adb -s {device} shell input swipe {center_x} {center_y - 200} {center_x} {center_y + 200} {scroll_duration}"
            else:
                # Default to scroll up
                cmd = f"adb -s {device} shell input swipe {center_x} {center_y + 200} {center_x} {center_y - 200} {scroll_duration}"
                
        # --- CORRECTED NAVIGATION ACTIONS ---
        elif action == "back":
            cmd = f"adb -s {device} shell input keyevent KEYCODE_BACK"  # More reliable than numeric code
        elif action == "home":
            cmd = f"adb -s {device} shell input keyevent KEYCODE_HOME"  # More reliable than numeric code
        elif action == "recent_apps" or action == "recents" or action == "app_switch":
            cmd = f"adb -s {device} shell input keyevent 187" # KEYCODE_APP_SWITCH. [3]
            
        # --- CORRECTED VOLUME CONTROLS ---
        elif action == "volume_up":
            cmd = f"adb -s {device} shell input keyevent 24"  # KEYCODE_VOLUME_UP. [7, 21]
        elif action == "volume_down":
            cmd = f"adb -s {device} shell input keyevent 25"  # KEYCODE_VOLUME_DOWN. [7, 21]
        elif action == "mute":
            cmd = f"adb -s {device} shell input keyevent 164" # KEYCODE_VOLUME_MUTE
            
        # --- CORRECTED SYSTEM ACTIONS ---
        elif action == "power":
            cmd = f"adb -s {device} shell input keyevent 26"  # KEYCODE_POWER. [2, 8]
        elif action == "screenshot":
            cmd = f"adb -s {device} shell input keyevent 120" # KEYCODE_SYSRQ
        elif action == "lock":
            cmd = f"adb -s {device} shell input keyevent 223" # KEYCODE_SLEEP
        elif action == "wake":
            cmd = f"adb -s {device} shell input keyevent 224" # KEYCODE_WAKEUP
            
        # Status Bar Actions
        elif action == "notification_panel" or action == "notifications":
            device_size = get_device_size(device)
            cmd = f"adb -s {device} shell input swipe {device_size['width']//2} 0 {device_size['width']//2} {device_size['height']//2} 300"
        elif action == "quick_settings":
            device_size = get_device_size(device)
            cmd1 = f"adb -s {device} shell input swipe {device_size['width']//2} 0 {device_size['width']//2} {device_size['height']//2} 300"
            execute_adb(cmd1)
            time.sleep(0.3)
            cmd = cmd1  # Swipe a second time for quick settings
            
        # --- CORRECTED ADDITIONAL USEFUL ACTIONS ---
        elif action == "enter":
            cmd = f"adb -s {device} shell input keyevent 66"  # KEYCODE_ENTER. [4, 13]
        elif action == "tab":
            cmd = f"adb -s {device} shell input keyevent 61"  # KEYCODE_TAB
        elif action == "escape":
            cmd = f"adb -s {device} shell input keyevent 111" # KEYCODE_ESCAPE
        elif action == "menu":
            cmd = f"adb -s {device} shell input keyevent 82"  # KEYCODE_MENU. [1, 13]
        elif action == "search":
            cmd = f"adb -s {device} shell input keyevent 84"  # KEYCODE_SEARCH
            
        else:
            return json.dumps({"status": "error", "message": f"Invalid or unsupported action: {action}"})
        
        # Execute the generated command if it's not empty
        if cmd:
            result = execute_adb(cmd)
            if result != "ERROR":
                return json.dumps({"status": "success", "message": f"Action '{action}' executed"})
            else:
                return json.dumps({"status": "error", "message": f"Command execution failed for action '{action}'"})
        
        # This part will be reached by actions like 'clear_text' that don't set cmd
        return json.dumps({"status": "success", "message": f"Action '{action}' completed."})
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "message": str(e)})

# =====================================================
# Coordinate Conversion (matching tool/coordinate_converter.py)
# =====================================================

def convert_element_to_click_coordinates(element: Dict, device_size: Dict) -> Tuple[int, int]:
    """
    Convert element bbox to click coordinates - matches tool/coordinate_converter.py
    """
    try:
        # Try to import the actual converter
        from tool.coordinate_converter import convert_element_to_click_coordinates as actual_converter
        return actual_converter(element, device_size)
    except ImportError:
        # Fallback implementation
        bbox = element.get("bbox", [0.5, 0.5, 0.5, 0.5])
        
        # Calculate center point
        if len(bbox) >= 4:
            # bbox format: [x1, y1, x2, y2] as ratios
            center_x_ratio = (bbox[0] + bbox[2]) / 2
            center_y_ratio = (bbox[1] + bbox[3]) / 2
        else:
            center_x_ratio = 0.5
            center_y_ratio = 0.5
        
        # Convert to absolute coordinates
        center_x = int(center_x_ratio * device_size["width"])
        center_y = int(center_y_ratio * device_size["height"])
        
        return center_x, center_y

def get_element_bounds(element: Dict) -> Tuple[float, float, float, float]:
    """
    Get element bounds - matches tool/coordinate_converter.py
    """
    try:
        from tool.coordinate_converter import get_element_bounds as actual_get_bounds
        return actual_get_bounds(element)
    except ImportError:
        bbox = element.get("bbox", [0, 0, 1, 1])
        if len(bbox) >= 4:
            return bbox[0], bbox[1], bbox[2], bbox[3]
        return 0, 0, 1, 1

def get_element_center(element: Dict, device_size: Dict = None) -> Tuple[int, int]:
    """
    Get the center coordinates of an element from OmniParser data
    
    OmniParser elements have:
    - 'coordinates': [x, y] - the center point already calculated
    - 'bbox': [x1, y1, x2, y2] - bounding box corners in absolute pixels
    """
    # First check if coordinates are directly provided (OmniParser format)
    if 'coordinates' in element:
        coords = element['coordinates']
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return int(coords[0]), int(coords[1])
    
    # Check for bbox in OmniParser format [x1, y1, x2, y2] or dict format
    if 'bbox' in element:
        bbox = element['bbox']
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            # Calculate center from corners
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            return center_x, center_y
        elif isinstance(bbox, dict):
            # Dict format with x, y, width, height (OmniParser uses this)
            x = bbox.get('x', 0)
            y = bbox.get('y', 0)
            width = bbox.get('width', 100)
            height = bbox.get('height', 100)
            center_x = x + width // 2
            center_y = y + height // 2
            return center_x, center_y
    
    # Fallback to screen center
    if device_size:
        return device_size['width'] // 2, device_size['height'] // 2
    return 540, 1200  # Default Android screen center

def extrapolate_coordinate(instruction: str, elements_data: List[Dict], device_size: Dict) -> Optional[Tuple[int, int]]:
    """
    Calculate coordinates from AI's spatial instructions.
    The AI uses screen dimensions as reference for distance estimation.
    
    Supported formats:
    - "tap between element 3 and element 7"
    - "tap 150 pixels above element 5" (based on screen proportions)
    - "tap coordinate 500,800" (absolute position)
    - "tap 30% right of element 5" (percentage of element or screen)
    - "tap screen position 0.5,0.75" (fractional screen coordinates)
    
    Args:
        instruction: The vision model's decision string
        elements_data: List of detected elements with their positions
        device_size: Device screen dimensions (typically 1080x2400)
        
    Returns:
        Tuple of (x, y) coordinates or None if parsing fails
    """
    import re
    
    # Get screen dimensions for reference
    screen_width = device_size.get("width", 1080)
    screen_height = device_size.get("height", 2400)
    
    print(f"[COORDINATE] Screen size: {screen_width}x{screen_height}")
    
    # Debug: Show element structure for first few elements
    if elements_data and len(elements_data) > 0:
        sample_elem = elements_data[0]
        if 'bbox' in sample_elem:
            print(f"[COORDINATE] Element bbox format: {type(sample_elem['bbox'])}, value: {sample_elem.get('bbox', 'none')[:100] if isinstance(sample_elem.get('bbox'), str) else sample_elem.get('bbox')}")
        if 'coordinates' in sample_elem:
            print(f"[COORDINATE] Element has coordinates: {sample_elem['coordinates']}")
    
    # Edge-based positioning - be VERY flexible with patterns
    edge_patterns = [
        # Standard patterns
        r'(?:tap\s+)?(?:just\s+)?(?:to\s+the\s+)?(above|below|left|right)\s+(?:of\s+)?element\s*(\d+)',
        r'(?:tap\s+)?(?:just\s+)(above|below|left|right)\s+element\s*(\d+)',
        r'(?:tap\s+)?(?:immediately\s+)?(above|below|left|right)\s+element\s*(\d+)',
        r'(?:tap\s+)?(?:directly\s+)?(above|below|left|right)\s+element\s*(\d+)',
        r'(?:tap\s+)?(?:to\s+the\s+)(above|below|left|right)\s+(?:of\s+)?element\s*(\d+)',
        
        # More variations
        r'(?:tap\s+)?(?:near|next\s+to|beside)\s+element\s*(\d+)\s+(?:on\s+the\s+)?(above|below|left|right)',
        r'(?:tap\s+)?element\s*(\d+).*?(above|below|left|right)',
        r'(above|below|left|right).*?element\s*(\d+)',
        
        # Very loose pattern as fallback
        r'(\w+)\s+(?:of\s+|from\s+|to\s+)?element\s*(\d+)'
    ]
    
    for pattern in edge_patterns:
        match = re.search(pattern, instruction, re.IGNORECASE)
        if match and "pixel" not in instruction.lower() and "%" not in instruction:
            # Handle different group positions based on which pattern matched
            groups = match.groups()
            # Find the direction (above/below/left/right) and element ID
            direction = None
            elem_id = None
            for g in groups:
                if g and g.lower() in ['above', 'below', 'left', 'right']:
                    direction = g.lower()
                elif g and g.lower() in ['near', 'next', 'beside', 'close']:
                    # Default to right for "near/beside"
                    direction = 'right'
                elif g and g.isdigit():
                    elem_id = int(g)
            
            if direction is None or elem_id is None:
                continue
            
            if elem_id < len(elements_data):
                element = elements_data[elem_id]
                center_x, center_y = get_element_center(element, device_size)
                
                print(f"[COORDINATE] Edge positioning: '{direction}' of element {elem_id}")
                
                # Use bounding box edges for positioning
                if 'bbox' in element:
                    bbox = element['bbox']
                    x1, y1, x2, y2 = None, None, None, None
                    
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        # List format [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    elif isinstance(bbox, dict):
                        # Dict format {x, y, width, height}
                        x1 = bbox.get('x', 0)
                        y1 = bbox.get('y', 0)
                        x2 = x1 + bbox.get('width', 100)
                        y2 = y1 + bbox.get('height', 100)
                    
                    if x1 is not None:
                        # Position just outside the element's edge
                        padding = 10  # Small padding from edge
                        
                        result_x, result_y = center_x, center_y
                        
                        if direction == "above":
                            result_x, result_y = center_x, max(0, y1 - padding)
                        elif direction == "below":
                            result_x, result_y = center_x, min(screen_height, y2 + padding)
                        elif direction == "left":
                            result_x, result_y = max(0, x1 - padding), center_y
                        elif direction == "right":
                            result_x, result_y = min(screen_width, x2 + padding), center_y
                        
                        print(f"[COORDINATE] Positioned {direction} of element {elem_id} bbox ({x1},{y1},{x2},{y2}) -> ({result_x},{result_y})")
                        return result_x, result_y
                
                # Fallback if no bbox
                offset = 50  # Default small offset
                result_x, result_y = center_x, center_y
                
                if direction == "above":
                    result_x, result_y = center_x, max(0, center_y - offset)
                elif direction == "below":
                    result_x, result_y = center_x, min(screen_height, center_y + offset)
                elif direction == "left":
                    result_x, result_y = max(0, center_x - offset), center_y
                elif direction == "right":
                    result_x, result_y = min(screen_width, center_x + offset), center_y
                    
                print(f"[COORDINATE] Positioned {direction} of element {elem_id} (no bbox) -> ({result_x},{result_y})")
                return result_x, result_y
    
    # Between two elements - calculate midpoint (with safety check)
    if "between" in instruction.lower():
        match = re.search(r'between element (\d+) and element (\d+)', instruction, re.IGNORECASE)
        if match:
            elem1_id = int(match.group(1))
            elem2_id = int(match.group(2))
            
            if elem1_id < len(elements_data) and elem2_id < len(elements_data):
                x1, y1 = get_element_center(elements_data[elem1_id], device_size)
                x2, y2 = get_element_center(elements_data[elem2_id], device_size)
                
                # Safety check: Don't use "between" if elements are too far apart
                x_distance = abs(x2 - x1)
                y_distance = abs(y2 - y1)
                
                # If vertical distance is more than 1/3 of screen height, this is probably wrong
                if y_distance > screen_height / 3:
                    print(f"[COORDINATE] WARNING: Elements {elem1_id} and {elem2_id} are {y_distance}px apart vertically!")
                    print(f"[COORDINATE] This is likely a mistake - elements are on different parts of the page")
                    print(f"[COORDINATE] Falling back to element {elem1_id} position")
                    return x1, y1
                
                # If horizontal distance is more than 2/3 of screen width, also suspicious
                if x_distance > screen_width * 2 / 3:
                    print(f"[COORDINATE] WARNING: Elements {elem1_id} and {elem2_id} are {x_distance}px apart horizontally!")
                    print(f"[COORDINATE] Using element {elem1_id} position instead")
                    return x1, y1
                
                print(f"[COORDINATE] Calculating midpoint between element {elem1_id} ({x1},{y1}) and element {elem2_id} ({x2},{y2})")
                return (x1 + x2) // 2, (y1 + y2) // 2
    
    # AI-specified pixel distance from element
    if "pixel" in instruction.lower():
        # Pattern: "tap 150 pixels above element 5" or "150 pixels right of element 10"
        pattern = r'(\d+)\s*pixels?\s*(above|below|left|right|up|down)\s*(?:of|from)?\s*element\s*(\d+)'
        match = re.search(pattern, instruction, re.IGNORECASE)
        if match:
            distance = int(match.group(1))  # AI-determined distance
            direction = match.group(2).lower()
            elem_id = int(match.group(3))
            
            if elem_id < len(elements_data):
                element = elements_data[elem_id]
                x, y = get_element_center(element, device_size)
                
                # For better positioning, use element bounds when moving horizontally/vertically
                if 'bbox' in element:
                    bbox = element['bbox']
                    x1, y1, x2, y2 = None, None, None, None
                    
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    elif isinstance(bbox, dict):
                        # Dict format {x, y, width, height}
                        x1 = bbox.get('x', 0)
                        y1 = bbox.get('y', 0)
                        x2 = x1 + bbox.get('width', 100)
                        y2 = y1 + bbox.get('height', 100)
                    
                    if x1 is not None:
                        # Use edge of bounding box for more accurate positioning
                        if direction in ["above", "up"]:
                            # Start from top edge of element
                            base_y = y1
                            return x, max(0, base_y - distance)
                        elif direction in ["below", "down"]:
                            # Start from bottom edge of element
                            base_y = y2
                            return x, min(device_size.get("height", 2400), base_y + distance)
                        elif direction == "left":
                            # Start from left edge of element
                            base_x = x1
                            return max(0, base_x - distance), y
                        elif direction == "right":
                            # Start from right edge of element
                            base_x = x2
                            return min(device_size.get("width", 1080), base_x + distance), y
                
                # Fallback to center-based calculation
                print(f"[COORDINATE] Element {elem_id} center at ({x},{y}), moving {distance}px {direction}")
                
                if direction in ["above", "up"]:
                    return x, max(0, y - distance)
                elif direction in ["below", "down"]:
                    return x, min(device_size.get("height", 2400), y + distance)
                elif direction == "left":
                    return max(0, x - distance), y
                elif direction == "right":
                    return min(device_size.get("width", 1080), x + distance), y
    
    # Direct coordinates from AI - handle multiple formats
    if "coordinate" in instruction.lower():
        # Try format: "coordinate (540, 85)" or "coordinate 540, 85"
        patterns = [
            r'coordinate\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',  # With or without parentheses
            r'at\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?',          # "at (540, 85)"
            r'\((\d+)\s*,\s*(\d+)\)',                        # Just "(540, 85)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                print(f"[COORDINATE] Direct coordinate: ({x},{y})")
                return x, y
    
    # Percentage-based positioning
    if "%" in instruction:
        pattern = r'(\d+)%\s*(left|right|up|down|above|below)\s*(?:of|from)?\s*element\s*(\d+)'
        match = re.search(pattern, instruction, re.IGNORECASE)
        if match:
            percentage = int(match.group(1)) / 100
            direction = match.group(2).lower()
            elem_id = int(match.group(3))
            
            if elem_id < len(elements_data):
                elem = elements_data[elem_id]
                x, y = get_element_center(elem, device_size)
                
                # Get actual element dimensions from bbox
                width = 200  # Default
                height = 100  # Default
                
                if 'bbox' in elem:
                    bbox = elem['bbox']
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        # bbox format: [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        width = abs(x2 - x1)
                        height = abs(y2 - y1)
                    elif isinstance(bbox, dict):
                        # Legacy format
                        width = bbox.get('width', 200)
                        height = bbox.get('height', 100)
                
                print(f"[COORDINATE] Element {elem_id} at ({x},{y}), size {width}x{height}, moving {int(percentage*100)}% {direction}")
                
                if direction in ["right", "left"]:
                    # Use element width for horizontal movement
                    offset = int(width * percentage)
                    new_x = x + offset if direction == "right" else x - offset
                    return max(0, min(device_size.get("width", 1080), new_x)), y
                else:
                    # Use element height for vertical movement
                    offset = int(height * percentage)
                    new_y = y + offset if direction in ["down", "below"] else y - offset
                    return x, max(0, min(device_size.get("height", 2400), new_y))
    
    # Screen position patterns (top-right, bottom-left, center, etc.)
    screen_positions = {
        "top-left": (0.1, 0.1),
        "top-center": (0.5, 0.1),
        "top-right": (0.9, 0.1),
        "center-left": (0.1, 0.5),
        "center": (0.5, 0.5),
        "center-right": (0.9, 0.5),
        "bottom-left": (0.1, 0.9),
        "bottom-center": (0.5, 0.9),
        "bottom-right": (0.9, 0.9),
        "top": (0.5, 0.1),
        "bottom": (0.5, 0.9),
        "left": (0.1, 0.5),
        "right": (0.9, 0.5)
    }
    
    for position_name, (x_ratio, y_ratio) in screen_positions.items():
        if position_name in instruction.lower():
            x = int(screen_width * x_ratio)
            y = int(screen_height * y_ratio)
            print(f"[COORDINATE] Screen position '{position_name}': ({x},{y})")
            return x, y
    
    # Fractional screen coordinates (e.g., "screen position 0.5,0.75")
    if "screen position" in instruction.lower():
        match = re.search(r'screen position\s*([\d.]+)\s*,\s*([\d.]+)', instruction, re.IGNORECASE)
        if match:
            x_fraction = float(match.group(1))
            y_fraction = float(match.group(2))
            x = int(screen_width * x_fraction)
            y = int(screen_height * y_fraction)
            print(f"[COORDINATE] Screen fraction ({x_fraction},{y_fraction}): ({x},{y})")
            return x, y
    
    # Last resort - look for any "near element X" pattern
    if "near" in instruction.lower() or "beside" in instruction.lower() or "next to" in instruction.lower():
        match = re.search(r'element\s*(\d+)', instruction, re.IGNORECASE)
        if match:
            elem_id = int(match.group(1))
            if elem_id < len(elements_data):
                element = elements_data[elem_id]
                center_x, center_y = get_element_center(element, device_size)
                # Default to 100 pixels to the right for "near"
                print(f"[COORDINATE] 'Near' element {elem_id}, defaulting to 100px right")
                return min(screen_width-10, center_x + 100), center_y
    
    print(f"[COORDINATE] Could not parse instruction: {instruction}")
    return None

# =====================================================
# React Agent Navigation (matching deployment.py:670)
# =====================================================

def fallback_to_react(state: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Two-stage navigation: Vision analysis -> Action execution
    Stage 1: Vision model looks at labeled screenshot and decides what to click
    Stage 2: Action model gets coordinates from JSON and executes
    """
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 20)
    print(f"\n[STEP] Step {current_step + 1}/{max_steps}: Executing action...")
    
    task = state["task"]
    device = state["device"]
    
    # Small delay for screen stability
    time.sleep(0.5)
    
    # Initialize message history if first time
    if not state.get("messages"):
        state["messages"] = []
    
    # Check if we already have a screenshot from capture_screen_node
    if not state.get("current_page") or not state["current_page"].get("screenshot"):
        # Only capture if we don't have one already
        state = capture_and_parse_screen(state)
        if not state["current_page"]["screenshot"]:
            state["execution_status"] = "error"
            print("Unable to capture or parse screen")
            return state
    
    # Get screen info
    screenshot_path = state["current_page"]["screenshot"]
    elements_json_path = state["current_page"]["elements_json"]
    device_size = get_device_size(device)
    
    # Load elements JSON for Stage 2
    with open(elements_json_path, "r", encoding="utf-8") as f:
        elements_data = json.load(f)
    
    # Find labeled image
    labeled_image_path = screenshot_path
    processed_dir = os.path.join(os.path.dirname(screenshot_path), "processed_images")
    if os.path.exists(processed_dir):
        base_name = os.path.basename(screenshot_path)
        labeled_path = os.path.join(processed_dir, f"labeled_{base_name}")
        if os.path.exists(labeled_path):
            labeled_image_path = labeled_path
    
    # Load labeled image as base64 for vision model
    with open(labeled_image_path, "rb") as f:
        image_data = f.read()
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Build messages for AI
    messages = [
        SystemMessage(
            content=f"""Below is the current page information and user intent. Please analyze and execute the next action using smart_action_with_viz tool.

AVAILABLE ACTIONS:
• Touch: tap, double_tap, long_press (requires element_id or x,y)
• Text: text, clear_text (use element_id for text fields)
• Movement: swipe, scroll (can use direction: up/down/left/right)
• Navigation: back, home, recent_apps, app_switch
• System: volume_up, volume_down, mute, power, screenshot, lock, wake
• Status Bar: notification_panel, quick_settings
• Keyboard: enter, tab, escape, menu, search

IMPORTANT: Use the tool exactly as shown in these examples!

EXAMPLES OF CORRECT TOOL USAGE:
- To tap element 5: smart_action_with_viz(device="{device}", action="tap", element_id=5, json_path="{elements_json_path}")
- To type text: smart_action_with_viz(device="{device}", action="text", element_id=5, input_str="hello", json_path="{elements_json_path}")
- To swipe up: smart_action_with_viz(device="{device}", action="swipe", direction="up")
- To go back: smart_action_with_viz(device="{device}", action="back")
- To go home: smart_action_with_viz(device="{device}", action="home")
- To long press: smart_action_with_viz(device="{device}", action="long_press", element_id=10, json_path="{elements_json_path}")

The tool REQUIRES the device parameter to be included!

DECISION PROCESS:
1. Analyze task requirements and current screen
2. Identify relevant elements or required action
3. Choose most appropriate action
4. Execute with proper parameters

Required parameters:
- device: always "{device}"
- action: one of the available actions
- element_id + json_path: for element-based actions
- direction: for swipe/scroll
- input_str: for text action"""
        ),
        HumanMessage(
            content=f"The current device is: {device}, the device screen size is {device_size}. The user's current task intent is: {task}"
        ),
        HumanMessage(
            content="Below is the current page's parsed JSON data. Each element has an index (starting from 0) that you should use as element_id. PAY ATTENTION TO THE 'content' field to find the right element:\n"
            + elements_text
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "Below is the LABELED screenshot with numbered bounding boxes. Each number corresponds to an element ID in the JSON. Look at the numbers on the image to identify which element to click:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data_base64}"}}
            ]
        )
    ]
    
    # Add to state messages
    state["messages"].extend(messages)
    
    print(f"\n[DEBUG] Total messages in state: {len(state['messages'])}")
    print(f"[DEBUG] Passing last 4 messages to agent")
    
    # Log model input with expanded display
    print("\n" + "="*60)
    print("[MODEL INPUT]")
    print("="*60)
    for i, msg in enumerate(state["messages"][-4:], 1):
        if isinstance(msg, SystemMessage):
            print(f"\n{i}. SYSTEM PROMPT:")
            if len(msg.content) > 800:
                print(f"   {msg.content[:400]}")
                print(f"   ... [truncated {len(msg.content)-800} chars] ...")
                print(f"   {msg.content[-400:]}")
            else:
                print(f"   {msg.content}")
        elif isinstance(msg, HumanMessage):
            print(f"\n{i}. HUMAN MESSAGE:")
            if isinstance(msg.content, list):
                # This is an image + text message
                for item in msg.content:
                    if item.get("type") == "text":
                        print(f"   Text: {item.get('text', '')[:200]}...")
                    elif item.get("type") == "image_url":
                        print(f"   Image: [Screenshot attached]")
            else:
                if len(msg.content) > 600:
                    print(f"   {msg.content[:300]}")
                    print(f"   ... [truncated {len(msg.content)-600} chars] ...")
                    print(f"   {msg.content[-300:]}")
                else:
                    print(f"   {msg.content}")
    
    # Call action_agent for decision making and action execution
    print("\n[DEBUG] Calling React agent with model...")
    try:
        action_result = action_agent.invoke({"messages": state["messages"][-4:]})
        print("[DEBUG] React agent returned successfully")
    except Exception as e:
        print(f"[ERROR] React agent failed: {str(e)}")
        import traceback
        traceback.print_exc()
        state["execution_status"] = "error"
        return state
    
    # Parse results
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add AI reply to message history
        ai_message = final_messages[-1]
        state["messages"].append(ai_message)
        
        # Check if there were any tool calls
        if hasattr(ai_message, 'tool_calls') and ai_message.tool_calls:
            print(f"[DEBUG] Tool calls detected: {len(ai_message.tool_calls)}")
            for tc in ai_message.tool_calls:
                print(f"[DEBUG] Tool call: {tc}")
        
        # Extract recommended action from final_message
        recommended_action = ai_message.content.strip()
        
        # Log model output with expanded display
        print("\n" + "="*60)
        print("[MODEL OUTPUT]")
        print("="*60)
        if len(recommended_action) > 1000:
            # Show first 400 and last 400 characters for very long responses
            print(recommended_action[:400])
            print(f"\n... [truncated {len(recommended_action)-800} chars] ...\n")
            print(recommended_action[-400:])
        else:
            print(recommended_action)
        print("="*60)
        
        # Update execution status
        state["current_step"] += 1
        state["history"].append({
            "step": state["current_step"],
            "screenshot": screenshot_path,
            "elements_json": elements_json_path,
            "action": "react_mode",
            "recommended_action": recommended_action,
            "status": "success"
        })
        
        state["execution_status"] = "success"
        
        # Concise action feedback - Enhanced for all actions
        action_summary = ""
        action_lower = recommended_action.lower()
        
        # Touch actions
        if "tap" in action_lower or "click" in action_lower:
            elem_match = re.search(r'element[_\s]+(\d+)', action_lower)
            if "double" in action_lower:
                action_summary = f"Double-tapped element {elem_match.group(1)}" if elem_match else "Double-tapped"
            elif "long" in action_lower:
                action_summary = f"Long-pressed element {elem_match.group(1)}" if elem_match else "Long-pressed"
            else:
                action_summary = f"Tapped element {elem_match.group(1)}" if elem_match else "Tapped"
                
        # Text actions
        elif "text" in action_lower or "type" in action_lower:
            if "clear" in action_lower:
                action_summary = "Cleared text"
            else:
                text_match = re.search(r'"([^"]+)"', recommended_action)
                action_summary = f"Typed: \"{text_match.group(1)}\"" if text_match else "Typed text"
                
        # Movement actions
        elif "swipe" in action_lower:
            dir_match = re.search(r'direction["\s:=]+(\w+)', action_lower)
            action_summary = f"Swiped {dir_match.group(1)}" if dir_match else "Swiped"
        elif "scroll" in action_lower:
            dir_match = re.search(r'direction["\s:=]+(\w+)', action_lower)
            action_summary = f"Scrolled {dir_match.group(1)}" if dir_match else "Scrolled"
            
        # Navigation actions
        elif "back" in action_lower:
            action_summary = "Pressed back"
        elif "home" in action_lower:
            action_summary = "Went to home screen"
        elif "recent" in action_lower or "app_switch" in action_lower:
            action_summary = "Opened recent apps"
            
        # System actions
        elif "volume" in action_lower:
            if "up" in action_lower:
                action_summary = "Volume up"
            elif "down" in action_lower:
                action_summary = "Volume down"
            else:
                action_summary = "Muted"
        elif "power" in action_lower:
            action_summary = "Pressed power button"
        elif "screenshot" in action_lower:
            action_summary = "Took screenshot"
        elif "lock" in action_lower:
            action_summary = "Locked device"
        elif "wake" in action_lower:
            action_summary = "Woke device"
            
        # Status bar actions
        elif "notification" in action_lower:
            action_summary = "Opened notifications"
        elif "quick_settings" in action_lower:
            action_summary = "Opened quick settings"
            
        # Keyboard actions
        elif "enter" in action_lower:
            action_summary = "Pressed enter"
        elif "tab" in action_lower:
            action_summary = "Pressed tab"
        elif "escape" in action_lower:
            action_summary = "Pressed escape"
        elif "menu" in action_lower:
            action_summary = "Opened menu"
        elif "search" in action_lower:
            action_summary = "Opened search"
        else:
            action_summary = "Action completed"
        
        print(f"\n[ACTION EXECUTED] {action_summary}")
    else:
        error_msg = "React mode execution failed: No messages returned"
        print(f"[ERROR] {error_msg}")
        
        # Update execution status
        state["history"].append({
            "step": state["current_step"],
            "screenshot": screenshot_path,
            "elements_json": elements_json_path,
            "action": "react_mode",
            "status": "error",
            "error": error_msg
        })
        
        state["execution_status"] = "error"
    
    return state

# =====================================================
# Task Completion Check (matching deployment.py:2232)
# =====================================================

def check_task_completion(state: Dict[str, Any], model) -> Dict[str, Any]:
    """
    Check if task is completed using GPT-4 vision - matches deployment.py:2232
    """
    # Check max steps
    max_steps = state.get("max_steps", 20)
    current_step = state.get("current_step", 0)
    
    if current_step >= max_steps:
        print(f"[WARNING] Reached maximum steps ({max_steps})")
        state["completed"] = True
        state["execution_status"] = "max_steps_reached"
        return state
    
    # Don't check too early
    if current_step < 3:
        print(f"[INFO] Step {current_step}: Continuing task execution...")
        state["completed"] = False
        return state
    
    print("[CHECK] Evaluating task completion...")
    
    task = state.get("task", "")
    
    # Step 1: Generate task completion criteria
    completion_messages = [
        SystemMessage(
            content="You are an assistant that will help analyze task completion criteria. Please carefully read the following user task:"
        ),
        HumanMessage(
            content=f"The user's task is: {task}\nPlease describe clear, checkable task completion criteria. For example: 'When certain elements or states appear on the page, it indicates the task is complete.'"
        )
    ]
    
    completion_response = model.invoke(completion_messages)
    completion_criteria = completion_response.content
    
    # Collect recent screenshots
    recent_screenshots = []
    if "history" in state:
        for step in state["history"][-3:]:
            if "screenshot" in step and step["screenshot"]:
                recent_screenshots.append(step["screenshot"])
    
    if not recent_screenshots and state.get("current_page", {}).get("screenshot"):
        recent_screenshots.append(state["current_page"]["screenshot"])
    
    if not recent_screenshots:
        print("[WARNING] No screenshots available")
        state["completed"] = False
        return state
    
    # Build image messages
    image_messages = []
    for idx, img_path in enumerate(recent_screenshots, start=1):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"Here is data for screenshot {idx}:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                    ]
                )
            )
    
    # Step 2: Determine if task is complete
    judgement_messages = [
        SystemMessage(
            content="You are a page assessment assistant that will determine if a task is complete based on completion criteria and current page screenshots. Be strict - only say 'yes' if the ENTIRE task is fully completed."
        ),
        HumanMessage(
            content=f"Original task: {task}\n"
            f"Completion criteria: {completion_criteria}\n\n"
            f"Based on the following screenshots, determine if the ENTIRE task is complete.\n"
            f"- If the task is partially done but not finished, respond 'no'\n"
            f"- Only respond 'yes' if the full goal has been achieved\n"
            f"- For example, if the task is 'search for blue cheese', just opening the search bar is NOT complete - the search must be performed\n"
            f"Respond with ONLY 'yes' or 'no'."
        )
    ]
    
    # Combine all messages
    all_messages = judgement_messages + image_messages
    
    # Log completion check input
    print("\n" + "="*60)
    print("[COMPLETION CHECK]")
    print("="*60)
    print(f"Task: {task}")
    print(f"\nCompletion Criteria Generated:")
    if len(completion_criteria) > 500:
        print(f"{completion_criteria[:250]}...")
        print(f"[truncated {len(completion_criteria)-500} chars]")
        print(f"...{completion_criteria[-250:]}")
    else:
        print(completion_criteria)
    print(f"\nScreenshots being analyzed: {len(image_messages)}")
    
    # Call LLM for judgment
    judgement_response = model.invoke(all_messages)
    judgement_answer = judgement_response.content.strip()
    
    # Log completion check output
    print(f"\nModel Decision: {judgement_answer.upper()}")
    print("="*60)
    
    # Update task completion status
    if "yes" in judgement_answer.lower() or "complete" in judgement_answer.lower():
        state["completed"] = True
        state["execution_status"] = "completed"
        print(f"[SUCCESS] Task completed")
    else:
        state["completed"] = False
        print(f"[INFO] Task not completed, continuing...")
    
    # Add to history
    if "history" not in state:
        state["history"] = []
    
    state["history"].append({
        "step": state["current_step"],
        "action": "task_completion_check",
        "completion_criteria": completion_criteria,
        "judgement": judgement_answer,
        "status": "success",
        "completed": state["completed"]
    })
    
    return state

# =====================================================
# Folder Management
# =====================================================

def create_task_folder(task: str) -> str:
    """Create a folder for task screenshots"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_task = re.sub(r'[^\w\s-]', '', task)[:50].strip().replace(' ', '_')
    folder = f"run_{timestamp}_{clean_task}"
    return folder

# =====================================================
# State Creation (matching data/State.py:DeploymentState)
# =====================================================

def create_deployment_state(
    task: str,
    device: str,
    max_retries: int = 3,
    max_steps: int = 20,
    task_folder: str = None
) -> Dict[str, Any]:
    """Create initial deployment state - matches data/State.py"""
    if not task_folder:
        task_folder = create_task_folder(task)
    
    return {
        # Task related
        "task": task,
        "completed": False,
        "current_step": 0,
        "total_steps": 0,
        "execution_status": "ready",
        "retry_count": 0,
        "max_retries": max_retries,
        "max_steps": max_steps,
        
        # Device related
        "device": device,
        "task_folder": task_folder,
        
        # Page information
        "current_page": {
            "screenshot": None,
            "elements_json": None,
            "elements_data": []
        },
        
        # Enhanced context fields for vision system
        "screen_context": {
            "type": "home",  # home/app/settings/browser/unknown
            "app_name": None,  # Chrome/Settings/Gmail etc
            "previous_screen": None,
            "previous_app": None
        },
        
        # Action tracking for enhanced vision
        "action_history": [],  # Detailed action history with reasoning
        
        # Task progress tracking
        "task_progress": {
            "original_task": task,
            "current_objective": "Start task",
            "stuck_counter": 0
        },
        
        # Execution related
        "current_element": None,
        "current_action": None,
        "matched_elements": [],
        "associated_shortcuts": [],
        "execution_template": None,
        
        # Records and messages
        "history": [],
        "messages": [],
        
        # Flow control
        "should_fallback": False,
        "should_execute_shortcut": False
    }

# =====================================================
# Main Execution Functions (from deployment.py)
# =====================================================

def execute_deployment_task(device: str, task_description: str) -> Dict[str, Any]:
    """
    Main entry point from demo.py:1189
    This wraps run_task for compatibility
    """
    return run_task(task_description, device)

def run_task(task: str, device: str = "emulator-5554") -> Dict[str, Any]:
    """
    Execute a task - matches deployment.py:1197
    Main entry point matching deployment.py behavior
    """
    # Check for multiple tasks
    if ';' in task or '\n' in task:
        tasks = [t.strip() for t in task.replace('\n', ';').split(';') if t.strip()]
        if len(tasks) > 1:
            print(f"[INFO] Detected {len(tasks)} tasks, running in sequence...")
            return run_multiple_tasks(tasks, device)
    
    # Single task execution
    return run_task_with_folder(task, device)

def run_task_with_folder(task: str, device: str = "emulator-5554", folder: str = None) -> Dict[str, Any]:
    """
    Execute a single task with specific folder - matches deployment.py:1137
    """
    print(f"[START] Starting task execution: {task}")
    
    try:
        # Create folder if not provided
        if not folder:
            folder = create_task_folder(task)
        
        print(f"[FOLDER] Saving screenshots to: ./log/screenshots/{folder}/")
        
        # Initialize state
        state = create_deployment_state(
            task=task,
            device=device,
            max_retries=3,
            max_steps=20,
            task_folder=folder
        )
        
        # Build and execute workflow
        from langgraph.graph import StateGraph, END
        workflow = build_workflow()
        app = workflow.compile()
        result = app.invoke(state)
        
        if "task_folder" not in result:
            result["task_folder"] = folder
        
        # Display final status
        if result["execution_status"] == "success" or result["execution_status"] == "completed":
            print(f"[SUCCESS] Task completed successfully in {result.get('current_step', 0)} steps")
        elif result["execution_status"] == "max_steps_reached":
            print(f"[WARNING] Task stopped after reaching max steps ({state['max_steps']})")
        else:
            print(f"[ERROR] Task failed: {result.get('execution_status', 'unknown')}")
        
        return {
            "status": result["execution_status"],
            "message": "Task execution completed",
            "steps_completed": result.get("current_step", 0),
            "folder": folder
        }
        
    except Exception as e:
        print(f"[ERROR] Error executing task: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Error executing task: {str(e)}",
            "error": str(e)
        }

def run_multiple_tasks(tasks: list, device: str = "emulator-5554") -> Dict[str, Any]:
    """Execute multiple tasks in sequence"""
    import datetime
    
    # Create master folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_folder = f"multi_run_{timestamp}"
    
    print(f"[FOLDER] Master folder: ./log/screenshots/{master_folder}/")
    print(f"[TASKS] Executing {len(tasks)} tasks in sequence")
    
    results = []
    for idx, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"[TASK] Task {idx}/{len(tasks)}: {task}")
        print(f"{'='*60}")
        
        # Create subfolder
        clean_task = re.sub(r'[^\w\s-]', '', task)[:30].strip().replace(' ', '_')
        task_subfolder = f"{master_folder}/task_{idx}_{clean_task}"
        
        # Run task
        result = run_task_with_folder(task, device, task_subfolder)
        results.append({
            "task": task,
            "result": result,
            "folder": task_subfolder
        })
        
        # Pause between tasks
        if idx < len(tasks):
            print(f"\n[PAUSE] Pausing 2 seconds before next task...")
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Completed all {len(tasks)} tasks")
    print(f"[FOLDER] All results saved in: ./log/screenshots/{master_folder}/")
    
    return {
        "total_tasks": len(tasks),
        "results": results,
        "master_folder": master_folder
    }

def build_workflow():
    """
    Build workflow state graph - matches deployment.py:2202
    Creates StateGraph with capture_screen → fallback → check_completion
    """
    from langgraph.graph import StateGraph, END
    from typing_extensions import TypedDict
    
    # Import or define DeploymentState
    class DeploymentState(TypedDict):
        """State machine for deployment execution"""
        task: str
        completed: bool
        current_step: int
        total_steps: int
        execution_status: str
        retry_count: int
        max_retries: int
        max_steps: int
        device: str
        task_folder: str
        current_page: Dict
        matched_elements: List[Dict]
        messages: list
        history: List[Dict]
        should_fallback: bool
        should_execute_shortcut: bool
    
    def capture_screen_node(state: DeploymentState) -> DeploymentState:
        """Capture and parse screen - deployment.py:1865"""
        task_folder = state.get("task_folder")
        state = capture_and_parse_screen(state)
        
        if task_folder:
            state["task_folder"] = task_folder
        
        if not state["current_page"]["screenshot"]:
            state["should_fallback"] = True
            print("[ERROR] Unable to capture screen")
        
        return state
    
    def fallback_node(state: DeploymentState) -> DeploymentState:
        """Execute two-stage navigation"""
        from .nich_utils_twostage import two_stage_navigation
        model = get_model()
        state = two_stage_navigation(state, model)
        state["completed"] = False  # Let check_completion decide
        return state
    
    def check_completion_node(state: DeploymentState) -> DeploymentState:
        """Check if task is completed - deployment.py:2232"""
        model = get_model()
        state = check_task_completion(state, model)
        return state
    
    def is_task_completed(state: DeploymentState) -> str:
        """Check if task is completed for routing"""
        if state.get("completed", False):
            return "ui_test_agent"
        return "continue"
    
    def is_ui_test_completed_node(state: DeploymentState) -> str:
        """Check if UI test is completed for routing"""
        if state.get("execution_status") in ["success"]:
            return "end"
        return "continue"
    
    # Build workflow - deployment.py:2207-2229
    workflow = StateGraph(DeploymentState)
    
    # Add nodes
    workflow.add_node("capture_screen", capture_screen_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("check_completion", check_completion_node)
    workflow.add_node("ui_test_agent", ui_test_agent_node)
    workflow.add_node("is_ui_test_completed", is_ui_test_completed_node)
    
    # Set flow: capture_screen → fallback → check_completion
    workflow.set_entry_point("capture_screen")
    workflow.add_edge("capture_screen", "fallback")
    workflow.add_edge("fallback", "check_completion")
    workflow.add_edge("check_completion", "ui_test_agent")

    
    # Conditional routing
    workflow.add_conditional_edges(
        "check_completion",
        is_task_completed,
        {"end": "ui_test_agent", "continue": "capture_screen"}
    )

    workflow.add_conditional_edges(
        "check_completion",
        is_task_completed,
        {"end": END, "continue": "ui_test_agent"}
    )

    workflow.add_edge("ui_test_agent", "is_ui_test_completed")
    
    return workflow

# =====================================================
# Visual Ruler and Distance Annotation
# =====================================================

def draw_ruler_on_screenshot(image_path: str, element_ids: List[int], elements_data: List[Dict], 
                            target_description: str = None, save_path: str = None) -> str:
    """
    Draw visual ruler lines and pixel measurements on screenshot to help AI with positioning
    
    Args:
        image_path: Path to the screenshot
        element_ids: List of reference elements to draw rulers from (or single int)
        elements_data: List of all detected elements
        target_description: Optional description of what we're trying to tap
        save_path: Where to save the annotated image
        
    Returns:
        Path to the annotated image
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import os
        
        # Load image
        img = Image.open(image_path)
        # Create RGBA image for transparency
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = img.size
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            small_font = ImageFont.truetype("arial.ttf", 14)
            large_font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            small_font = font
            large_font = font
        
        # Handle single element or list of elements
        if isinstance(element_ids, int):
            element_ids = [element_ids]
        
        # Colors for different elements
        colors = ["blue", "purple", "teal", "brown", "pink"]
        
        # Draw rulers for each reference element
        for idx, element_id in enumerate(element_ids):
            if element_id >= len(elements_data):
                continue
                
            ref_element = elements_data[element_id]
            ref_x, ref_y = get_element_center(ref_element, {"width": width, "height": height})
            color = colors[idx % len(colors)]
            
            # Get bounding box if available
            ref_bbox = None
            if 'bbox' in ref_element:
                bbox = ref_element['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    ref_bbox = bbox  # [x1, y1, x2, y2]
            
            # Draw reference element highlight
            if ref_bbox:
                x1, y1, x2, y2 = ref_bbox
                # Draw bounding box with thick line
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                # Mark corners
                corner_size = 6
                for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    draw.ellipse([cx-corner_size, cy-corner_size, cx+corner_size, cy+corner_size], 
                               fill=color, outline=color)
                # Add element label with background
                label = f"Element {element_id}"
                bbox_label = draw.textbbox((x1, y1-30), label, font=font)
                draw.rectangle(bbox_label, fill="white", outline=color, width=2)
                draw.text((x1, y1-30), label, fill=color, font=font)
            else:
                # Just draw center point
                draw.ellipse([ref_x-8, ref_y-8, ref_x+8, ref_y+8], fill=color, outline=color, width=2)
            
            # Draw ruler lines from element edges (if bbox available) or center
            if ref_bbox:
                x1, y1, x2, y2 = ref_bbox
                
                # RIGHT ruler - horizontal line from right edge
                ruler_y = (y1 + y2) // 2
                draw.line([(x2, ruler_y), (min(width-10, x2+300), ruler_y)], fill="red", width=2)
                
                # Draw clear distance markers with backgrounds
                for dist in [50, 100, 150, 200, 250]:
                    marker_x = x2 + dist
                    if marker_x < width - 20:
                        # Draw marker line
                        draw.line([(marker_x, ruler_y-10), (marker_x, ruler_y+10)], fill="red", width=2)
                        # Distance label with white background
                        label_text = f"{dist}px RIGHT"
                        bbox_text = draw.textbbox((marker_x-30, ruler_y+15), label_text, font=small_font)
                        draw.rectangle(bbox_text, fill="white", outline="red")
                        draw.text((marker_x-30, ruler_y+15), label_text, fill="red", font=small_font)
                
                # LEFT ruler - horizontal line from left edge  
                draw.line([(max(10, x1-300), ruler_y), (x1, ruler_y)], fill="magenta", width=2)
                for dist in [50, 100, 150, 200, 250]:
                    marker_x = x1 - dist
                    if marker_x > 20:
                        draw.line([(marker_x, ruler_y-10), (marker_x, ruler_y+10)], fill="magenta", width=2)
                        label_text = f"{dist}px LEFT"
                        bbox_text = draw.textbbox((marker_x-30, ruler_y+15), label_text, font=small_font)
                        draw.rectangle(bbox_text, fill="white", outline="magenta")
                        draw.text((marker_x-30, ruler_y+15), label_text, fill="magenta", font=small_font)
                
                # BELOW ruler - vertical line from bottom edge
                ruler_x = (x1 + x2) // 2
                draw.line([(ruler_x, y2), (ruler_x, min(height-10, y2+300))], fill="green", width=2)
                
                for dist in [50, 100, 150, 200, 250]:
                    marker_y = y2 + dist
                    if marker_y < height - 20:
                        draw.line([(ruler_x-10, marker_y), (ruler_x+10, marker_y)], fill="green", width=2)
                        label_text = f"{dist}px BELOW"
                        bbox_text = draw.textbbox((ruler_x+15, marker_y-8), label_text, font=small_font)
                        draw.rectangle(bbox_text, fill="white", outline="green")
                        draw.text((ruler_x+15, marker_y-8), label_text, fill="green", font=small_font)
                
                # ABOVE ruler - vertical line from top edge
                draw.line([(ruler_x, max(10, y1-300)), (ruler_x, y1)], fill="orange", width=2)
                
                for dist in [50, 100, 150, 200, 250]:
                    marker_y = y1 - dist
                    if marker_y > 20:
                        draw.line([(ruler_x-10, marker_y), (ruler_x+10, marker_y)], fill="orange", width=2)
                        label_text = f"{dist}px ABOVE"
                        bbox_text = draw.textbbox((ruler_x+15, marker_y-8), label_text, font=small_font)
                        draw.rectangle(bbox_text, fill="white", outline="orange")
                        draw.text((ruler_x+15, marker_y-8), label_text, fill="orange", font=small_font)
            
        # Add semi-transparent grid overlay (every 100 pixels)
        for x in range(0, width, 100):
            draw.line([(x, 0), (x, height)], fill=(200, 200, 200, 30), width=1)
        for y in range(0, height, 100):
            draw.line([(0, y), (width, y)], fill=(200, 200, 200, 30), width=1)
        
        # Add instruction text with prominent background
        if target_description:
            instruction_text = f"Target: {target_description}"
            bbox_inst = draw.textbbox((10, 10), instruction_text, font=font)
            draw.rectangle([(bbox_inst[0]-5, bbox_inst[1]-5), (bbox_inst[2]+5, bbox_inst[3]+5)], 
                         fill="yellow", outline="black", width=2)
            draw.text((10, 10), instruction_text, fill="black", font=font)
        
        # Add ruler legend in corner
        legend_text = "RULERS:\n→ RED = RIGHT\n← MAGENTA = LEFT\n↓ GREEN = BELOW\n↑ ORANGE = ABOVE"
        bbox_legend = draw.textbbox((10, height-120), legend_text, font=small_font)
        draw.rectangle([(bbox_legend[0]-5, bbox_legend[1]-5), (bbox_legend[2]+5, bbox_legend[3]+5)], 
                      fill="white", outline="black", width=2)
        draw.text((10, height-120), legend_text, fill="black", font=small_font)
        
        # Composite the overlay onto the original image
        img = Image.alpha_composite(img.convert('RGBA'), overlay)
        
        # Save annotated image
        if not save_path:
            base_name = os.path.splitext(image_path)[0]
            save_path = f"{base_name}_with_ruler.png"
        
        img.save(save_path)
        print(f"[RULER] Saved annotated screenshot with ruler: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[RULER] Error drawing ruler: {e}")
        return image_path  # Return original if annotation fails

# =====================================================
# UI Test Integration for Supervisor Pipeline
# =====================================================

def run_ui_test_for_subtask(subtask: Dict, screenshot_path: str, device: str = "emulator-5554") -> Dict[str, Any]:
    """
    Run UI test agent for a specific subtask that requires testing
    
    Args:
        subtask: The subtask dictionary with requires_test flag
        screenshot_path: Path to current screenshot
        device: ADB device identifier
        
    Returns:
        Test result dictionary
    """
    if not subtask.get("requires_test", False):
        return {"tested": False, "reason": "Subtask doesn't require testing"}
    
    try:
        # Create state for UI test agent
        test_state = {
            "task": f"Test: {subtask.get('task', '')}",
            "screenshot_path": screenshot_path,
            "device": device,
            "completed": False
        }
        
        # Run UI test agent
        result_state = ui_test_agent_node(test_state)
        
        # Extract test results - let supervisor interpret success/failure
        return {
            "tested": True,
            "success": result_state.get("execution_status") == "success",
            "result": result_state.get("agent_response", ""),
            "messages": result_state.get("messages", [])
        }
        
    except Exception as e:
        print(f"[UI TEST] Error running test: {e}")
        return {
            "tested": False,
            "error": str(e)
        }

# =====================================================
# Additional utilities for visualization if needed
# =====================================================

def draw_click_marker(image_path: str, click_pos: Tuple[int, int], save_path: str = None):
    """Draw click marker on image - matches tool/click_visualizer.py"""
    try:
        from tool.click_visualizer import draw_click_marker as actual_draw
        return actual_draw(image_path, click_pos, save_path)
    except ImportError:
        # Fallback - just return the original image
        return image_path

def save_action_visualization(
    labeled_image_path: str,
    click_coordinates: Tuple[int, int],
    action_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Save action visualization - matches tool/click_visualizer.py"""
    try:
        from tool.click_visualizer import save_action_visualization as actual_save
        result = actual_save(labeled_image_path, click_coordinates, action_info)
        # Only print the final saved image path
        if "clicked_image" in result and os.path.exists(result["clicked_image"]):
            print(f"[ACTION] Click marker saved: {result['clicked_image']}")
        return result
    except ImportError:
        # Fallback - return empty result
        return {"clicked_image": labeled_image_path}