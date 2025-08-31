"""
Simple two-level logging system for the pipeline
"""

import os

class Logger:
    """Simple logger with debug and normal modes"""
    
    def __init__(self, debug_mode=False):
        self.debug = debug_mode
        
    def set_debug(self, debug: bool):
        """Set debug mode"""
        self.debug = debug
        
    def log_debug(self, message: str):
        """Log debug messages (only shown in debug mode)"""
        if self.debug:
            print(f"[DEBUG] {message}")
            
    def log_info(self, message: str):
        """Log info messages (always shown)"""
        print(message)
        
    def log_progress(self, current: int, total: int, task: str = ""):
        """Log progress (always shown)"""
        percent = (current / total * 100) if total > 0 else 0
        print(f"Progress: {current}/{total} ({percent:.0f}%) - {task}")
        
    def log_success(self, message: str):
        """Log success messages (always shown)"""
        print(f"✓ {message}")
        
    def log_task(self, task_num: int, task: str):
        """Log task execution (always shown)"""
        print(f"\n[Task {task_num}] {task}")
        
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test results (always shown)"""
        status = "PASSED" if passed else "FAILED"
        print(f"  Test: {test_name} - {status}")
        if details and self.debug:
            print(f"    Details: {details}")
            
    def log_action(self, action: str, element: str = None):
        """Log actions (simplified in normal mode)"""
        if self.debug:
            if element:
                print(f"[ACTION] {action} on element {element}")
            else:
                print(f"[ACTION] {action}")
        else:
            # In normal mode, just show key actions
            if "Tap" in action or "Type" in action:
                print(f"  → {action}")
                
    def log_api_call(self, api_name: str):
        """Log API calls (only in debug mode)"""
        if self.debug:
            print(f"[API] Calling {api_name}")
            
    def log_adb(self, command: str):
        """Log ADB commands (only in debug mode)"""
        if self.debug:
            print(f"[ADB] {command}")

# Global logger instance
logger = Logger()

def set_debug_mode(debug: bool):
    """Set global debug mode"""
    logger.set_debug(debug)