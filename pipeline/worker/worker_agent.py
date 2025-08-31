"""
Worker Agent - The eyes and hands that execute subtasks
Wraps the existing two-stage navigation system with subtask focus
"""

import os
import sys
import time
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import existing two-stage system
from pipeline.utils.nich_utils_twostage import two_stage_navigation
from pipeline.utils.nich_utils import (
    capture_and_parse_screen,
    create_deployment_state,
    get_model
)


class WorkerAgent:
    """
    Worker Agent responsible for:
    - Executing specific subtasks
    - Using vision to analyze screens
    - Performing actions via ADB
    - Reporting results back to Supervisor
    """
    
    def __init__(self, vision_model=None, device="emulator-5554"):
        """
        Initialize Worker with vision model
        
        Args:
            vision_model: LangChain ChatOpenAI model with vision
            device: Android device ID
        """
        self.vision_model = vision_model or get_model()
        self.device = device
        self.current_state = None
        self.last_screenshot = None
        self.last_elements = None
        
    def execute_subtask(self, subtask: Dict, context: Dict = None) -> Dict[str, Any]:
        """
        Execute a single subtask using the two-stage system
        
        Args:
            subtask: Subtask dictionary from Supervisor
            context: Context hints from Supervisor
            
        Returns:
            Execution result dictionary
        """
        print(f"\n[WORKER] Executing subtask {subtask.get('id', 0)}: {subtask.get('task', '')}")
        
        # Initialize or update state for this subtask
        if not self.current_state:
            self.current_state = create_deployment_state(
                task=subtask.get("task", ""),
                device=self.device
            )
        else:
            # Update task to current subtask
            self.current_state["task"] = subtask.get("task", "")
            
        # Add supervisor context to state
        if context:
            self.current_state["supervisor_context"] = context
            if context.get("hints"):
                print(f"[WORKER] Using hints: {context['hints']}")
        
        # Capture and parse screen first
        self.current_state = capture_and_parse_screen(self.current_state)
        
        if not self.current_state.get("current_page", {}).get("screenshot"):
            return {
                "success": False,
                "error": "Failed to capture screen",
                "subtask": subtask.get("task", "")
            }
        
        # Store for later reference
        self.last_screenshot = self.current_state["current_page"]["screenshot"]
        self.last_elements = self.current_state["current_page"].get("elements_data", [])
        
        # Execute using enhanced two-stage navigation
        try:
            # Modify the task in state to be more specific for this subtask
            original_task = self.current_state["task"]
            
            # Add context hints to the task if provided
            subtask_text = subtask.get('task', '')
            
            # Get specific value from subtask or context
            specific_value = subtask.get('specific_value') or (context and context.get('specific_value'))
            
            # Special handling for typing tasks
            if ("type" in subtask_text.lower() or "enter" in subtask_text.lower()) and specific_value:
                # Make sure we use the specific value provided
                subtask_text = f"Type '{specific_value}' in the email or phone input field"
                print(f"[WORKER] Will type specific value: {specific_value}")
            elif "type" in subtask_text.lower() and "field" in subtask_text.lower():
                # Extract number from original task if available
                import re
                number_match = re.search(r'\b\d{6,15}\b', original_task)
                if number_match:
                    specific_value = number_match.group()
                    subtask_text = f"Type '{specific_value}' in the email or phone input field"
                    print(f"[WORKER] Extracted value to type: {specific_value}")
            
            if context and context.get("hints"):
                self.current_state["task"] = f"{subtask_text}. Hint: {context['hints']}"
            else:
                self.current_state["task"] = subtask_text
            
            # Run two-stage navigation for this subtask
            self.current_state = two_stage_navigation(self.current_state, self.vision_model)
            
            # Restore original task
            self.current_state["task"] = original_task
            
            # Extract results
            execution_status = self.current_state.get("execution_status", "unknown")
            action_history = self.current_state.get("action_history", [])
            last_action = action_history[-1] if action_history else {}
            
            result = {
                "success": execution_status == "success",
                "subtask": subtask.get("task", ""),
                "decision": last_action.get("vision_decision", ""),
                "action": last_action.get("action", ""),
                "result": last_action.get("result", ""),
                "screenshot": self.last_screenshot,
                "elements_count": len(self.last_elements),
                "confidence": last_action.get("confidence", "medium")
            }
            
            # Add reasoning if available
            if last_action.get("reasoning"):
                result["reasoning"] = last_action["reasoning"]
            
            print(f"[WORKER] Result: {result['result']}")
            
            return result
            
        except Exception as e:
            print(f"[WORKER] Error executing subtask: {e}")
            return {
                "success": False,
                "error": str(e),
                "subtask": subtask.get("task", ""),
                "screenshot": self.last_screenshot
            }
    
    def retry_with_hints(self, hints: str) -> Dict[str, Any]:
        """
        Retry the last action with additional hints
        
        Args:
            hints: Specific hints from Supervisor
            
        Returns:
            Execution result
        """
        print(f"[WORKER] Retrying with hints: {hints}")
        
        if not self.current_state:
            return {
                "success": False,
                "error": "No previous state to retry"
            }
        
        # Add hints to context
        self.current_state["supervisor_context"] = {"hints": hints, "retry": True}
        
        # Force new screenshot for retry
        self.current_state["current_page"] = {}
        self.current_state = capture_and_parse_screen(self.current_state)
        
        # Execute with hints
        self.current_state = two_stage_navigation(self.current_state, self.vision_model)
        
        action_history = self.current_state.get("action_history", [])
        last_action = action_history[-1] if action_history else {}
        
        return {
            "success": self.current_state.get("execution_status") == "success",
            "decision": last_action.get("vision_decision", ""),
            "result": last_action.get("result", ""),
            "screenshot": self.last_screenshot
        }
    
    def get_current_screen_info(self) -> Dict[str, Any]:
        """
        Get information about the current screen
        
        Returns:
            Screen information dictionary
        """
        if not self.current_state:
            self.current_state = create_deployment_state("", self.device)
        
        # Capture current screen
        self.current_state = capture_and_parse_screen(self.current_state)
        
        return {
            "screenshot": self.current_state.get("current_page", {}).get("screenshot"),
            "elements_count": len(self.current_state.get("current_page", {}).get("elements_data", [])),
            "screen_type": self.current_state.get("screen_context", {}).get("type", "unknown")
        }
    
    def reset(self):
        """Reset worker state for new task"""
        self.current_state = None
        self.last_screenshot = None
        self.last_elements = None
        print("[WORKER] State reset for new task")