"""
Async Worker Agent - Executes specific tasks on Android device
"""

import os
import sys
import asyncio
import aiohttp
from typing import Dict, Any
from datetime import datetime

# Add parent directory to path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.utils.nich_utils_twostage import TwoStageNavigator
from pipeline.utils.nich_utils import take_screenshot_adb


class WorkerAgentAsync:
    """
    Async Worker Agent that executes subtasks on Android device using two-stage navigation
    """
    
    def __init__(self, vision_model, device="emulator-5554"):
        """
        Initialize Async Worker with two-stage navigation
        
        Args:
            vision_model: Vision-capable LLM model
            device: Android device ID
        """
        self.device = device
        self.navigator = TwoStageNavigator(vision_model=vision_model, device=device)
        self.execution_count = 0
        self.max_steps = 20
        
    async def execute_subtask(self, subtask: Dict, context: Dict = None) -> Dict[str, Any]:
        """
        Execute a specific subtask asynchronously
        
        Args:
            subtask: The subtask to execute
            context: Optional context/hints from supervisor
            
        Returns:
            Execution result dictionary
        """
        self.execution_count += 1
        
        task_description = subtask.get("task", "")
        element_hint = subtask.get("element_hint", "")
        
        # Add context hints if provided
        if context:
            if context.get("visual_hints"):
                task_description += f" (Hint: {context['visual_hints']})"
            if context.get("correction"):
                task_description = f"Correction needed: {context['correction']}. Original task: {task_description}"
            if context.get("recovery"):
                print(f"[WORKER] Recovery mode: {context.get('reason', '')}")
        
        print(f"\n[WORKER] Executing subtask {subtask.get('id', '')}: {task_description}")
        
        if element_hint:
            print(f"[WORKER] Element hint: {element_hint}")
        
        # Use two-stage navigation asynchronously
        # For now, we'll run synchronous code in executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.navigator.navigate_to_task,
            task_description,
            self.execution_count,
            self.max_steps
        )
        
        return {
            "subtask_id": subtask.get("id"),
            "task": task_description,
            "success": result.get("success", False),
            "decision": result.get("decision", ""),
            "result": result.get("result", ""),
            "screenshot": result.get("screenshot", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_current_screen_info(self) -> Dict[str, Any]:
        """
        Get current screen information asynchronously
        
        Returns:
            Dictionary with screenshot path and element count
        """
        # Take screenshot
        loop = asyncio.get_event_loop()
        screenshot_path = await loop.run_in_executor(
            None,
            take_screenshot_adb,
            self.device,
            f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # For now, return basic info
        # In future, could parse elements asynchronously
        return {
            "screenshot": screenshot_path,
            "elements_count": 0,  # Would need async OmniParser call
            "timestamp": datetime.now().isoformat()
        }
    
    async def retry_with_hints(self, hints: str) -> Dict[str, Any]:
        """
        Retry last action with additional hints asynchronously
        
        Args:
            hints: Additional guidance for retry
            
        Returns:
            Execution result
        """
        print(f"[WORKER] Retrying with hints: {hints}")
        
        # Use navigator's retry mechanism
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.navigator.navigate_to_task,
            f"Retry with guidance: {hints}",
            self.execution_count,
            self.max_steps
        )
        
        return {
            "action": "retry",
            "hints": hints,
            "success": result.get("success", False),
            "result": result.get("result", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset worker for new task"""
        self.execution_count = 0
        self.navigator.reset_context()
        print("[WORKER] Reset for new task")