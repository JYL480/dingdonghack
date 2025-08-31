"""
Orchestrator - Coordinates between Supervisor and Worker agents
"""

import os
import sys
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.supervisor import SupervisorAgent
try:
    from pipeline.supervisor.supervisor_agent_v2 import SupervisorAgentV2
    VISION_SUPERVISOR_AVAILABLE = True
except ImportError:
    VISION_SUPERVISOR_AVAILABLE = False
from pipeline.worker import WorkerAgent
from pipeline.utils.nich_utils import get_model, create_task_folder


class Orchestrator:
    """
    Orchestrator manages the execution flow between Supervisor and Worker
    """
    
    def __init__(self, device="emulator-5554", max_steps=50, use_vision_supervisor=True):
        """
        Initialize Orchestrator with both agents
        
        Args:
            device: Android device ID
            max_steps: Maximum total steps before timeout
            use_vision_supervisor: Use vision-capable supervisor (v2) if available
        """
        print("\n" + "="*60)
        print("INITIALIZING SUPERVISOR-WORKER ARCHITECTURE")
        print("="*60)
        
        # Initialize agents
        if use_vision_supervisor and VISION_SUPERVISOR_AVAILABLE:
            print("[INFO] Using Vision-Capable Supervisor V2")
            self.supervisor = SupervisorAgentV2(model=get_model())  # Vision model
            self.using_vision_supervisor = True
        else:
            print("[INFO] Using Text-Only Supervisor V1")
            self.supervisor = SupervisorAgent(model=get_model())  # Text-only model
            self.using_vision_supervisor = False
            
        self.worker = WorkerAgent(vision_model=get_model(), device=device)  # Vision model
        
        # Configuration
        self.device = device
        self.max_steps = max_steps
        self.current_step = 0
        
        # State tracking
        self.task = ""
        self.subtasks = []
        self.execution_history = []
        self.start_time = None
        self.task_folder = None
        
    def run_task(self, task: str) -> Dict[str, Any]:
        """
        Main execution loop for running a task
        
        Args:
            task: The user's task description
            
        Returns:
            Execution result dictionary
        """
        self.task = task
        self.start_time = datetime.now()
        self.task_folder = create_task_folder(task)
        
        print(f"\n[ORCHESTRATOR] Starting task: {task}")
        print(f"[ORCHESTRATOR] Task folder: ./log/screenshots/{self.task_folder}/")
        
        # Step 1: Decompose task into subtasks
        # If using vision supervisor, provide initial screenshot for better planning
        if self.using_vision_supervisor:
            initial_screen = self.worker.get_current_screen_info()
            self.subtasks = self.supervisor.decompose_task(task, initial_screen.get("screenshot"))
        else:
            self.subtasks = self.supervisor.decompose_task(task)
        
        if not self.subtasks:
            return {
                "status": "error",
                "message": "Failed to decompose task",
                "task": task
            }
        
        # Step 2: Main execution loop
        while self.current_step < self.max_steps:
            self.current_step += 1
            
            # Get progress summary
            progress = self.supervisor.get_progress_summary()
            print(f"\n[ORCHESTRATOR] Step {self.current_step}/{self.max_steps}")
            print(f"[ORCHESTRATOR] Progress: {progress['completed']}/{progress['total_subtasks']} subtasks complete")
            
            # Check if all subtasks complete
            if self._all_subtasks_complete():
                print("\n[ORCHESTRATOR] All subtasks completed successfully!")
                return self._create_success_result()
            
            # Get next action from Supervisor
            worker_feedback = self.execution_history[-1] if self.execution_history else {}
            
            # If using vision supervisor, provide screenshot for better decisions
            if self.using_vision_supervisor:
                current_screen = self.worker.get_current_screen_info()
                supervisor_decision = self.supervisor.get_next_action(
                    worker_feedback, 
                    current_screen.get("screenshot")
                )
            else:
                supervisor_decision = self.supervisor.get_next_action(worker_feedback)
            
            print(f"[ORCHESTRATOR] Supervisor decision: {supervisor_decision.get('action', 'unknown')}")
            
            # Sync subtasks from supervisor (in case refinement happened during get_next_action)
            self.subtasks = self.supervisor.subtasks
            
            # Execute based on Supervisor's decision
            if supervisor_decision["action"] == "complete":
                print("\n[ORCHESTRATOR] Task marked complete by Supervisor")
                return self._create_success_result()
                
            elif supervisor_decision["action"] == "execute":
                # Worker executes the subtask
                subtask = supervisor_decision.get("subtask", {})
                context = supervisor_decision.get("context", {})
                
                # Check if this is a test/verification task
                is_test_task = subtask.get("requires_test", False)
                
                # Fallback: Check keywords if requires_test flag is not set
                if not is_test_task:
                    task_text = subtask.get('task', '').lower()
                    test_keywords = ['check', 'verify', 'test', 'ensure', 'validate', 'confirm', 'locate and verify']
                    if any(keyword in task_text for keyword in test_keywords):
                        is_test_task = True
                        print(f"[ORCHESTRATOR] Warning: Test task detected by keyword fallback: '{subtask.get('task')}'")
                        print(f"[ORCHESTRATOR] Supervisor should have set requires_test=true for this task")
                
                if is_test_task:
                    print(f"\n[ORCHESTRATOR] Executing test task: {subtask.get('task')}")
                    
                    # Get current screen for test
                    current_screen = self.worker.get_current_screen_info()
                    screenshot_path = current_screen.get("screenshot")
                    
                    # Run UI test agent directly
                    from ..utils.nich_utils import run_ui_test_for_subtask
                    
                    test_result = run_ui_test_for_subtask(
                        subtask=subtask,
                        screenshot_path=screenshot_path,
                        device=self.device
                    )
                    
                    # Store test result details in subtask for supervisor validation
                    subtask['test_result_details'] = test_result
                    
                    # Also update the supervisor's copy of the subtask
                    for st in self.supervisor.subtasks:
                        if st.get('id') == subtask.get('id'):
                            st['test_result_details'] = test_result
                            break
                    
                    # Create worker_result format from test result
                    # Let supervisor determine if test passed based on the response
                    worker_result = {
                        "success": test_result.get("success", False),
                        "test_passed": test_result.get("success", False),
                        "subtask": subtask.get("task", ""),
                        "result": test_result.get("result", "Test executed"),
                        "screenshot": screenshot_path,
                        "is_test": True,
                        "test_response": test_result.get("result", "")
                    }
                    
                    # Print as informational - supervisor will determine pass/fail
                    print(f"[ORCHESTRATOR] Test executed")
                    if test_result.get("result"):
                        print(f"[ORCHESTRATOR] Test response: {test_result.get('result')}")
                    
                else:
                    # Regular action task - execute through worker
                    worker_result = self.worker.execute_subtask(subtask, context)
                
                # Record execution
                self.execution_history.append({
                    "step": self.current_step,
                    "subtask": subtask,
                    "worker_result": worker_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Small delay for screen stability
                time.sleep(1)
                
                # ALWAYS have Supervisor check if action was correct
                if self.using_vision_supervisor:
                    # Wait longer for screen transitions (especially for sign-in flows)
                    time.sleep(2)  # Additional wait for screen to fully load
                    
                    print("\n[ORCHESTRATOR] Supervisor validating action...")
                    current_screen = self.worker.get_current_screen_info()
                    screenshot_path = current_screen.get("screenshot")
                    
                    # Ask supervisor if worker did the right thing
                    is_correct = self.supervisor.verify_subtask_completion(
                        subtask,
                        screenshot_path
                    )
                    
                    # Sync subtasks from supervisor after verification (which may trigger refinement)
                    self.subtasks = self.supervisor.subtasks
                    
                    # Consider both UI test and supervisor validation
                    if is_correct and not subtask.get("test_failed", False):
                        # Mark subtask as completed only if both pass
                        subtask["status"] = "completed"
                        print(f"[ORCHESTRATOR] ✓ Subtask {subtask.get('id')} completed successfully")
                    else:
                        if subtask.get("test_failed"):
                            print("[ORCHESTRATOR] Subtask failed UI test, getting correction...")
                        else:
                            print("[ORCHESTRATOR] Worker action incorrect, getting correction...")
                        # Get correction from supervisor
                        correction_action = self.supervisor.get_correction_action()
                        
                        # Execute correction immediately
                        if correction_action["action"] == "execute":
                            print(f"[ORCHESTRATOR] Applying correction: {correction_action.get('context', {}).get('correction', '')}")
                            worker_result = self.worker.execute_subtask(
                                correction_action["subtask"],
                                correction_action["context"]
                            )
                            self.execution_history.append({
                                "step": self.current_step,
                                "subtask": correction_action["subtask"],
                                "worker_result": worker_result,
                                "correction": True,
                                "timestamp": datetime.now().isoformat()
                            })
                
            elif supervisor_decision["action"] == "check_completion":
                # Check if subtask is complete
                subtask = supervisor_decision.get("subtask", {})
                screen_info = self.worker.get_current_screen_info()
                
                is_complete = self.supervisor.check_completion(
                    subtask,
                    {"screenshot": screen_info.get("screenshot"), "elements": screen_info.get("elements_count")}
                )
                
                if is_complete:
                    print(f"[ORCHESTRATOR] Subtask {subtask.get('id')} verified complete")
                
            elif supervisor_decision["action"] == "retry":
                # Retry with hints
                hints = supervisor_decision.get("context", {}).get("hints", "")
                worker_result = self.worker.retry_with_hints(hints)
                
                self.execution_history.append({
                    "step": self.current_step,
                    "action": "retry",
                    "worker_result": worker_result,
                    "timestamp": datetime.now().isoformat()
                })
                
            elif supervisor_decision["action"] == "skip":
                print(f"[ORCHESTRATOR] Skipping subtask: {supervisor_decision.get('reason', '')}")
                
            # Check for timeout
            if self._is_timeout():
                print("\n[ORCHESTRATOR] Execution timeout reached")
                return self._create_timeout_result()
        
        # Max steps reached
        print("\n[ORCHESTRATOR] Maximum steps reached")
        return self._create_timeout_result()
    
    def _all_subtasks_complete(self) -> bool:
        """Check if all subtasks are complete, skipped, or already_completed"""
        for subtask in self.subtasks:
            if subtask["status"] not in ["completed", "skipped", "already_completed"]:
                return False
        return True
    
    def _is_timeout(self) -> bool:
        """Check if execution has timed out"""
        if not self.start_time:
            return False
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return elapsed > 300  # 5 minute timeout
    
    def _create_success_result(self) -> Dict[str, Any]:
        """Create success result dictionary"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        # Count completed and already_completed as completed work
        completed_count = sum(1 for st in self.subtasks if st["status"] in ["completed", "already_completed"])
        skipped_count = sum(1 for st in self.subtasks if st["status"] == "skipped")
        
        return {
            "status": "success",
            "task": self.task,
            "subtasks_total": len(self.subtasks),
            "subtasks_completed": completed_count,
            "subtasks_skipped": skipped_count,
            "steps_taken": self.current_step,
            "execution_time": f"{elapsed_time:.1f} seconds",
            "folder": self.task_folder,
            "subtask_details": self.subtasks
        }
    
    def _create_timeout_result(self) -> Dict[str, Any]:
        """Create timeout/error result dictionary"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        # Count completed and already_completed as completed work
        completed_count = sum(1 for st in self.subtasks if st["status"] in ["completed", "already_completed"])
        
        return {
            "status": "timeout",
            "task": self.task,
            "subtasks_total": len(self.subtasks),
            "subtasks_completed": completed_count,
            "steps_taken": self.current_step,
            "execution_time": f"{elapsed_time:.1f} seconds",
            "folder": self.task_folder,
            "last_subtask": self.supervisor.get_current_subtask()
        }
    
    def reset(self):
        """Reset orchestrator for new task"""
        self.worker.reset()
        
        # Recreate supervisor based on type
        if self.using_vision_supervisor and VISION_SUPERVISOR_AVAILABLE:
            self.supervisor = SupervisorAgentV2(model=get_model())
        else:
            self.supervisor = SupervisorAgent(model=get_model())
            
        self.task = ""
        self.subtasks = []
        self.execution_history = []
        self.current_step = 0
        self.start_time = None
        print("[ORCHESTRATOR] Reset for new task")