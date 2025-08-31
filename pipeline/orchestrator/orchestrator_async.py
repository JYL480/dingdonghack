"""
Async Orchestrator - Coordinates between Supervisor and Worker agents with async support
"""

import os
import sys
import time
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.supervisor import SupervisorAgent
try:
    from pipeline.supervisor.supervisor_agent_v2_async import SupervisorAgentV2Async
    VISION_SUPERVISOR_AVAILABLE = True
except ImportError:
    try:
        from pipeline.supervisor.supervisor_agent_v2 import SupervisorAgentV2
        VISION_SUPERVISOR_AVAILABLE = True
    except ImportError:
        VISION_SUPERVISOR_AVAILABLE = False
        
from pipeline.worker.worker_agent_async import WorkerAgentAsync
from pipeline.utils.nich_utils import get_model, create_task_folder


class OrchestratorAsync:
    """
    Async Orchestrator manages the execution flow between Supervisor and Worker
    """
    
    def __init__(self, device="emulator-5554", max_steps=50, use_vision_supervisor=True):
        """
        Initialize Async Orchestrator with both agents
        
        Args:
            device: Android device ID
            max_steps: Maximum total steps before timeout
            use_vision_supervisor: Use vision-capable supervisor (v2) if available
        """
        print("\n" + "="*60)
        print("INITIALIZING SUPERVISOR-WORKER ARCHITECTURE (ASYNC)")
        print("="*60)
        
        # Initialize agents
        if use_vision_supervisor and VISION_SUPERVISOR_AVAILABLE:
            print("[INFO] Using Vision-Capable Supervisor V2 (Async)")
            try:
                from pipeline.supervisor.supervisor_agent_v2_async import SupervisorAgentV2Async
                self.supervisor = SupervisorAgentV2Async(model=get_model())  # Vision model
            except:
                from pipeline.supervisor.supervisor_agent_v2 import SupervisorAgentV2
                self.supervisor = SupervisorAgentV2(model=get_model())  # Vision model
            self.using_vision_supervisor = True
        else:
            print("[INFO] Using Text-Only Supervisor V1")
            self.supervisor = SupervisorAgent(model=get_model())  # Text-only model
            self.using_vision_supervisor = False
            
        self.worker = WorkerAgentAsync(vision_model=get_model(), device=device)  # Vision model
        
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
        
    async def run_task(self, task: str) -> Dict[str, Any]:
        """
        Main async execution loop for running a task
        
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
            initial_screen = await self.worker.get_current_screen_info()
            if hasattr(self.supervisor, 'decompose_task_async'):
                self.subtasks = await self.supervisor.decompose_task_async(task, initial_screen.get("screenshot"))
            else:
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
                current_screen = await self.worker.get_current_screen_info()
                if hasattr(self.supervisor, 'get_next_action_async'):
                    supervisor_decision = await self.supervisor.get_next_action_async(
                        worker_feedback, 
                        current_screen.get("screenshot")
                    )
                else:
                    supervisor_decision = self.supervisor.get_next_action(
                        worker_feedback, 
                        current_screen.get("screenshot")
                    )
            else:
                supervisor_decision = self.supervisor.get_next_action(worker_feedback)
            
            print(f"[ORCHESTRATOR] Supervisor decision: {supervisor_decision.get('action', 'unknown')}")
            
            # Execute based on Supervisor's decision
            if supervisor_decision["action"] == "complete":
                print("\n[ORCHESTRATOR] Task marked complete by Supervisor")
                return self._create_success_result()
                
            elif supervisor_decision["action"] == "execute":
                # Worker executes the subtask
                subtask = supervisor_decision.get("subtask", {})
                context = supervisor_decision.get("context", {})
                
                worker_result = await self.worker.execute_subtask(subtask, context)
                
                # Record execution
                self.execution_history.append({
                    "step": self.current_step,
                    "subtask": subtask,
                    "worker_result": worker_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Small delay for screen stability
                await asyncio.sleep(1)
                
                # ALWAYS have Supervisor check if action was correct
                if self.using_vision_supervisor:
                    print("\n[ORCHESTRATOR] Supervisor validating worker action...")
                    current_screen = await self.worker.get_current_screen_info()
                    
                    # Ask supervisor if worker did the right thing
                    if hasattr(self.supervisor, 'verify_subtask_completion_async'):
                        is_correct = await self.supervisor.verify_subtask_completion_async(
                            subtask,
                            current_screen.get("screenshot")
                        )
                    else:
                        is_correct = self.supervisor.verify_subtask_completion(
                            subtask,
                            current_screen.get("screenshot")
                        )
                    
                    if is_correct:
                        # Mark subtask as completed
                        subtask["status"] = "completed"
                        print(f"[ORCHESTRATOR] ✓ Subtask {subtask.get('id')} completed successfully")
                    else:
                        print("[ORCHESTRATOR] Worker action incorrect, getting correction...")
                        # Get correction from supervisor
                        if hasattr(self.supervisor, 'get_correction_action_async'):
                            correction_action = await self.supervisor.get_correction_action_async()
                        else:
                            correction_action = self.supervisor.get_correction_action()
                        
                        # Execute correction immediately
                        if correction_action["action"] == "execute":
                            print(f"[ORCHESTRATOR] Applying correction: {correction_action.get('context', {}).get('correction', '')}")
                            worker_result = await self.worker.execute_subtask(
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
                
            elif supervisor_decision["action"] == "retry":
                # Retry with hints
                hints = supervisor_decision.get("context", {}).get("hints", "")
                worker_result = await self.worker.retry_with_hints(hints)
                
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
        """Check if all subtasks are complete or skipped"""
        for subtask in self.subtasks:
            if subtask["status"] not in ["completed", "skipped"]:
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
        
        completed_count = sum(1 for st in self.subtasks if st["status"] == "completed")
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
        
        completed_count = sum(1 for st in self.subtasks if st["status"] == "completed")
        
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
            try:
                from pipeline.supervisor.supervisor_agent_v2_async import SupervisorAgentV2Async
                self.supervisor = SupervisorAgentV2Async(model=get_model())
            except:
                from pipeline.supervisor.supervisor_agent_v2 import SupervisorAgentV2
                self.supervisor = SupervisorAgentV2(model=get_model())
        else:
            self.supervisor = SupervisorAgent(model=get_model())
            
        self.task = ""
        self.subtasks = []
        self.execution_history = []
        self.current_step = 0
        self.start_time = None
        print("[ORCHESTRATOR] Reset for new task")