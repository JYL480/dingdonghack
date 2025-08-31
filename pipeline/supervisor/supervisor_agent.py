"""
Supervisor Agent - The brain that plans and tracks task execution
"""

import json
import re
from typing import Dict, List, Any, Optional
from langchain.schema.messages import HumanMessage, SystemMessage

from .prompts import (
    TASK_DECOMPOSITION_PROMPT,
    PROGRESS_TRACKING_PROMPT,
    STUCK_RECOVERY_PROMPT,
    COMPLETION_CHECK_PROMPT
)


class SupervisorAgent:
    """
    Supervisor Agent responsible for:
    - Task decomposition into subtasks
    - Progress tracking
    - Stuck detection and recovery
    - Completion checking decisions
    """
    
    def __init__(self, model):
        """
        Initialize Supervisor with a text-only model (no vision needed)
        
        Args:
            model: LangChain ChatOpenAI model (text-only, e.g., gpt-4-turbo)
        """
        self.model = model
        self.subtasks = []
        self.current_subtask_idx = 0
        self.stuck_counter = 0
        self.action_history = []
        self.max_retries = 3
        self.completion_check_frequency = 3  # Check every 3 subtasks
        
    def decompose_task(self, task: str) -> List[Dict[str, Any]]:
        """
        Break down a high-level task into executable subtasks
        
        Args:
            task: The user's task description
            
        Returns:
            List of subtask dictionaries
        """
        print("\n[SUPERVISOR] Decomposing task into subtasks...")
        
        prompt = TASK_DECOMPOSITION_PROMPT.format(task=task)
        
        messages = [
            SystemMessage(content="You are a task planning assistant. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            result = json.loads(content)
            self.subtasks = result.get("subtasks", [])
            
            # Add status to each subtask
            for subtask in self.subtasks:
                subtask["status"] = "pending"
                subtask["attempts"] = 0
            
            print(f"[SUPERVISOR] Created {len(self.subtasks)} subtasks:")
            for st in self.subtasks:
                print(f"  {st['id']}. {st['task']}")
            
            return self.subtasks
            
        except Exception as e:
            print(f"[SUPERVISOR] Error decomposing task: {e}")
            # Fallback to simple single task
            self.subtasks = [{
                "id": 1,
                "task": task,
                "type": "action",
                "status": "pending",
                "attempts": 0
            }]
            return self.subtasks
    
    def get_current_subtask(self) -> Optional[Dict[str, Any]]:
        """Get the current subtask to execute"""
        for subtask in self.subtasks:
            if subtask["status"] == "pending" or subtask["status"] == "in_progress":
                return subtask
        return None
    
    def get_next_action(self, worker_feedback: Dict) -> Dict[str, Any]:
        """
        Decide what to do next based on worker feedback
        
        Args:
            worker_feedback: Results from the worker's last action
            
        Returns:
            Dictionary with next action decision
        """
        current_subtask = self.get_current_subtask()
        
        if not current_subtask:
            return {
                "action": "complete",
                "reason": "All subtasks completed"
            }
        
        # Add to action history
        self.action_history.append(worker_feedback)
        
        # Check if stuck (same action repeated)
        if self._is_stuck():
            self.stuck_counter += 1
            if self.stuck_counter >= self.max_retries:
                return self._handle_stuck(current_subtask)
        else:
            self.stuck_counter = 0
        
        # If no worker feedback yet, start execution
        if not worker_feedback:
            return {
                "action": "execute",
                "subtask": current_subtask,
                "context": self._get_context_hints(current_subtask)
            }
        
        # Analyze worker result and decide
        prompt = PROGRESS_TRACKING_PROMPT.format(
            subtask=current_subtask["task"],
            result=json.dumps(worker_feedback),
            recent_actions=self._format_recent_actions(),
            stuck_counter=self.stuck_counter
        )
        
        messages = [
            SystemMessage(content="You are a task progress manager. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            decision = json.loads(content)
            
            # Process decision
            action = decision.get("action", "continue")
            
            if action == "continue":
                current_subtask["status"] = "completed"
                print(f"[SUPERVISOR] Subtask {current_subtask['id']} completed")
                # Get next subtask
                next_subtask = self.get_current_subtask()
                if next_subtask:
                    return {
                        "action": "execute",
                        "subtask": next_subtask,
                        "context": self._get_context_hints(next_subtask)
                    }
                else:
                    return {"action": "complete", "reason": "All subtasks done"}
                    
            elif action == "retry":
                current_subtask["attempts"] += 1
                return {
                    "action": "execute",
                    "subtask": current_subtask,
                    "context": {
                        "hints": decision.get("hints", ""),
                        "retry_number": current_subtask["attempts"]
                    }
                }
                
            elif action == "skip":
                current_subtask["status"] = "skipped"
                print(f"[SUPERVISOR] Skipping subtask {current_subtask['id']}: {decision.get('reason')}")
                next_subtask = self.get_current_subtask()
                if next_subtask:
                    return {
                        "action": "execute",
                        "subtask": next_subtask,
                        "context": self._get_context_hints(next_subtask)
                    }
                    
            elif action == "check":
                return {
                    "action": "check_completion",
                    "subtask": current_subtask
                }
                
        except Exception as e:
            print(f"[SUPERVISOR] Error in decision making: {e}")
            # Default to retry with basic hints
            return {
                "action": "execute",
                "subtask": current_subtask,
                "context": {"hints": "Try a different approach"}
            }
    
    def _is_stuck(self) -> bool:
        """Check if worker is stuck repeating same actions"""
        if len(self.action_history) < 3:
            return False
        
        # Check last 3 actions
        last_actions = self.action_history[-3:]
        decisions = [a.get("decision", "") for a in last_actions]
        
        # If all same decision, we're stuck
        return len(set(decisions)) == 1
    
    def _handle_stuck(self, current_subtask: Dict) -> Dict[str, Any]:
        """Handle stuck situation with recovery strategy"""
        print(f"[SUPERVISOR] Stuck detected on subtask {current_subtask['id']}")
        
        prompt = STUCK_RECOVERY_PROMPT.format(
            action_history=self._format_recent_actions(),
            subtask=current_subtask["task"],
            attempts=current_subtask["attempts"]
        )
        
        messages = [
            SystemMessage(content="You are a recovery strategist. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            strategy = json.loads(content)
            
            if strategy.get("strategy") == "skip":
                current_subtask["status"] = "skipped"
                print(f"[SUPERVISOR] Skipping stuck subtask: {strategy.get('skip_reason')}")
                next_subtask = self.get_current_subtask()
                if next_subtask:
                    return {
                        "action": "execute",
                        "subtask": next_subtask,
                        "context": self._get_context_hints(next_subtask)
                    }
            else:
                # Try alternative approach
                return {
                    "action": "execute",
                    "subtask": current_subtask,
                    "context": {
                        "hints": strategy.get("hints", ""),
                        "alternative": strategy.get("alternative_action", "")
                    }
                }
                
        except Exception as e:
            print(f"[SUPERVISOR] Error in stuck recovery: {e}")
            # Skip if can't recover
            current_subtask["status"] = "skipped"
            return {
                "action": "execute",
                "subtask": self.get_current_subtask(),
                "context": {}
            }
    
    def _get_context_hints(self, subtask: Dict) -> Dict[str, Any]:
        """Generate context hints for a subtask"""
        hints = {}
        
        task_lower = subtask["task"].lower()
        
        # Provide specific hints based on subtask type
        if "chrome" in task_lower:
            hints["app_hint"] = "Chrome has multicolored circular icon (red, green, yellow, blue)"
        elif "sign" in task_lower and "click" in task_lower:
            hints["sign_in_hint"] = "Look for 'Sign in' button or link, usually in top right or center of page"
        elif "type" in task_lower:
            if "81291126" in task_lower or "mobile" in task_lower or "phone" in task_lower:
                hints["input_hint"] = "Type in the 'Email or phone' field - tap it first to focus, then type the number"
            else:
                hints["input_hint"] = "Tap the input field first to focus, then type"
        elif "next" in task_lower or "submit" in task_lower:
            hints["button_hint"] = "Look for blue 'NEXT' or 'Submit' button, usually at bottom right"
        elif "search" in task_lower:
            hints["search_hint"] = "Look for search bar or address bar at top"
        
        return hints
    
    def _format_recent_actions(self) -> str:
        """Format recent action history for context"""
        if not self.action_history:
            return "No actions yet"
        
        recent = self.action_history[-5:]  # Last 5 actions
        formatted = []
        for i, action in enumerate(recent):
            formatted.append(f"{i+1}. {action.get('decision', 'unknown')} -> {action.get('result', 'unknown')}")
        
        return "\n".join(formatted)
    
    def check_completion(self, subtask: Dict, evidence: Dict) -> bool:
        """
        Check if a subtask is complete
        
        Args:
            subtask: The subtask to check
            evidence: Evidence from worker (screenshot, result, etc.)
            
        Returns:
            True if complete, False otherwise
        """
        prompt = COMPLETION_CHECK_PROMPT.format(
            subtask=subtask["task"],
            evidence=json.dumps(evidence),
            has_screenshot=evidence.get("screenshot") is not None
        )
        
        messages = [
            SystemMessage(content="You are a completion checker. Output only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.model.invoke(messages)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            result = json.loads(content)
            is_complete = result.get("complete", False)
            
            if is_complete:
                subtask["status"] = "completed"
                print(f"[SUPERVISOR] Subtask {subtask['id']} verified complete")
            else:
                print(f"[SUPERVISOR] Subtask incomplete: {result.get('reason')}")
            
            return is_complete
            
        except Exception as e:
            print(f"[SUPERVISOR] Error checking completion: {e}")
            return False
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        total = len(self.subtasks)
        completed = sum(1 for st in self.subtasks if st["status"] == "completed")
        skipped = sum(1 for st in self.subtasks if st["status"] == "skipped")
        
        return {
            "total_subtasks": total,
            "completed": completed,
            "skipped": skipped,
            "in_progress": self.get_current_subtask(),
            "progress_percentage": (completed / total * 100) if total > 0 else 0
        }